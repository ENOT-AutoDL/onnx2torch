__all__ = ['OnnxResize']

import warnings
from typing import Optional

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode

_TORCH_MODES = {
    ('nearest', 1): 'nearest',
    ('nearest', 2): 'nearest',
    ('nearest', 3): 'nearest',
    ('linear', 1): 'linear',
    ('linear', 2): 'bilinear',
    ('linear', 3): 'trilinear',
    ('cubic', 2): 'bicubic',
}


def _get_torch_align_corners(mode: str, coordinate_transformation_mode: str) -> Optional[bool]:
    if mode == 'nearest':
        return None

    return coordinate_transformation_mode == 'align_corners'


def _dimension_mode(mode: str, dim_size: int) -> str:
    torch_mode = _TORCH_MODES.get((mode, dim_size), None)
    if torch_mode is None:
        raise NotImplementedError(f'{dim_size}D input is not implemented for "{mode}" mode.')

    return torch_mode


class OnnxResize(nn.Module):

    def __init__(
            self,
            mode: str = 'nearest',
            align_corners: Optional[bool] = None,
    ):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(
            self,
            input_tensor: torch.Tensor,
            roi: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None,
            sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, input_tensor.dim() - 2)
        if roi is not None:
            raise NotImplementedError('roi logic is not implemented.')

        # Format of onnx scales and sizes is [n, c, d, h, w]
        # But in torch only [d, h, w] (without batch and channel dimensions)
        input_shape = list(input_tensor.shape)
        if sizes is not None:
            sizes = sizes.tolist()
            if input_shape[:2] != sizes[:2]:
                raise NotImplementedError('Pytorch\'s interpolate cannot resize channel or batch dimensions.')
            sizes = sizes[2:]
        elif scales is not None:
            scales = scales.tolist()
            if scales[:2] != [1, 1]:
                raise NotImplementedError('Pytorch\'s interpolate cannot scale channel or batch dimensions.')
            scales = scales[2:]
        else:
            raise ValueError('One of scales or sizes should be defined.')

        return torch.nn.functional.interpolate(
            input_tensor,
            size=sizes,
            scale_factor=scales,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class OnnxResizeV10(nn.Module):

    def __init__(self, mode: str = 'nearest'):
        super().__init__()
        self._resize = OnnxResize(mode=mode)

    def forward(self, input_tensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return self._resize(input_tensor, scales=scales)


@add_converter(operation_type='Resize', version=10)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    mode = node_attributes.get('mode', 'nearest')

    torch_module = OnnxResizeV10(mode=mode)
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type='Resize', version=11)
@add_converter(operation_type='Resize', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    coordinate_transformation_mode = node_attributes.get('coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a = node_attributes.get('cubic_coeff_a', -0.75)
    exclude_outside = node_attributes.get('exclude_outside', 0)
    extrapolation_value = node_attributes.get('extrapolation_value', 0.0)
    mode = node_attributes.get('mode', 'nearest')
    nearest_mode = node_attributes.get('nearest_mode', 'round_prefer_floor')

    if mode == 'nearest':
        if nearest_mode != 'floor':
            warnings.warn(
                'Pytorch\'s nearest neighbor interpolate uses the "floor" nearest_mode. '
                'For others modes, the results might differ significantly!'
            )

        if coordinate_transformation_mode != 'asymmetric':
            warnings.warn(
                'Pytorch\'s nearest neighbor interpolation uses "asymmetric" coordinate_transformation_mode. '
                'For others modes, the results might differ significantly!'
            )
    else:
        if coordinate_transformation_mode not in ['pytorch_half_pixel', 'half_pixel']:
            warnings.warn(
                'For linear and cubic interpolation in "asymmetric" and "align_corners" coordinate_transformation_mode'
                'results might differ significantly!'
            )

    if cubic_coeff_a != -0.75:
        warnings.warn('With a cubic coefficient value other than 0.75, the results might differ significantly!')

    if exclude_outside != 0:
        warnings.warn('With a exclude outside value other than 0, the results might differ significantly!')

    if extrapolation_value != 0.0:
        warnings.warn('With a extrapolation value other than 0.0, the results might differ significantly!')

    return OperationConverterResult(
        torch_module=OnnxResize(
            mode=mode,
            align_corners=_get_torch_align_corners(mode, coordinate_transformation_mode),
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )
