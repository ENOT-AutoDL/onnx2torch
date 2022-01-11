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

VALID_COORDINATE_TRANSFORM_MODES = ['asymmetric', 'align_corners', 'pytorch_half_pixel', 'half_pixel']


DIMENSION_MODE = {
    'linear': {1: 'linear', 2: 'bilinear', 3: 'trilinear'},
    'cubic': {1: None, 2: 'bicubic', 3: None},
    'nearest': {dim: 'nearest' for dim in range(3)},
}


def _get_torch_align_corners(mode: str, coordinate_transformation_mode: str) -> Optional[bool]:
    align_corners = coordinate_transformation_mode == 'align_corners'
    if mode == "nearest" and align_corners is False:
        return None
    return align_corners


def _dimension_mode(mode: str, data_dim) -> str:
    torch_mode = DIMENSION_MODE.get(mode, None)
    if torch_mode is None:
        return mode
    torch_mode = torch_mode.get(data_dim, None)
    if torch_mode is None:
        raise NotImplementedError(f"{data_dim}D input is not implemented for {mode} mode.")
    return torch_mode


class OnnxResize(nn.Module):

    def __init__(
            self,
            coordinate_transformation_mode: str = 'asymmetric',
            mode: str = 'nearest',
    ):
        super().__init__()
        self.mode = mode
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.align_corners = _get_torch_align_corners(self.mode, self.coordinate_transformation_mode)

    def forward(
            self,
            x: torch.Tensor,
            roi: torch.Tensor = torch.tensor([]),
            scales: torch.Tensor = torch.tensor([]),
            sizes: torch.Tensor = torch.tensor([]),
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, x.dim() - 2)
        if roi.numel() != 0:
            warnings.warn('Roi only takes effect when coordinate_transformation_mode is "tf_crop_and_resize".')
            warnings.warn('Pytorch\'s interpolate doesn\'t use roi. Result might differ."')

        # In onnx scales and sizes in format [n, c, d, h, w]
        # but in torch only [d, h, w]
        sizes, scales = sizes.tolist(), scales.tolist()
        input_shape = list(x.shape)
        if input_shape[:2] == sizes[:2]:
            sizes = sizes[2:]
        elif scales[:2] == [1, 1]:
            scales = scales[2:]
        elif len(sizes) == 0 and len(scales) == 0:
            raise ValueError('One of scales or sizes should be defined.')
        else:
            raise NotImplementedError('Pytorch\'s interpolate cannot scale channel or batch dimensions.')

        if len(scales) == 0:
            scales = None
        elif len(sizes) == 0:
            sizes = None
        else:
            raise ValueError('Only one of the two, scales or sizes, needs to be defined.')

        return torch.nn.functional.interpolate(
            x,
            size=sizes,
            scale_factor=scales,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class OnnxResizeV10(nn.Module):

    def __init__(self, mode: str = 'nearest'):
        super().__init__()
        self.mode = mode
        if self.mode == 'linear':
            warnings.warn('Pytorch\'s linear interpolate and onnx resize might differ!"')

    def forward(
            self,
            x: torch.Tensor,
            scales: torch.Tensor,
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, x.dim() - 2)

        # In onnx scales and sizes in format [n, c, d, h, w]
        # but in torch only [d, h, w]
        scales = scales.tolist()
        if scales[:2] == [1, 1]:
            scales = scales[2:]
        elif len(scales) == 0:
            raise ValueError('Scales should be defined.')
        else:
            raise NotImplementedError('Pytorch\'s interpolate cannot scale channel or batch dimensions.')

        return torch.nn.functional.interpolate(
            x,
            scale_factor=tuple((float(x) for x in scales)),
            mode=self.mode,
        )


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

    if coordinate_transformation_mode not in VALID_COORDINATE_TRANSFORM_MODES:
        raise NotImplementedError(
            f'Coordinate transformation mode "tf_crop_and_resize" and "tf_half_pixel_for_nn" are not implemented.'
        )

    if mode == 'nearest':
        if nearest_mode != 'floor':
            raise NotImplementedError('Only floor nearest mode is implemented.')
        if coordinate_transformation_mode != 'asymmetric':
            raise ValueError(
                f'''Pytorch\'s nearest neighbor used asymmetric mode. 
                But got {coordinate_transformation_mode}.'''
            )
    else:
        if coordinate_transformation_mode == 'asymmetric':
            raise ValueError('For linear and cubic "half_pixel", "pytorch_half_pixel" and "align_corners" are valid')
        warnings.warn('Results might differ!')

    if cubic_coeff_a != -0.75:
        raise NotImplementedError('Only -0.75 value for cubic_coeff_a is implemented in pytorch\'s interpolate.')

    if exclude_outside != 0:
        raise NotImplementedError('Only 0 value for exclude_outside is implemented.')

    if extrapolation_value != 0.0:
        raise NotImplementedError('Only 0.0 value for extrapolation_value is implemented.')

    torch_module = OnnxResize(
        coordinate_transformation_mode=coordinate_transformation_mode,
        mode=mode,
    )
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node),
    )
