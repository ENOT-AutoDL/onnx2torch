__all__ = ['OnnxResize']

import warnings

import torch
from torch import nn
from typing import Optional

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode

LINEAR_MODES = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}


def _get_torch_align_corners(mode: str, coordinate_transformation_mode: str) -> Optional[bool]:
    align_corners = coordinate_transformation_mode == 'align_corners'
    if mode == "nearest":
        return None
    return align_corners


def _dimension_mode(mode: str, dim_size: int) -> str:
    if dim_size not in [1, 2, 3]:
        raise RuntimeError('Input tensor for resize has to have from 1 to 3 dimensions.')

    if mode == 'nearest':
        return mode

    elif 'cubic' in mode:
        if dim_size == 2:
            return 'bicubic'
        else:
            raise NotImplementedError(f"{dim_size}D input is not implemented for {mode} mode.")

    elif 'linear' in mode:
        return LINEAR_MODES[dim_size]

    else:
        raise RuntimeError(f'Got unexpected mode for interpolation ({mode}).')


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
            input_tensor: torch.Tensor,
            roi: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None,
            sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, input_tensor.dim() - 2)
        if roi is not None:
            warnings.warn('Roi only takes effect when coordinate_transformation_mode is "tf_crop_and_resize".')
            warnings.warn('Pytorch\'s interpolate doesn\'t use roi. Result might differ.')

        if scales is not None and sizes is not None:
            raise ValueError('Only one of the two, scales or sizes, needs to be defined.')

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
        self.mode = mode
        if self.mode == 'linear':
            warnings.warn('Pytorch\'s linear interpolate and onnx linear resize might differ significantly!')

    def forward(
            self,
            input_tensor: torch.Tensor,
            scales: torch.Tensor,
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, input_tensor.dim() - 2)

        # Format of onnx scales is [n, c, d, h, w]
        # But in torch only [d, h, w] (without batch and channel dimensions)
        scales = scales.tolist()
        if scales[:2] == [1, 1]:
            scales = scales[2:]
        elif len(scales) == 0:
            raise ValueError('Scales should be defined.')
        else:
            raise NotImplementedError('Pytorch\'s interpolate cannot scale channel or batch dimensions.')

        return torch.nn.functional.interpolate(
            input_tensor,
            scale_factor=scales,
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
            coordinate_transformation_mode=coordinate_transformation_mode,
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )
