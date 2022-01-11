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

LINEAR_MODES = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}


def _get_torch_align_corners(mode: str, coordinate_transformation_mode: str) -> Optional[bool]:
    align_corners = coordinate_transformation_mode == 'align_corners'
    if mode == "nearest":
        return None
    return align_corners


def _dimension_mode(mode: str, dim_size: int) -> str:
    if dim_size <= 0 and dim_size > 3:
        raise RuntimeError('Input tensor has to have from 1 to 3 dimensions.')

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
        raise RuntimeError(f'Got unexpected mode for interpolation ({mode})')


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
            roi: torch.Tensor = torch.tensor([]),
            scales: torch.Tensor = torch.tensor([]),
            sizes: torch.Tensor = torch.tensor([]),
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, input_tensor.dim() - 2)
        if roi.numel() != 0:
            warnings.warn('Roi only takes effect when coordinate_transformation_mode is "tf_crop_and_resize".')
            warnings.warn('Pytorch\'s interpolate doesn\'t use roi. Result might differ."')

        # In onnx scales and sizes in format [n, c, d, h, w]
        # but in torch only [d, h, w]
        sizes, scales = sizes.tolist(), scales.tolist()
        input_shape = list(input_tensor.shape)
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
            warnings.warn('Pytorch\'s linear interpolate and onnx resize significantly differ!"')

    def forward(
            self,
            input_tensor: torch.Tensor,
            scales: torch.Tensor,
    ) -> torch.Tensor:
        self.mode = _dimension_mode(self.mode, input_tensor.dim() - 2)

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
            input_tensor,
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
            'Coordinate transformation modes "tf_crop_and_resize" and "tf_half_pixel_for_nn" are not implemented.'
        )

    if mode == 'nearest':
        if nearest_mode != 'floor':
            raise NotImplementedError('Only floor nearest mode is implemented.')
        if coordinate_transformation_mode not in ('asymmetric', 'align_corners'):
            raise ValueError(
                f'''Pytorch\'s nearest neighbor interpolation uses asymmetric mode. 
                But got {coordinate_transformation_mode}.'''
            )
    else:
        if coordinate_transformation_mode == 'asymmetric':
            raise ValueError('For linear and cubic interpolation in pytorch '
                             '"half_pixel", "pytorch_half_pixel" and "align_corners" are valid')

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
