__all__ = [
    'OnnxRoiAlign',
]

from enum import Enum
from typing import Any
from typing import Dict

import torch
from torch import nn
from torchvision.ops import roi_align

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class CoordinateTransformationModeOnnxAttr(Enum):
    """
    Representation of new attribute in 16 opset: `coordinate_transformation_mode`.

    Allowed values are `half_pixel` and `output_half_pixel`.
    Use the value `half_pixel` to pixel shift the input coordinates by -0.5 (the recommended behavior).
    Use the value `output_half_pixel` to omit the pixel shift for the inputs
    (use this for a backward-compatible behavior).
    """

    HALF_PIXEL = 'half_pixel'
    OUTPUT_HALF_PIXEL = 'output_half_pixel'


CTMOnnxAttr = CoordinateTransformationModeOnnxAttr  # Type alias.


class OnnxRoiAlign(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(
        self,
        coordinate_transformation_mode: CTMOnnxAttr = CTMOnnxAttr.HALF_PIXEL,
        mode: str = 'avg',
        output_height: int = 1,
        output_width: int = 1,
        sampling_ratio: int = 0,
        spatial_scale: float = 1.0,
    ):
        super().__init__()

        self._coordinate_transformation_mode = coordinate_transformation_mode

        if mode != 'avg':
            raise NotImplementedError(f'"{mode}" roi align mode is not implemented.')
        self._mode = mode

        self._output_height = output_height
        self._output_width = output_width
        self._sampling_ratio = sampling_ratio
        self._spatial_scale = spatial_scale

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        onnx_attrs: Dict[str, Any] = {
            'mode_s': self._mode,
            'output_height_i': self._output_height,
            'output_width_i': self._output_width,
            'sampling_ratio_i': self._sampling_ratio,
            'spatial_scale_f': self._spatial_scale,
        }

        if opset_version < 16:
            if self._coordinate_transformation_mode != CTMOnnxAttr.OUTPUT_HALF_PIXEL:
                raise ValueError(
                    'RoiAlign from opset 10 does not support coordinate_transform_mode != "output_half_pixel"'
                    f', got {self._coordinate_transformation_mode.value}'
                )
            return onnx_attrs

        onnx_attrs['coordinate_transformation_mode_s'] = self._coordinate_transformation_mode.value
        return onnx_attrs

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        rois: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        def _forward():
            fixed_batch_indices = batch_indices.unsqueeze(1).to(rois.dtype)
            batched_rois = torch.cat([fixed_batch_indices, rois], dim=1)
            output_size = self._output_height, self._output_width
            sampling_ratio = self._sampling_ratio
            spatial_scale = self._spatial_scale

            return roi_align(
                input=input_tensor,
                boxes=batched_rois,
                output_size=output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio,
                aligned=self._coordinate_transformation_mode == CTMOnnxAttr.HALF_PIXEL,
            )

        if torch.onnx.is_in_onnx_export():
            onnx_attrs = self._onnx_attrs(get_onnx_version())
            return DefaultExportToOnnx.export(_forward, 'RoiAlign', input_tensor, rois, batch_indices, onnx_attrs)

        return _forward()


def converter_schema(  # pylint: disable=missing-function-docstring, unused-argument
    node: OnnxNode,
    graph: OnnxGraph,
    default_ctm: str,
) -> OperationConverterResult:
    node_attributes = node.attributes
    coordinate_transformation_mode = CTMOnnxAttr(node_attributes.get('coordinate_transformation_mode', default_ctm))
    mode = node_attributes.get('mode', 'avg')
    output_height = node_attributes.get('output_height', 1)
    output_width = node_attributes.get('output_width', 1)
    sampling_ratio = node_attributes.get('sampling_ratio', 0)
    spatial_scale = node_attributes.get('spatial_scale', 1.0)

    return OperationConverterResult(
        torch_module=OnnxRoiAlign(
            coordinate_transformation_mode=coordinate_transformation_mode,
            mode=mode,
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type='RoiAlign', version=10)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return converter_schema(node=node, graph=graph, default_ctm='output_half_pixel')


@add_converter(operation_type='RoiAlign', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return converter_schema(node=node, graph=graph, default_ctm='half_pixel')
