__all__ = ['OnnxRoiAlign']

from typing import Tuple

import torch
import torch._C as torch_C
from torch import nn
from torchvision.ops import roi_align

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxRoiAlign(nn.Module, OnnxToTorchModuleWithCustomExport):

    def __init__(
            self,
            mode: str = 'avg',
            output_height: int = 1,
            output_width: int = 1,
            sampling_ratio: int = 0,
            spatial_scale: float = 1.0,
    ):
        super().__init__()
        if mode != 'avg':
            raise NotImplementedError(f'"{mode}" roi align mode is not implemented.')

        self._output_height = output_height
        self._output_width = output_width
        self._sampling_ratio = sampling_ratio
        self._spatial_scale = spatial_scale

    @staticmethod
    def _do_forward(
            input_tensor: torch.Tensor,
            rois: torch.Tensor,
            batch_indices: torch.Tensor,
            output_size: Tuple[int, int],
            sampling_ratio: int,
            spatial_scale: float,
    ) -> torch.Tensor:

        batch_indices = batch_indices.unsqueeze(1).to(rois.dtype)
        batched_rois = torch.concat([batch_indices, rois], dim=1)

        return roi_align(
            input=input_tensor,
            boxes=batched_rois,
            output_size=output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=False,
        )

    def forward(
            self,
            input_tensor: torch.Tensor,
            rois: torch.Tensor,
            batch_indices: torch.Tensor,

    ) -> torch.Tensor:

        output = self._do_forward(
            input_tensor=input_tensor,
            rois=rois,
            batch_indices=batch_indices,
            output_size=(self._output_height, self._output_width),
            sampling_ratio=self._sampling_ratio,
            spatial_scale=self._spatial_scale,
        )
        if torch.onnx.is_in_onnx_export():
            args = [
                input_tensor,
                rois,
                batch_indices,
                self._output_height,
                self._output_width,
                self._sampling_ratio,
                self._spatial_scale
            ]
            return _RoiAlignExportToOnnx.set_output_and_apply(output, *args)

        return output


class _RoiAlignExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        input_args = args[:3]
        output_height, output_width, sampling_ratio, spatial_scale = args[3:]
        return graph.op(
            'RoiAlign',
            *input_args,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
            outputs=1,
        )


@add_converter(operation_type='RoiAlign', version=10)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    mode = node_attributes.get('mode', 'avg')
    output_height = node_attributes.get('output_height', 1)
    output_width = node_attributes.get('output_width', 1)
    sampling_ratio = node_attributes.get('sampling_ratio', 0)
    spatial_scale = node_attributes.get('spatial_scale', 1.0)

    return OperationConverterResult(
        torch_module=OnnxRoiAlign(
            mode=mode,
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )
