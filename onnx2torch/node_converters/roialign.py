__all__ = ['OnnxRoiAlign']

import torch
from torch import nn
from torchvision.ops import roi_align

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxRoiAlign(nn.Module, OnnxToTorchModule):

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

        self._output_size = (output_height, output_width)
        self._sampling_ratio = sampling_ratio
        self._spatial_scale = spatial_scale

    def forward(
            self,
            input_tensor: torch.Tensor,
            rois: torch.Tensor,
            batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        batched_rois = torch.concat([batch_indices.unsqueeze(1).to(rois.dtype), rois], dim=1)

        return roi_align(
            input=input_tensor,
            boxes=batched_rois,
            output_size=self._output_size,
            spatial_scale=self._spatial_scale,
            sampling_ratio=self._sampling_ratio,
            aligned=False,
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
