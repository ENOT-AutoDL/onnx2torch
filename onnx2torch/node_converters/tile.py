__all__ = ['OnnxTile']

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxTile(nn.Module):

    def forward(self, input_tensor: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
        # torch.tile(input_tensor, repeats) is not supported for exporting
        return input_tensor.repeat(torch.Size(repeats))


@add_converter(operation_type='Tile', version=6)
@add_converter(operation_type='Tile', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument

    return OperationConverterResult(
        torch_module=OnnxTile(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
