__all__ = ['OnnxGlobalAveragePool']

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxGlobalAveragePool(nn.Module):

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=no-self-use
        x_dims = list(range(2, len(input_tensor.shape)))
        return torch.mean(input_tensor, dim=x_dims, keepdim=True)


@add_converter(operation_type='GlobalAveragePool', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxGlobalAveragePool(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
