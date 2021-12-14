__all__ = ['OnnxShape']

from typing import Optional

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxShape(nn.Module):

    def __init__(self, start: Optional[int] = None, end: Optional[int] = None):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            input_tensor.shape[self.start:self.end],
            device=input_tensor.device,
        )


@add_converter(operation_type='Shape', version=1)
@add_converter(operation_type='Shape', version=13)
@add_converter(operation_type='Shape', version=15)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxShape(
            start=node.attributes.get('start', None),
            end=node.attributes.get('end', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
