__all__ = [
    'OnnxMatMul',
]

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OnnxToTorchModule
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import onnx_mapping_from_node


class OnnxMatMul(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        # TODO: Figure out a way to simplify this code
        if x.dim() == 3 and y.dim() == 3 and x.shape[0] == y.shape[0]:
            out = torch.bmm(x, y)
        else:
            try:
                out = torch.matmul(x, y)
            except:  # pylint: disable=bare-except
                out = torch.mm(x, y)
        return out


@add_converter(operation_type='MatMul', version=1)
@add_converter(operation_type='MatMul', version=9)
@add_converter(operation_type='MatMul', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxMatMul(node.name),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
