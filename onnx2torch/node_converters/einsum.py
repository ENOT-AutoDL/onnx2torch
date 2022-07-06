__all__ = [
    'OnnxEinsum',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxEinsum(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, equation: str):
        super().__init__()
        self.equation = equation

    def forward(self, *args):  # pylint: disable=missing-function-docstring
        return torch.einsum(self.equation, *args)


@add_converter(operation_type='Einsum', version=12)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxEinsum(
            equation=node.attributes['equation'],
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
