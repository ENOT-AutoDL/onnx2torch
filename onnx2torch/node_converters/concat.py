__all__ = [
    'OnnxConcat',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxConcat(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, *input_tensors) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.cat(input_tensors, self.axis)


@add_converter(operation_type='Concat', version=4)
@add_converter(operation_type='Concat', version=11)
@add_converter(operation_type='Concat', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    torch_module = OnnxConcat(
        axis=axis,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
