__all__ = [
    'OnnxFunction',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

# Exporting from pytorch to onnx operators atanh, asinh, acosh, cosh, sinh are not supported
_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'Abs': torch.abs,
    'Acos': torch.acos,
    'Asin': torch.asin,
    'Atan': torch.atan,
    'Cos': torch.cos,
    'Exp': torch.exp,
    'Log': torch.log,
    'Sign': torch.sign,
    'Sin': torch.sin,
    'Tan': torch.tan,
    'Tanh': torch.tanh,
}


class OnnxFunction(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, function_type: str):
        super().__init__()
        self.function = _TORCH_FUNCTION_FROM_ONNX_TYPE[function_type]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return self.function(input_tensor)


@add_converter(operation_type='Abs', version=13)
@add_converter(operation_type='Abs', version=6)
@add_converter(operation_type='Acos', version=7)
@add_converter(operation_type='Asin', version=7)
@add_converter(operation_type='Atan', version=7)
@add_converter(operation_type='Cos', version=7)
@add_converter(operation_type='Exp', version=6)
@add_converter(operation_type='Exp', version=13)
@add_converter(operation_type='Log', version=13)
@add_converter(operation_type='Log', version=6)
@add_converter(operation_type='Sign', version=13)
@add_converter(operation_type='Sign', version=9)
@add_converter(operation_type='Sin', version=7)
@add_converter(operation_type='Tan', version=7)
@add_converter(operation_type='Tanh', version=13)
@add_converter(operation_type='Tanh', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxFunction(node.operation_type),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
