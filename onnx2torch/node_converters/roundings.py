__all__ = [
    'OnnxRound',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

_TORCH_ROUND_FROM_ONNX_TYPE = {
    'Ceil': torch.ceil,
    'Floor': torch.floor,
    'Round': torch.round,
}


class OnnxRound(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, round_type: str):
        super().__init__()
        self.round_function = _TORCH_ROUND_FROM_ONNX_TYPE[round_type]

    def forward(self, input_tensor: torch.Tensor):  # pylint: disable=missing-function-docstring
        return self.round_function(input_tensor)


@add_converter(operation_type='Ceil', version=13)
@add_converter(operation_type='Ceil', version=6)
@add_converter(operation_type='Floor', version=13)
@add_converter(operation_type='Floor', version=6)
@add_converter(operation_type='Round', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxRound(node.operation_type),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
