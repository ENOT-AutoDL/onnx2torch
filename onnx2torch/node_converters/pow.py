__all__ = [
    'OnnxPow',
    'OnnxSqrt',
]

from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import old_style_broadcast
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxPow(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, broadcast: Optional[int] = None, axis: Optional[int] = None):
        super().__init__()
        self.axis = axis
        self.broadcast = broadcast

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        exponent: torch.Tensor,
    ) -> torch.Tensor:
        if self.broadcast == 1 and self.axis is not None:
            exponent = old_style_broadcast(input_tensor, exponent, self.axis)

        return torch.pow(input_tensor, exponent)


class OnnxSqrt(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.sqrt(input_tensor)


@add_converter(operation_type='Pow', version=1)
@add_converter(operation_type='Pow', version=7)
@add_converter(operation_type='Pow', version=12)
@add_converter(operation_type='Pow', version=13)
@add_converter(operation_type='Pow', version=15)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxPow(
            broadcast=node.attributes.get('broadcast', None),
            axis=node.attributes.get('axis', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Sqrt', version=1)
@add_converter(operation_type='Sqrt', version=6)
@add_converter(operation_type='Sqrt', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxSqrt(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
