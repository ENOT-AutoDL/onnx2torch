__all__ = [
    'OnnxNot',
    'OnnxLogical',
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
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport

_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'Or': torch.logical_or,
    'And': torch.logical_and,
    'Xor': torch.logical_xor,
}


class OnnxNot(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        forward_lambda = lambda: torch.logical_not(input_tensor)
        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(forward_lambda, 'Not', input_tensor, {})

        return forward_lambda()


class OnnxLogical(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, operation_type: str, broadcast: Optional[int] = None, axis: Optional[int] = None):
        super().__init__()
        self.broadcast = broadcast
        self.axis = axis

        self.logic_op_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

    def forward(  # pylint: disable=missing-function-docstring
        self, first_tensor: torch.Tensor, second_tensor: torch.Tensor
    ):
        if self.broadcast == 1 and self.axis is not None:
            second_tensor = old_style_broadcast(first_tensor, second_tensor, self.axis)

        return self.logic_op_function(first_tensor, second_tensor)


@add_converter(operation_type='Xor', version=1)
@add_converter(operation_type='Xor', version=7)
@add_converter(operation_type='And', version=1)
@add_converter(operation_type='And', version=7)
@add_converter(operation_type='Or', version=1)
@add_converter(operation_type='Or', version=7)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxLogical(
            operation_type=node.operation_type,
            broadcast=node.attributes.get('broadcast', None),
            axis=node.attributes.get('axis', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Not', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxNot(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
