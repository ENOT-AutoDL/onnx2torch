__all__ = ['OnnxNot', 'OnnxLogical']

from typing import Optional

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import SkipTorchTracing
from onnx2torch.common import old_style_broadcast
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode

_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'Or': torch.logical_or,
    'And': torch.logical_and,
    'Xor': torch.logical_xor,
}


class OnnxNot(nn.Module):

    def _do_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            with SkipTorchTracing():
                output = self._do_forward(input_tensor)
                return _NotExportToOnnx.set_output_and_apply(output, input_tensor)

        return self._do_forward(input_tensor)


class _NotExportToOnnx(CustomExportToOnnx):

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Not', *args, outputs=1)


class OnnxLogical(nn.Module):
    def __init__(self, operation_type: str, broadcast: Optional[int] = None,  axis: Optional[int] = None):
        super().__init__()
        self.broadcast = broadcast
        self.axis = axis

        self.logic_op_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

    def forward(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor):
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
