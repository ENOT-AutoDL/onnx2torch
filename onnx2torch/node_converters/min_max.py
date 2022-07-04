__all__ = [
    'OnnxMinMax',
]

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx

_OPERATORS = {
    'Min': torch.min,
    'Max': torch.max,
}


class _MinExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Min', *args, outputs=1)


class _MaxExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Max', *args, outputs=1)


_CUSTOM_ONNX_EXPORT_CLASS = {
    'Min': _MinExportToOnnx,
    'Max': _MaxExportToOnnx,
}


class OnnxMinMax(nn.Module):  # pylint: disable=missing-docstring
    def __init__(self, operation_type: str):
        super().__init__()
        self._operator = _OPERATORS[operation_type]
        self._onnx_export_class = _CUSTOM_ONNX_EXPORT_CLASS[operation_type]

    def forward(self, *input_tensors: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(input_tensors) == 1:
            # If there is a single element, return it (no op).
            # Also, no need for manually building the ONNX node.
            return input_tensors[0]

        shapes = [t.shape for t in input_tensors]
        broadcast_shape = torch.broadcast_shapes(*shapes)

        broadcast_tensors = [t.broadcast_to(broadcast_shape) for t in input_tensors]
        stacked_tensors = torch.stack(broadcast_tensors)

        output = self._operator(stacked_tensors, dim=0)[0]

        if torch.onnx.is_in_onnx_export():
            return self._onnx_export_class.set_output_and_apply(output, *input_tensors)

        return output


@add_converter(operation_type='Min', version=8)
@add_converter(operation_type='Min', version=12)
@add_converter(operation_type='Min', version=13)
@add_converter(operation_type='Max', version=8)
@add_converter(operation_type='Max', version=12)
@add_converter(operation_type='Max', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxMinMax(node.operation_type),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
