__all__ = [
    'OnnxSum',
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


class OnnxSum(nn.Module):  # pylint: disable=missing-docstring
    def forward(self, *input_tensors: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(input_tensors) == 1:
            # If there is a single element, return it (no op).
            # Also, no need for manually building the ONNX node.
            return input_tensors[0]

        shapes = [t.shape for t in input_tensors]
        broadcast_shape = torch.broadcast_shapes(*shapes)
        output = torch.zeros(broadcast_shape, dtype=input_tensors[0].dtype, device=input_tensors[0].device)

        for y in input_tensors:
            output.add_(y)

        if torch.onnx.is_in_onnx_export():
            return _SumExportToOnnx.set_output_and_apply(output, *input_tensors)

        return output


class _SumExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Sum', *args, outputs=1)


@add_converter(operation_type='Sum', version=8)
@add_converter(operation_type='Sum', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxSum(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
