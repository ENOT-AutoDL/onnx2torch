__all__ = [
    'OnnxSum',
]

import torch
import torch._C as torch_C

from onnx2torch.node_converters.base_element_wise import OnnxBaseElementWise
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx


class OnnxSum(OnnxBaseElementWise):  # pylint: disable=missing-docstring
    def __init__(self):
        super().__init__(_SumExportToOnnx)

    def apply_reduction(self, *tensors: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        broadcast_shape = self._broadcast_shape(*tensors)

        output = torch.zeros(broadcast_shape, dtype=tensors[0].dtype, device=tensors[0].device)
        for y in tensors:
            output.add_(y)

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
