__all__ = [
    'OnnxMinMax',
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

_OPERATORS = {
    'Min': torch.amin,
    'Max': torch.amax,
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


class OnnxMinMax(OnnxBaseElementWise):  # pylint: disable=missing-docstring
    def __init__(self, operation_type: str):
        super().__init__(_CUSTOM_ONNX_EXPORT_CLASS[operation_type])
        self._operator = _OPERATORS[operation_type]

    def apply_reduction(self, *tensors: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        broadcast_shape = self._broadcast_shape(*tensors)
        broadcast_tensors = [t.broadcast_to(broadcast_shape) for t in tensors]
        stacked_tensors = torch.stack(broadcast_tensors)
        output = self._operator(stacked_tensors, dim=0)
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
