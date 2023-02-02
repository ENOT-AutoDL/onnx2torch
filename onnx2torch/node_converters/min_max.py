__all__ = [
    'OnnxMinMax',
]

import torch

from onnx2torch.node_converters.base_element_wise import OnnxBaseElementWise
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxMinMax(OnnxBaseElementWise):  # pylint: disable=missing-docstring
    _OPERATORS = {
        'Min': torch.amin,
        'Max': torch.amax,
    }

    def __init__(self, op_type: str):
        super().__init__(op_type=op_type)
        self._operator = self._OPERATORS[op_type]

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
