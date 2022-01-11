__all__ = ['OnnxReduce']

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode

_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'ReduceMax': torch.max,
    'ReduceMean': torch.mean,
    'ReduceSum': torch.sum,
    'ReduceMin': torch.min,
    'ReduceProd': torch.prod,
}


class OnnxReduce(nn.Module):

    def __init__(
            self,
            operation_type: str,
            axes: List[int],
            keepdims: int = 1,
            noop_with_empty_axes: int = 0,
    ):
        super().__init__()
        self.math_op_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

        if axes is not None:
            axes = sorted(axes)

        self.keepdims = keepdims == 1
        self.noop_with_empty_axes = noop_with_empty_axes == 1
        self.axes = axes

    def forward(self, input_tensor: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if axes is not None:
            self.axes, _ = torch.sort(axes)
            self.axes = self.axes.tolist()

        if self.axes is None:
            if self.noop_with_empty_axes is True:
                return input_tensor

            if self.keepdims is False:
                return self.math_op_function(input_tensor)

            self.axes = list(range(input_tensor.dim()))

        result = input_tensor
        for passed_dims, axis in enumerate(self.axes):
            result = self.math_op_function(
                result,
                dim=axis if self.keepdims else axis - passed_dims,
                keepdim=self.keepdims,
            )
            if isinstance(result, tuple):
                result = result[0]

        return result


@add_converter(operation_type='ReduceMax', version=1)
@add_converter(operation_type='ReduceMax', version=11)
@add_converter(operation_type='ReduceMax', version=12)
@add_converter(operation_type='ReduceMax', version=13)
@add_converter(operation_type='ReduceMean', version=1)
@add_converter(operation_type='ReduceMean', version=11)
@add_converter(operation_type='ReduceMean', version=13)
@add_converter(operation_type='ReduceMin', version=1)
@add_converter(operation_type='ReduceMin', version=11)
@add_converter(operation_type='ReduceMin', version=12)
@add_converter(operation_type='ReduceMin', version=13)
@add_converter(operation_type='ReduceProd', version=1)
@add_converter(operation_type='ReduceProd', version=11)
@add_converter(operation_type='ReduceProd', version=13)
@add_converter(operation_type='ReduceSum', version=1)
@add_converter(operation_type='ReduceSum', version=11)
@add_converter(operation_type='ReduceSum', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    axes = node_attributes.get('axes', None)
    keepdims = node_attributes.get('keepdims', 1)
    noop_with_empty_axes = node_attributes.get('noop_with_empty_axes', 0)

    return OperationConverterResult(
        torch_module=OnnxReduce(
            operation_type=node.operation_type,
            axes=axes,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
