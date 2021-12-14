__all__ = ['OnnxReduceMax']

from typing import List

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxReduceMax(nn.Module):

    def __init__(self, axes: List[int], keepdims: int = 1):
        super().__init__()

        if axes is not None:
            axes = sorted(axes)

        self.keepdims = keepdims == 1
        self.axes = axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # exporting torch.amax to onnx is not supported now
        # return input_tensor.amax(self.axes, keepdim=self.keepdims)

        if self.axes is None:
            if not self.keepdims:
                return input_tensor.max()

            self.axes = list(range(input_tensor.dim()))

        result = input_tensor
        for passed_dims, axis in enumerate(self.axes):
            result, _ = torch.max(
                result,
                dim=axis if self.keepdims else axis - passed_dims,
                keepdim=self.keepdims,
            )

        return result


@add_converter(operation_type='ReduceMax', version=11)
@add_converter(operation_type='ReduceMax', version=12)
@add_converter(operation_type='ReduceMax', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    axes = node_attributes.get('axes', None)
    keepdims = node_attributes.get('keepdims', 1)

    return OperationConverterResult(
        torch_module=OnnxReduceMax(axes=axes, keepdims=keepdims),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
