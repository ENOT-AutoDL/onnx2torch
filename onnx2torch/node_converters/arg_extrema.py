# pylint: disable=missing-docstring
__all__ = [
    'OnnxArgExtremumOld',
    'OnnxArgExtremum',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

DEFAULT_AXIS = 0
DEFAULT_KEEPDIMS = 1
DEFAULT_SELECT_LAST_INDEX = 0

_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'ArgMax': torch.argmax,
    'ArgMin': torch.argmin,
}


class OnnxArgExtremumOld(nn.Module, OnnxToTorchModule):
    def __init__(self, operation_type: str, axis: int, keepdims: int):
        super().__init__()
        self.axis = axis
        self.keepdims = bool(keepdims)
        self.extremum_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.extremum_function(data, dim=self.axis, keepdim=self.keepdims)


class OnnxArgExtremum(nn.Module, OnnxToTorchModule):
    def __init__(self, operation_type: str, axis: int, keepdims: int, select_last_index: int):
        super().__init__()
        self.axis = axis
        self.keepdims = bool(keepdims)
        self.select_last_index = bool(select_last_index)
        self.extremum_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.select_last_index:
            # torch's argmax does not handle the select_last_index attribute from Onnx.
            # We flip the data, call the normal argmax, then map it back to the original
            flipped = torch.flip(data, dims=[self.axis])

            extremum_index_flipped = self.extremum_function(flipped, dim=self.axis, keepdim=self.keepdims)
            extremum_index_original = data.size(dim=self.axis) - 1 - extremum_index_flipped
            return extremum_index_original

        return self.extremum_function(data, dim=self.axis, keepdim=self.keepdims)


@add_converter(operation_type='ArgMax', version=12)
@add_converter(operation_type='ArgMax', version=13)
@add_converter(operation_type='ArgMin', version=12)
@add_converter(operation_type='ArgMin', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    return OperationConverterResult(
        torch_module=OnnxArgExtremum(
            operation_type=node.operation_type,
            axis=node.attributes.get('axis', DEFAULT_AXIS),
            keepdims=node.attributes.get('keepdims', DEFAULT_KEEPDIMS),
            select_last_index=node.attributes.get('select_last_index', DEFAULT_SELECT_LAST_INDEX),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='ArgMax', version=11)
@add_converter(operation_type='ArgMin', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    return OperationConverterResult(
        torch_module=OnnxArgExtremumOld(
            operation_type=node.operation_type,
            axis=node.attributes.get('axis', DEFAULT_AXIS),
            keepdims=node.attributes.get('keepdims', DEFAULT_KEEPDIMS),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
