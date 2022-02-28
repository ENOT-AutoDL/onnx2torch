__all__ = ['OnnxCeil', 'OnnxExp', 'OnnxFloor', 'OnnxLog']

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxCeil(nn.Module):

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.ceil(input_tensor)


class OnnxExp(nn.Module):

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(input_tensor)


class OnnxFloor(nn.Module):

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.floor(input_tensor)


class OnnxLog(nn.Module):

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(input_tensor)


@add_converter(operation_type='Ceil', version=13)
@add_converter(operation_type='Ceil', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=OnnxCeil(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Exp', version=6)
@add_converter(operation_type='Exp', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxExp(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Floor', version=13)
@add_converter(operation_type='Floor', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=OnnxFloor(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Log', version=13)
@add_converter(operation_type='Log', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxLog(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
