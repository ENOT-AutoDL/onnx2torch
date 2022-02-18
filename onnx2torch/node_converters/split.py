__all__ = ['OnnxSplit', 'OnnxSplit13']
from typing import Optional, List

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


def get_split(axis_len: int, split: Optional[torch.Tensor], num_splits: int):
    if split is None:
        quotient = axis_len // num_splits
        remainder = axis_len % num_splits
        if remainder == 0:
            split = [quotient] * num_splits
        else:
            split = [quotient] * num_splits + [remainder]

    return torch.tensor(split).tolist()


class OnnxSplit13(nn.Module):
    def __init__(self, num_splits: int, axis: int = 0):
        super().__init__()
        self.axis = axis
        self.num_splits = num_splits

    def forward(self, input_tensor: torch.Tensor, split: Optional[torch.Tensor] = None) -> torch.Tensor:
        axis_len = input_tensor.shape[self.axis]
        split = get_split(axis_len, split, self.num_splits)
        return torch.split(input_tensor, split, self.axis)


class OnnxSplit(nn.Module):
    def __init__(self, num_splits: int, axis: int = 0, split: Optional[List[int]] = None):
        super().__init__()
        self._split = OnnxSplit13(num_splits, axis)
        self.axis = axis
        self.split = split

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._split(input_tensor, self.split)


class OnnxSplitV1(OnnxSplit):

    def forward(self, input_tensor: torch.Tensor, split: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._split(input_tensor, split or self.split)


@add_converter(operation_type='Split', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    axis = node.attributes.get('axis', 0)
    num_splits = len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplit13(axis=axis, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Split', version=11)
@add_converter(operation_type='Split', version=2)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    axis = node.attributes.get('axis', 0)
    split = node.attributes.get('split', None)
    num_splits = len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplit(axis=axis, split=split, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Split', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    axis = node.attributes.get('axis', 0)
    split = node.attributes.get('split', None)
    num_splits = len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplitV1(axis=axis, split=split, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
