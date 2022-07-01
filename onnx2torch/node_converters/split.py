__all__ = [
    'OnnxSplit',
    'OnnxSplit13',
]

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxSplit13(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, num_splits: int, axis: int = 0):
        super().__init__()

        self.axis = axis
        self.num_splits = num_splits

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        split: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if split is None:
            axis_len = input_tensor.shape[self.axis]
            split_size_or_sections = axis_len // self.num_splits
        else:
            split_size_or_sections = split.tolist()

        return torch.split(input_tensor, split_size_or_sections, dim=self.axis)


class OnnxSplit(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, num_splits: int, axis: int = 0, split: Optional[List[int]] = None):
        super().__init__()

        self.axis = axis
        self.num_splits = num_splits
        self.split = split

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if self.split is None:
            axis_len = input_tensor.shape[self.axis]
            split_size_or_sections = axis_len // self.num_splits
        else:
            split_size_or_sections = self.split

        return torch.split(input_tensor, split_size_or_sections, dim=self.axis)


@add_converter(operation_type='Split', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    num_splits = len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplit13(axis=axis, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Split', version=11)
@add_converter(operation_type='Split', version=2)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    split = node.attributes.get('split', None)
    num_splits = len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplit(axis=axis, split=split, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
