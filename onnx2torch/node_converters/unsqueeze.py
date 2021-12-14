__all__ = ['OnnxUnsqueeze']

from typing import Optional

import torch
from torch import nn

from onnx2torch.common import OnnxMapping
from onnx2torch.common import OperationConverterResult
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxUnsqueeze(nn.Module):

    def __init__(self, axes: Optional[torch.Tensor] = None):
        super().__init__()
        self.axes = axes

    def forward(self, input_tensor: torch.Tensor, axes: Optional[torch.Tensor] = None):
        if axes is not None and self.axes is not None:
            raise ValueError(
                'Static axes are specified for Unsqueeze and dynamic axes are passed in forward. '
                'if you want to use dynamic axes you should not pass static axes during module creation. '
            )

        if axes is None and self.axes is None:
            raise ValueError(
                'Static axes or dynamic axes not provided. '
                'If you dont specified static axes during module creation, you must pass it in forward. '
            )

        axes = axes if axes is not None else self.axes
        axes, _ = torch.sort(axes)
        for i in axes:
            input_tensor = torch.unsqueeze(input_tensor, i)

        return input_tensor


@add_converter(operation_type='Unsqueeze', version=1)
@add_converter(operation_type='Unsqueeze', version=11)
@add_converter(operation_type='Unsqueeze', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    input_values = [node.input_values[0]]
    axes_value_name = node.input_values[1] if len(node.input_values) > 1 else None

    if axes_value_name is not None:
        axes = graph.initializers[axes_value_name].to_torch()
    else:
        axes = torch.tensor(node.attributes['axes'], dtype=torch.long)

    torch_module = OnnxUnsqueeze(
        axes=axes,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=tuple(input_values),
            outputs=node.output_values,
        ),
    )
