__all__ = ['OnnxSqueeze']

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.common import OnnxMapping
from onnx2torch.common import OperationConverterResult
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxSqueeze(nn.Module):

    def __init__(self, axes: Optional[List[int]] = None):
        super().__init__()
        self.axes = axes

    def forward(self, input_tensor: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        axes = axes if axes is not None else self.axes
        if axes is None:
            return torch.squeeze(input_tensor)

        axes, _ = torch.sort(axes)
        for axes_id in axes:
            input_tensor = torch.squeeze(input_tensor, dim=axes_id)
        return input_tensor


@add_converter(operation_type='Squeeze', version=11)
@add_converter(operation_type='Squeeze', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    input_values = [node.input_values[0]]
    axes_value_name = node.input_values[1] if len(node.input_values) > 1 else None

    if axes_value_name is not None:
        axes = graph.initializers.get(axes_value_name, None)
        if axes is not None:
            axes = axes.to_torch()
        else:
            input_values.append(node.input_values[1])
    else:
        axes = torch.tensor(node.attributes.get('axes', None), dtype=torch.long)

    return OperationConverterResult(
        torch_module=OnnxSqueeze(axes=axes),
        onnx_mapping=OnnxMapping(
            inputs=tuple(input_values),
            outputs=node.output_values,
        ),
    )
