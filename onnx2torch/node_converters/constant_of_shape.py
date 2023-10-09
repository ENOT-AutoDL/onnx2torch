__all__ = [
    'OnnxConstantOfShape',
]

from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxConstantOfShape(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, value: Optional[torch.Tensor] = None):
        super().__init__()

        if value is None:
            value = torch.tensor(0.0, dtype=torch.float32)

        if value.numel() != 1:
            raise ValueError('parameter "value" must be scalar')

        self.value: torch.Tensor
        self.register_buffer('value', value)

    def forward(self, shape: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        fill_value = self.value.item()

        return torch.full(
            size=torch.Size(shape),
            fill_value=int(fill_value) if isinstance(fill_value, bool) else fill_value,
            dtype=self.value.dtype,
            device=self.value.device,
        )


@add_converter(operation_type='ConstantOfShape', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes

    if 'value' in node_attributes:
        value = node_attributes['value'].to_torch()
    else:
        value = None

    return OperationConverterResult(
        torch_module=OnnxConstantOfShape(value=value),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
