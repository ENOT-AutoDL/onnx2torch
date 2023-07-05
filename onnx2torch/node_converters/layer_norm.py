__all__ = [
    'OnnxLayerNorm',
]

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_shape_from_value_info
from onnx2torch.utils.common import onnx_mapping_from_node

AXIS_DEFAULT_VALUE = -1
EPSILON_DEFAULT_VALUE = 1e-5


class OnnxLayerNorm(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int, epsilon: float):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon

    def forward(  # pylint: disable=missing-function-docstring
        self,
        inputs: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized_shape = inputs.shape[self.axis :]
        return F.layer_norm(
            input=inputs,
            normalized_shape=normalized_shape,
            weight=scale,
            bias=bias,
            eps=self.epsilon,
        )


@add_converter(operation_type='LayerNormalization', version=17)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes

    axis = node_attributes.get('axis', AXIS_DEFAULT_VALUE)
    epsilon = node_attributes.get('epsilon', EPSILON_DEFAULT_VALUE)

    if all(value_name in graph.initializers for value_name in node.input_values[1:]):
        input_value_info = graph.value_info[node.input_values[0]]
        input_shape = get_shape_from_value_info(input_value_info)

        torch_module = nn.LayerNorm(
            normalized_shape=input_shape[axis:],
            eps=epsilon,
            elementwise_affine=True,
        )

        scale_value_name = node.input_values[1]
        bias_value_name = node.input_values[2] if len(node.input_values) > 2 else None

        with torch.no_grad():
            torch_module.weight.data = graph.initializers[scale_value_name].to_torch()
            if bias_value_name is not None:
                torch_module.bias.data = graph.initializers[bias_value_name].to_torch()

        onnx_mapping = OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values)
    else:
        input_value_info = graph.value_info[node.input_values[0]]
        input_shape = get_shape_from_value_info(input_value_info)
        torch_module = OnnxLayerNorm(axis=axis, epsilon=epsilon)
        onnx_mapping = onnx_mapping_from_node(node)

    return OperationConverterResult(torch_module=torch_module, onnx_mapping=onnx_mapping)
