__all__ = [
    'OnnxLayerNorm',
]

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


class OnnxLayerNorm(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int, epsilon: float, stash_type: int):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.stash_type = stash_type

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_data: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return F.layer_norm(
            input_data,
            normalized_shape,
            weight = weight,
            bias = bias,
            eps = self.epsilon
        )


@add_converter(operation_type='LayerNormalization', version=17)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    axis = node_attributes.get('axis', -1)
    epsilon = node_attributes.get('epsilon', 1e-5)
    stash_type = node_attributes.get('stash_type', 1)
    if all(value_name in graph.initializers for value_name in node.input_values[1:]):
        input_value_info = graph.value_info[node.input_values[0]]
        input_shape = get_shape_from_value_info(input_value_info)

        scale_value_name = node.input_values[1]
        bias_value_name = node.input_values[2]

        scale = graph.initializers[scale_value_name].to_torch()
        torch_module = nn.LayerNorm(
            input_shape[axis],
            eps=epsilon,
            elementwise_affine=True
        )

        with torch.no_grad():
            torch_module.weight.data = graph.initializers[scale_value_name].to_torch()
            torch_module.bias.data = graph.initializers[bias_value_name].to_torch()

        onnx_mapping = OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values)
    else:
        torch_module = OnnxLayerNorm(momentum=momentum, axis=axis, epsilon=epsilon, stash_type=stash_type)
        onnx_mapping = onnx_mapping_from_node(node)

    return OperationConverterResult(torch_module=torch_module, onnx_mapping=onnx_mapping)
