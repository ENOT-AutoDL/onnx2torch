__all__ = [
    'OnnxInstanceNorm',
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

_IN_CLASS_FROM_SPATIAL_RANK = {
    0: nn.InstanceNorm1d,
    1: nn.InstanceNorm1d,
    2: nn.InstanceNorm2d,
    3: nn.InstanceNorm3d,
}


class OnnxInstanceNorm(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, momentum: float, epsilon: float):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_data: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return F.instance_norm(
            input=input_data,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=bias,
            use_input_stats=True,
            momentum=self.momentum,
            eps=self.epsilon,
        )


@add_converter(operation_type='InstanceNormalization', version=1)
@add_converter(operation_type='InstanceNormalization', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    epsilon = node_attributes.get('epsilon', 1e-5)
    momentum = 0.1

    if all(value_name in graph.initializers for value_name in node.input_values[1:]):
        input_value_info = graph.value_info[node.input_values[0]]
        input_shape = get_shape_from_value_info(input_value_info)
        spatial_rank = len(input_shape) - 2
        try:
            in_class = _IN_CLASS_FROM_SPATIAL_RANK[spatial_rank]
        except KeyError as exc:
            raise NotImplementedError(
                f'InstanceNorm operation with spatial rank == {spatial_rank} is not implemented'
            ) from exc

        scale_value_name = node.input_values[1]
        bias_value_name = node.input_values[2]

        scale = graph.initializers[scale_value_name].to_torch()
        torch_module = in_class(
            num_features=scale.size()[0],
            eps=epsilon,
            momentum=momentum,
            affine=True,
            track_running_stats=False,
        )
        with torch.no_grad():
            torch_module.weight.data = graph.initializers[scale_value_name].to_torch()
            torch_module.bias.data = graph.initializers[bias_value_name].to_torch()

        onnx_mapping = OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values)
    else:
        torch_module = OnnxInstanceNorm(momentum=momentum, epsilon=epsilon)
        onnx_mapping = onnx_mapping_from_node(node)

    return OperationConverterResult(torch_module=torch_module, onnx_mapping=onnx_mapping)
