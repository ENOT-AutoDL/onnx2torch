__all__ = [
    'OnnxBatchNorm',
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

_BN_CLASS_FROM_SPATIAL_RANK = {
    0: nn.BatchNorm1d,
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


class OnnxBatchNorm(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, momentum: float, epsilon: float):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_data: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        return F.batch_norm(
            input_data,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.epsilon,
        )


@add_converter(operation_type='BatchNormalization', version=15)
@add_converter(operation_type='BatchNormalization', version=14)
@add_converter(operation_type='BatchNormalization', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    epsilon = node_attributes.get('epsilon', 1e-5)
    momentum = 1 - node_attributes.get('momentum', 0.9)  # See PyTorch documentation for batch norm.

    if all(value_name in graph.initializers for value_name in node.input_values[1:]) and len(node.output_values) == 1:
        input_value_info = graph.value_info[node.input_values[0]]
        input_shape = get_shape_from_value_info(input_value_info)
        spatial_rank = len(input_shape) - 2
        try:
            bn_class = _BN_CLASS_FROM_SPATIAL_RANK[spatial_rank]
        except KeyError as exc:
            raise NotImplementedError(
                f'BatchNorm operation with spatial rank == {spatial_rank} is not implemented'
            ) from exc

        scale_value_name = node.input_values[1]
        bias_value_name = node.input_values[2]
        mean_value_name = node.input_values[3]
        var_value_name = node.input_values[4]

        scale = graph.initializers[scale_value_name].to_torch()
        torch_module = bn_class(
            num_features=scale.size()[0],
            eps=epsilon,
            momentum=momentum,
        )
        with torch.no_grad():
            torch_module.running_mean.data = graph.initializers[mean_value_name].to_torch()
            torch_module.running_var.data = graph.initializers[var_value_name].to_torch()
            torch_module.weight.data = scale
            torch_module.bias.data = graph.initializers[bias_value_name].to_torch()

        onnx_mapping = OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        )
    else:
        if len(node.output_values) != 1:
            raise NotImplementedError('BatchNorm operation with mean/var output is not implemented')

        torch_module = OnnxBatchNorm(momentum=momentum, epsilon=epsilon)
        onnx_mapping = onnx_mapping_from_node(node)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping,
    )
