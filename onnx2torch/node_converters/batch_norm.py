__all__ = []

import torch
from torch import nn

from onnx2torch.common import OnnxMapping
from onnx2torch.common import OperationConverterResult
from onnx2torch.common import get_shape_from_value_info
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode

_BN_CLASS_FROM_SPATIAL_RANK = {
    0: nn.BatchNorm1d,
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


@add_converter(operation_type='BatchNormalization', version=15)
@add_converter(operation_type='BatchNormalization', version=14)
@add_converter(operation_type='BatchNormalization', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    scale_value_name = node.input_values[1]
    scale = graph.initializers[scale_value_name]
    scale = scale.to_torch()

    bias_value_name = node.input_values[2]
    bias = graph.initializers[bias_value_name]
    bias = bias.to_torch()

    mean_value_name = node.input_values[3]
    mean = graph.initializers[mean_value_name]
    mean = mean.to_torch()

    var_value_name = node.input_values[4]
    var = graph.initializers[var_value_name]
    var = var.to_torch()

    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    spatial_rank = len(input_shape) - 2
    bn_class = _BN_CLASS_FROM_SPATIAL_RANK.get(spatial_rank, None)
    if bn_class is None:
        raise NotImplementedError(f'BatchNorm operation with spatial rank == {spatial_rank} is not implemented')

    node_attributes = node.attributes
    training_mode = node_attributes.get('training_mode', 0)
    epsilon = node_attributes.get('epsilon', 1e-5)
    momentum = node_attributes.get('momentum', 0.9)
    if training_mode != 0:
        raise NotImplementedError('BatchNorm nodes in training mode are not supported.')

    torch_module = bn_class(
        num_features=scale.size()[0],
        eps=epsilon,
        momentum=1 - momentum,  # See PyTorch documentation for batch norm.
    )
    torch_module.eval()
    with torch.no_grad():
        torch_module.running_mean.data = mean
        torch_module.running_var.data = var
        torch_module.weight.data = scale
        torch_module.bias.data = bias

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
