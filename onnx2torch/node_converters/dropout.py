__all__ = []

import random

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_shape_from_value_info


_DO_CLASS_FROM_SPATIAL_RANK = {
    2: nn.Dropout2d,
    3: nn.Dropout3d,
}

@add_converter(operation_type='Dropout', version=13)
@add_converter(operation_type='Dropout', version=12)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    seed = node_attributes.get('seed', random.randint)

    ratio_value_name = node.input_values[1]
    ratio = graph.initializers[ratio_value_name]

    training_mode_value_name = node.input_values[2]
    training_mode = graph.initializers[training_mode_value_name]

    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    spatial_rank = len(input_shape) - 2
    do_class = _DO_CLASS_FROM_SPATIAL_RANK.get(spatial_rank, None)

    if training_mode != 0:
        raise NotImplementedError('Dropout nodes in training mode are not supported.')

    if do_class is None:
        torch_module = nn.Dropout(p=ratio, training_mode=training_mode)
    else:
        torch_module = do_class(p=ratio, training_mode=training_mode)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


