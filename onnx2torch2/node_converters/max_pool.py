__all__ = []

from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_shape_from_value_info
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.padding import onnx_auto_pad_to_torch_padding

_MAXPOOL_CLASS_FROM_SPATIAL_RANK = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}


@add_converter(operation_type='MaxPool', version=12)
@add_converter(operation_type='MaxPool', version=11)
@add_converter(operation_type='MaxPool', version=10)
@add_converter(operation_type='MaxPool', version=8)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    spatial_rank = len(input_shape) - 2
    try:
        maxpool_class = _MAXPOOL_CLASS_FROM_SPATIAL_RANK[spatial_rank]
    except KeyError as exc:
        raise NotImplementedError(f'Max pool operation with spatial rank == {spatial_rank} is not implemented') from exc

    node_attributes = node.attributes
    # required
    kernel_shape = node_attributes['kernel_shape']
    # optional
    ceil_mode = node_attributes.get('ceil_mode', 0)
    dilation = node_attributes.get('dilations', 1)
    strides = node_attributes.get('strides', 1)
    storage_order = node_attributes.get('storage_order', 0)
    if storage_order != 0:
        raise NotImplementedError('Only row major (0) order is supported.')

    padding, padding_module = onnx_auto_pad_to_torch_padding(
        onnx_padding=node_attributes.get('pads', [0] * spatial_rank * 2),
        auto_pad=node_attributes.get('auto_pad', 'NOTSET'),
    )

    torch_module = maxpool_class(
        kernel_size=kernel_shape,
        stride=strides,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode == 1,
    )
    if padding_module is not None:
        # MaxPool must ignore padded values, so we should pad by -inf
        padding_module.constant_value = float('-inf')
        torch_module = nn.Sequential(padding_module, torch_module)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
