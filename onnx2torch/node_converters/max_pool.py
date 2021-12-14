__all__ = []

from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import get_shape_from_value_info
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode

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
    maxpool_class = _MAXPOOL_CLASS_FROM_SPATIAL_RANK.get(spatial_rank, None)
    if maxpool_class is None:
        raise NotImplementedError(f'Convolution operation with spatial rank == {spatial_rank} is not implemented')

    node_attributes = node.attributes
    ceil_mode = node_attributes.get('ceil_mode', 0)
    dilation = node_attributes.get('dilations', 1)
    padding = node_attributes.get('pads', [0] * spatial_rank * 2)
    kernel_shape = node_attributes.get('kernel_shape', None)
    strides = node_attributes.get('strides', 1)
    storage_order = node_attributes.get('storage_order', 0)
    if storage_order != 0:
        raise NotImplementedError('Only row major (0) order is supported.')
    if kernel_shape is None:
        raise RuntimeError('Kernel shape for MaxPool not specified. Kernel shape is mandatory parameters in onnx.')

    auto_pad = node_attributes.get('auto_pad', 'NOTSET')
    if auto_pad == 'NOTSET':
        half_len = len(padding) // 2
        if tuple(padding[:half_len]) != tuple(padding[half_len:]):
            raise NotImplementedError(f'Only symmetric padding is implemented ({padding})')

        padding = padding[:half_len]
    elif auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
        raise NotImplementedError(f'"{auto_pad}" auto_pad is not implemented')
    else:
        raise ValueError(f'Got unexpected auto_pad value "{auto_pad}"')

    torch_module = maxpool_class(
        kernel_size=kernel_shape,
        stride=strides,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode == 1,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
