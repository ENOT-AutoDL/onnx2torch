__all__ = []

from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_shape_from_value_info
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.padding import onnx_auto_pad_to_torch_padding

_AVGPOOL_CLASS_FROM_SPATIAL_RANK = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d,
    3: nn.AvgPool3d,
}


@add_converter(operation_type='AveragePool', version=7)
@add_converter(operation_type='AveragePool', version=10)
@add_converter(operation_type='AveragePool', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    spatial_rank = len(input_shape) - 2
    try:
        avgpool_class = _AVGPOOL_CLASS_FROM_SPATIAL_RANK[spatial_rank]
    except KeyError as exc:
        raise NotImplementedError(
            f'Average pool operation with spatial rank == {spatial_rank} is not implemented'
        ) from exc

    node_attributes = node.attributes
    # required
    kernel_shape = node_attributes['kernel_shape']
    # optional
    ceil_mode = node_attributes.get('ceil_mode', 0)
    strides = node_attributes.get('strides', 1)
    count_include_pad = node_attributes.get('count_include_pad', 0)

    padding, padding_module = onnx_auto_pad_to_torch_padding(
        onnx_padding=node_attributes.get('pads', [0] * spatial_rank * 2),
        auto_pad=node_attributes.get('auto_pad', 'NOTSET'),
    )
    if padding_module is not None:
        raise NotImplementedError('AvgPool with non symmetrical padding is not implemented.')

    torch_module = avgpool_class(
        kernel_size=kernel_shape,
        stride=strides,
        padding=padding,
        count_include_pad=count_include_pad == 1,
        ceil_mode=ceil_mode == 1,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
