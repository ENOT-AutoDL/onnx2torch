from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import torch
from onnx import ValueInfoProto
from torch import nn

from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxMapping(NamedTuple):
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


class OperationConverterResult(NamedTuple):
    torch_module: nn.Module
    onnx_mapping: OnnxMapping


def onnx_mapping_from_node(node: OnnxNode) -> OnnxMapping:
    return OnnxMapping(
        inputs=node.input_values,
        outputs=node.output_values,
    )


def get_shape_from_value_info(value_info: ValueInfoProto) -> List[int]:
    return [
        dim.dim_value
        for dim in value_info.type.tensor_type.shape.dim
    ]


def get_const_value(name: str, graph: OnnxGraph) -> Union[torch.Tensor, float, int, str, List]:
    if name in graph.initializers:
        return graph.initializers[name].to_torch()

    try:
        node, _ = graph.value_as_node_output(name)
    except KeyError:
        raise KeyError(f'Tensor "{name}" is not found in constant values')

    if node.operation_type == 'Constant':
        attr_name, attr_value = next(iter(node.attributes.items()))
        if attr_name == 'value':
            attr_value = attr_value.to_torch()

        return attr_value

    raise KeyError(f'Tensor "{name}" is not found in constant values')


def old_style_broadcast(first: torch.Tensor, second: torch.Tensor, axis: int) -> torch.Tensor:
    rank = len(first.shape)
    axis = axis + rank if axis < 0 else axis

    second_shape = [1]*axis + list(second.shape)
    second_shape = second_shape + [1]*(rank - len(second_shape))

    return second.view(second_shape)


def onnx_padding_to_torch_padding(padding: Tuple[int, ...], auto_pad: str) -> Tuple[int, ...]:
    if auto_pad == 'NOTSET':
        half_len = len(padding) // 2
        if padding[:half_len] != padding[half_len:]:
            raise NotImplementedError(f'Only symmetric padding is implemented ({padding})')

        padding = padding[:half_len]
    elif auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
        raise NotImplementedError(f'"{auto_pad}" auto_pad is not implemented')
    else:
        raise ValueError(f'Got unexpected auto_pad value "{auto_pad}"')

    return padding
