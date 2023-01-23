from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import torch
from onnx import ValueInfoProto  # pylint: disable=no-name-in-module
from torch import nn
from torch.onnx import symbolic_helper

from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """

    pass  # pylint: disable=unnecessary-pass


class OnnxMapping(NamedTuple):  # pylint: disable=missing-class-docstring
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


class OperationConverterResult(NamedTuple):  # pylint: disable=missing-class-docstring
    torch_module: nn.Module
    onnx_mapping: OnnxMapping


def onnx_mapping_from_node(node: OnnxNode) -> OnnxMapping:  # pylint: disable=missing-function-docstring
    return OnnxMapping(
        inputs=node.input_values,
        outputs=node.output_values,
    )


def get_onnx_version() -> int:
    """Returns opset version at the time of the export."""
    if hasattr(symbolic_helper, 'GLOBALS'):
        return symbolic_helper.GLOBALS.export_onnx_opset_version

    return symbolic_helper._export_onnx_opset_version  # pylint: disable=no-member, protected-access


def get_shape_from_value_info(value_info: ValueInfoProto) -> List[int]:  # pylint: disable=missing-function-docstring
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def get_const_value(  # pylint: disable=missing-function-docstring
    name: str,
    graph: OnnxGraph,
) -> Union[torch.Tensor, float, int, str, List]:
    if name in graph.initializers:
        return graph.initializers[name].to_torch()

    try:
        node, _ = graph.value_as_node_output(name)
    except KeyError as exc:
        raise KeyError(f'Tensor "{name}" is not found in constant values') from exc

    if node.operation_type == 'Constant':
        attr_name, attr_value = next(iter(node.attributes.items()))
        if attr_name == 'value':
            attr_value = attr_value.to_torch()

        return attr_value

    raise KeyError(f'Tensor "{name}" is not found in constant values')


def old_style_broadcast(  # pylint: disable=missing-function-docstring
    first: torch.Tensor,
    second: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    rank = len(first.shape)
    axis = axis + rank if axis < 0 else axis

    second_shape = [1] * axis + list(second.shape)
    second_shape = second_shape + [1] * (rank - len(second_shape))

    return second.view(second_shape)
