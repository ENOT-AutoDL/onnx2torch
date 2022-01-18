from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union
from warnings import catch_warnings
from warnings import filterwarnings

import torch
import torch._C as torch_C
from onnx import ValueInfoProto
from torch import nn
from torch.jit import TracerWarning

from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


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


class SkipTorchTracing:
    def __init__(self):
        self._catch_warnings = catch_warnings()
        self._state = None

    def __enter__(self):
        self._state = torch_C._get_tracing_state()
        self._catch_warnings.__enter__()
        filterwarnings(action='ignore', category=TracerWarning)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch_C._set_tracing_state(self._state)
        self._catch_warnings.__exit__(exc_type, exc_val, exc_tb)


def old_style_broadcast(first: torch.Tensor, second: torch.Tensor, axis: int) -> torch.Tensor:
    rank = len(first.shape)
    axis = axis + rank if axis < 0 else axis

    second_shape = [1]*axis + list(second.shape)
    second_shape = second_shape + [1]*(rank - len(second_shape))

    return second.view(second_shape)
