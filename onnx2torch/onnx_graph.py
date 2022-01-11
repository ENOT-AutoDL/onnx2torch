from collections import OrderedDict
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from typing import Tuple

from onnx.onnx_ml_pb2 import GraphProto
from onnx.onnx_ml_pb2 import ValueInfoProto

from onnx2torch.onnx_node import OnnxNode
from onnx2torch.onnx_tensor import OnnxTensor


class ValueType(Enum):
    GRAPH_INPUT = 0
    NODE_OUTPUT = 1
    GRAPH_INITIALIZER = 2
    UNKNOWN = 3
    EMPTY = 4


class OnnxGraph:
    def __init__(self, onnx_graph_proto: GraphProto):
        self._proto = onnx_graph_proto
        self._input_values = tuple(value_info.name for value_info in self._proto.input)
        self._output_values = tuple(value_info.name for value_info in self._proto.output)

        unique_names = []
        counters = {}
        for node in onnx_graph_proto.node:
            name = f'{node.domain}_{node.op_type}'.lstrip('_')
            name_counter = counters.setdefault(name, 0)
            counters[name] += 1
            unique_names.append(f'{name}_{name_counter}')

        self._nodes = OrderedDict(
            (name, OnnxNode(node, unique_name=name))
            for name, node in zip(unique_names, onnx_graph_proto.node)
        )
        self._initializers = {
            initializer.name: OnnxTensor(initializer)
            for initializer in onnx_graph_proto.initializer
        }
        self._node_output_values = {
            output_name: (node, i)
            for node in self._nodes.values()
            for i, output_name in enumerate(node.output_values)
        }
        self._value_info = {
            value_info.name: value_info
            for value_info in onnx_graph_proto.value_info
        }
        for input_value_info in onnx_graph_proto.input:
            self._value_info[input_value_info.name] = input_value_info
        for output_value_info in onnx_graph_proto.output:
            self._value_info[output_value_info.name] = output_value_info

    @property
    def proto(self) -> GraphProto:
        return self._proto

    @property
    def value_info(self) -> Mapping[str, ValueInfoProto]:
        return self._value_info

    @property
    def name(self) -> str:
        return self._proto.name

    @property
    def input_values(self) -> Tuple[str, ...]:
        return self._input_values

    @property
    def output_values(self) -> Tuple[str, ...]:
        return self._output_values

    @property
    def nodes(self) -> Mapping[str, OnnxNode]:
        return self._nodes

    @property
    def initializers(self) -> Mapping[str, OnnxTensor]:
        return MappingProxyType(self._initializers)

    def value_type(self, value_name: str) -> ValueType:
        if value_name in self._input_values:
            return ValueType.GRAPH_INPUT

        if value_name in self._node_output_values:
            return ValueType.NODE_OUTPUT

        if value_name in self._initializers:
            return ValueType.GRAPH_INITIALIZER
        
        if value_name == '':
            return ValueType.EMPTY

        return ValueType.UNKNOWN

    def value_as_node_output(self, value_name: str) -> Tuple[OnnxNode, int]:
        return self._node_output_values[value_name]
