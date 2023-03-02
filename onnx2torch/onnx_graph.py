from collections import OrderedDict
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from typing import Tuple

from onnx.onnx_ml_pb2 import GraphProto
from onnx.onnx_ml_pb2 import NodeProto
from onnx.onnx_ml_pb2 import ValueInfoProto

from onnx2torch.onnx_node import OnnxNode
from onnx2torch.onnx_tensor import OnnxTensor


class ValueType(Enum):  # pylint: disable=missing-class-docstring
    GRAPH_INPUT = 0
    NODE_OUTPUT = 1
    GRAPH_INITIALIZER = 2
    UNKNOWN = 3
    EMPTY = 4


class OnnxGraph:  # pylint: disable=missing-class-docstring
    def __init__(self, onnx_graph_proto: GraphProto):
        self._proto = onnx_graph_proto
        self._input_values = tuple(value_info.name for value_info in self._proto.input)
        self._output_values = tuple(value_info.name for value_info in self._proto.output)

        unique_names = []
        counters = {}
        for node in onnx_graph_proto.node:
            name = OnnxGraph.generate_node_name(node)
            name_counter = counters.setdefault(name, 0)
            counters[name] += 1
            unique_names.append(f'{name}' + (f'_{name_counter}' if name_counter > 0 else ''))

        self._nodes = OrderedDict(
            (name, OnnxNode(node, unique_name=name)) for name, node in zip(unique_names, onnx_graph_proto.node)
        )
        self._initializers = {initializer.name: OnnxTensor(initializer) for initializer in onnx_graph_proto.initializer}
        self._node_output_values = {
            output_name: (node, i) for node in self._nodes.values() for i, output_name in enumerate(node.output_values)
        }
        self._value_info = {value_info.name: value_info for value_info in onnx_graph_proto.value_info}
        for input_value_info in onnx_graph_proto.input:
            self._value_info[input_value_info.name] = input_value_info
        for output_value_info in onnx_graph_proto.output:
            self._value_info[output_value_info.name] = output_value_info

    @property
    def proto(self) -> GraphProto:  # pylint: disable=missing-function-docstring
        return self._proto

    @property
    def value_info(self) -> Mapping[str, ValueInfoProto]:  # pylint: disable=missing-function-docstring
        return self._value_info

    @property
    def name(self) -> str:  # pylint: disable=missing-function-docstring
        return self._proto.name

    @property
    def input_values(self) -> Tuple[str, ...]:  # pylint: disable=missing-function-docstring
        return self._input_values

    @property
    def output_values(self) -> Tuple[str, ...]:  # pylint: disable=missing-function-docstring
        return self._output_values

    @property
    def nodes(self) -> Mapping[str, OnnxNode]:  # pylint: disable=missing-function-docstring
        return self._nodes

    @property
    def initializers(self) -> Mapping[str, OnnxTensor]:  # pylint: disable=missing-function-docstring
        return MappingProxyType(self._initializers)

    def value_type(self, value_name: str) -> ValueType:  # pylint: disable=missing-function-docstring
        if value_name in self._input_values:
            return ValueType.GRAPH_INPUT

        if value_name in self._node_output_values:
            return ValueType.NODE_OUTPUT

        if value_name in self._initializers:
            return ValueType.GRAPH_INITIALIZER

        if value_name == '':
            return ValueType.EMPTY

        return ValueType.UNKNOWN

    def value_as_node_output(  # pylint: disable=missing-function-docstring
        self,
        value_name: str,
    ) -> Tuple[OnnxNode, int]:
        return self._node_output_values[value_name]

    @staticmethod
    def generate_node_name(node: NodeProto) -> str:
        """Generate a torch module name from the given onnx node import it with.

        Uses the ONNX node's name by default, falling back to the op_type in case the former is empty. The node's
        domain is prepended to this.

        Dots (.) are not allowed within names in torch, so they are replaced with a slash (/) instead.

        Parameters
        ----------
        node
            The ONNX node to create a name from.

        Returns
        -------
        A torch-compatible module name based on the given node's properties.
        """
        return (f'{node.domain}/' + (node.name.replace('.', '/') or node.op_type)).lstrip('/')
