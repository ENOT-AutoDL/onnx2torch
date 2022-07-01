from types import MappingProxyType
from typing import Any
from typing import Mapping
from typing import Tuple

from onnx.onnx_ml_pb2 import AttributeProto
from onnx.onnx_ml_pb2 import NodeProto

from onnx2torch.onnx_tensor import OnnxTensor


class OnnxNode:  # pylint: disable=missing-class-docstring
    def __init__(self, onnx_node_proto: NodeProto, unique_name: str):
        self._proto = onnx_node_proto
        self._unique_name = unique_name
        self._input_values = tuple(onnx_node_proto.input)
        self._output_values = tuple(onnx_node_proto.output)
        self._inputs = None

        self._proto_attributes = {
            attribute.name: OnnxNode._parse_attribute_value(attribute) for attribute in self._proto.attribute
        }

    @staticmethod
    def _parse_attribute_value(attribute: AttributeProto) -> Any:
        if attribute.HasField('i'):
            value = attribute.i
        elif attribute.HasField('f'):
            value = attribute.f
        elif attribute.HasField('s'):
            value = str(attribute.s, 'utf-8')
        elif attribute.HasField('t'):
            value = OnnxTensor(attribute.t)
        elif attribute.ints:
            value = list(attribute.ints)
        elif attribute.floats:
            value = list(attribute.floats)
        elif attribute.strings:
            value = [str(s, 'utf-8') for s in attribute.strings]
        elif attribute.tensors:
            value = [OnnxTensor(t) for t in attribute.tensors]
        else:
            value = attribute

        return value

    @property
    def proto(self) -> NodeProto:  # pylint: disable=missing-function-docstring
        return self._proto

    @property
    def name(self) -> str:  # pylint: disable=missing-function-docstring
        return self._proto.name

    @property
    def unique_name(self) -> str:  # pylint: disable=missing-function-docstring
        return self._unique_name

    @property
    def domain(self) -> str:  # pylint: disable=missing-function-docstring
        return self._proto.domain

    @property
    def operation_type(self) -> str:  # pylint: disable=missing-function-docstring
        return self._proto.op_type

    @property
    def input_values(self) -> Tuple[str, ...]:  # pylint: disable=missing-function-docstring
        return self._input_values

    @property
    def output_values(self) -> Tuple[str, ...]:  # pylint: disable=missing-function-docstring
        return self._output_values

    @property
    def attributes(self) -> Mapping[str, Any]:  # pylint: disable=missing-function-docstring
        return MappingProxyType(self._proto_attributes)
