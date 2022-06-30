import logging
from typing import Callable
from typing import NamedTuple

from onnx import defs

from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult

_LOGGER = logging.getLogger(__name__)
_CONVERTER_REGISTRY = {}


class OperationDescription(NamedTuple):  # pylint: disable=missing-class-docstring
    domain: str
    operation_type: str
    version: int


TConverter = Callable[[OnnxNode, OnnxGraph], OperationConverterResult]


def add_converter(  # pylint: disable=missing-function-docstring
    operation_type: str,
    version: int,
    domain: str = defs.ONNX_DOMAIN,
):
    description = OperationDescription(
        domain=domain,
        operation_type=operation_type,
        version=version,
    )

    def deco(converter: TConverter):
        if description in _CONVERTER_REGISTRY:
            raise ValueError(f'Operation "{description}" already registered')

        _CONVERTER_REGISTRY[description] = converter
        _LOGGER.info(f'Operation converter registered {description}')

        return converter

    return deco


def get_converter(  # pylint: disable=missing-function-docstring
    operation_type: str,
    version: int,
    domain: str = defs.ONNX_DOMAIN,
) -> TConverter:
    try:
        version = defs.get_schema(
            operation_type,
            domain=domain,
            max_inclusive_version=version,
        ).since_version
    except (RuntimeError, defs.SchemaError):
        pass

    description = OperationDescription(
        domain=domain,
        operation_type=operation_type,
        version=version,
    )

    converter = _CONVERTER_REGISTRY.get(description, None)
    if converter is None:
        raise NotImplementedError(f'Converter is not implemented ({description})')

    return converter
