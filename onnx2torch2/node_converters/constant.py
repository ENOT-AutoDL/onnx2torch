__all__ = [
    'OnnxConstant',
]

from typing import Any

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

_CONSTANT_PARSING_MAPPING = {
    'value': lambda x: x.to_torch(),
    'value_float': torch.tensor,
    'value_floats': torch.tensor,
    'value_int': torch.tensor,
    'value_ints': torch.tensor,
    'value_string': lambda x: x,
    'value_strings': lambda x: x,
}


class OnnxConstant(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, value: Any):
        super().__init__()
        # We need it for placing constant to cuda.
        if isinstance(value, torch.Tensor):
            self.register_buffer('value', value)
        else:
            self.value = value

    def forward(self) -> Any:  # pylint: disable=missing-function-docstring
        return self.value


def _prepare_output_value(value: Any, attr_name: str) -> Any:
    if attr_name in _CONSTANT_PARSING_MAPPING:
        return _CONSTANT_PARSING_MAPPING[attr_name](value)

    raise NotImplementedError(f'value type "{attr_name}" not supported yet.')


@add_converter(operation_type='Constant', version=9)
@add_converter(operation_type='Constant', version=11)
@add_converter(operation_type='Constant', version=12)
@add_converter(operation_type='Constant', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    attr_name, value = list(node.attributes.items())[0]
    prepared_value = _prepare_output_value(value, attr_name)

    torch_module = OnnxConstant(
        value=prepared_value,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
