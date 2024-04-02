# pylint: disable=missing-docstring
__all__ = [
    'OnnxIsInf',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult


class OnnxIsInf(nn.Module, OnnxToTorchModule):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.isinf(input_tensor)


@add_converter(operation_type='IsInf', version=10)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    torch_module = OnnxIsInf()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
