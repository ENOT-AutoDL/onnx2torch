__all__ = [
    'OnnxGlobalAveragePool',
    'OnnxGlobalAveragePoolWithKnownInputShape',
]

from typing import List

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_shape_from_value_info
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxGlobalAveragePool(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-docstring
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        def _forward():
            x_dims = list(range(2, len(input_tensor.shape)))
            return torch.mean(input_tensor, dim=x_dims, keepdim=True)

        if torch.onnx.is_in_onnx_export():
            return _GlobalAveragePoolExportToOnnx.set_forward_and_apply(_forward, input_tensor)

        return _forward()


class OnnxGlobalAveragePoolWithKnownInputShape(
    nn.Module, OnnxToTorchModuleWithCustomExport
):  # pylint: disable=missing-docstring
    def __init__(self, input_shape: List[int]):
        super().__init__()
        self.x_dims = list(range(2, len(input_shape)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        forward_lambda = lambda: torch.mean(input_tensor, dim=self.x_dims, keepdim=True)
        if torch.onnx.is_in_onnx_export():
            return _GlobalAveragePoolExportToOnnx.set_forward_and_apply(forward_lambda, input_tensor)

        return forward_lambda()


class _GlobalAveragePoolExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('GlobalAveragePool', *args, outputs=1)


@add_converter(operation_type='GlobalAveragePool', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    if input_shape is not None:
        torch_module = OnnxGlobalAveragePoolWithKnownInputShape(input_shape=input_shape)
    else:
        torch_module = OnnxGlobalAveragePool()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
