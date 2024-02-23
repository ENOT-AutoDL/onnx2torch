__all__ = [
    'OnnxTile',
]

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import onnx_mapping_from_node
from onnx2torch2.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch2.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxTile(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, repeats: list):
        super().__init__()
        self.repeats = repeats

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        repeats: torch.Tensor,
    ) -> torch.Tensor:
        # torch.tile(input_tensor, repeats) is not supported for exporting
        repeats = self.repeats if self.repeats else repeats
        forward_lambda = lambda: input_tensor.repeat(torch.Size(repeats))
        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(forward_lambda, 'Tile', input_tensor, repeats, {})

        return forward_lambda()


@add_converter(operation_type='Tile', version=6)
@add_converter(operation_type='Tile', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    repeats = []
    if node.input_values[1] in graph.initializers:
        repeats = graph.initializers[node.input_values[1]].to_torch().cpu().tolist()
    return OperationConverterResult(
        torch_module=OnnxTile(repeats),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
