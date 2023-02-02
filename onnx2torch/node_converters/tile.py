__all__ = [
    'OnnxTile',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxTile(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        repeats: torch.Tensor,
    ) -> torch.Tensor:
        # torch.tile(input_tensor, repeats) is not supported for exporting
        forward_lambda = lambda: input_tensor.repeat(torch.Size(repeats))
        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(forward_lambda, 'Tile', input_tensor, repeats, {})

        return forward_lambda()


@add_converter(operation_type='Tile', version=6)
@add_converter(operation_type='Tile', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxTile(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
