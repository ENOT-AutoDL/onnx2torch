__all__ = ['OnnxTile']

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxTile(nn.Module, OnnxToTorchModuleWithCustomExport):

    def forward(self, input_tensor: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
        # torch.tile(input_tensor, repeats) is not supported for exporting
        output = input_tensor.repeat(torch.Size(repeats))
        if torch.onnx.is_in_onnx_export():
            return _TileExportToOnnx.set_output_and_apply(output, input_tensor, repeats)

        return output


class _TileExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Tile', *args, outputs=1)


@add_converter(operation_type='Tile', version=6)
@add_converter(operation_type='Tile', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument

    return OperationConverterResult(
        torch_module=OnnxTile(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
