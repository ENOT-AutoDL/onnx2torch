__all__ = ['OnnxExpand']

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.common import SkipTorchTracing
from onnx2torch.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxExpand(nn.Module):

    @staticmethod
    def _do_forward(input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return input_tensor * torch.ones(torch.Size(shape), dtype=input_tensor.dtype, device=input_tensor.device)

    def forward(self, input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            with SkipTorchTracing():
                output = self._do_forward(input_tensor, shape)
                return _ExpandExportToOnnx.set_output_and_apply(output, input_tensor, shape)

        return self._do_forward(input_tensor, shape)


class _ExpandExportToOnnx(CustomExportToOnnx):

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Expand', *args, outputs=1)


@add_converter(operation_type='Expand', version=8)
@add_converter(operation_type='Expand', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxExpand(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
