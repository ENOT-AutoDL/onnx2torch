__all__ = ['OnnxReshape']

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


class OnnxReshape(nn.Module, OnnxToTorchModuleWithCustomExport):

    @staticmethod
    def _do_reshape(input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        if torch.any(shape == 0):
            shape = [
                input_tensor.shape[i] if dim_size == 0 else dim_size
                for i, dim_size in enumerate(shape)
            ]

        return torch.reshape(input_tensor, torch.Size(shape))

    def forward(self, input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:  # pylint: disable=no-self-use
        output = self._do_reshape(input_tensor, shape)

        if torch.onnx.is_in_onnx_export():
            return _ReshapeExportToOnnx.set_output_and_apply(output, input_tensor, shape)

        return output


class _ReshapeExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Reshape', *args, outputs=1)


@add_converter(operation_type='Reshape', version=5)
@add_converter(operation_type='Reshape', version=13)
@add_converter(operation_type='Reshape', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node.attributes.get('allowzero', 0) == 1:
        raise NotImplementedError('"allowzero=1" is not implemented')

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
