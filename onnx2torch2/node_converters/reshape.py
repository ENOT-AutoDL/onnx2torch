__all__ = [
    'OnnxReshape',
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


class OnnxReshape(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    @staticmethod
    def _do_reshape(input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        if torch.any(shape == 0):
            shape = [input_tensor.shape[i] if dim_size == 0 else dim_size for i, dim_size in enumerate(shape)]

        return torch.reshape(input_tensor, torch.Size(shape))

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        forward_lambda = lambda: self._do_reshape(input_tensor, shape)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(forward_lambda, 'Reshape', input_tensor, shape, {})

        return forward_lambda()


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
