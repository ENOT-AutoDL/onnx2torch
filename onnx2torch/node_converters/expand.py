__all__ = [
    'OnnxExpand',
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


class OnnxExpand(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-docstring
    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        def _forward():
            return input_tensor * torch.ones(torch.Size(shape), dtype=input_tensor.dtype, device=input_tensor.device)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'Expand', input_tensor, shape, {})

        return _forward()


@add_converter(operation_type='Expand', version=8)
@add_converter(operation_type='Expand', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxExpand(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
