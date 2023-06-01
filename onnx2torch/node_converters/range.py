__all__ = [
    'OnnxRange',
]

from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxRange(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy_buffer', torch.Tensor(), persistent=False)

    @staticmethod
    def _get_scalar(value) -> Union[float, int]:
        if isinstance(value, torch.Tensor):
            return value.item()

        return value

    def _arange(
        self,
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        return torch.arange(
            start=self._get_scalar(start),
            end=self._get_scalar(limit),
            step=self._get_scalar(delta),
            device=self.dummy_buffer.device,
        )

    def forward(  # pylint: disable=missing-function-docstring
        self,
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        forward_lambda = lambda: self._arange(start, limit, delta)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(forward_lambda, 'Range', start, limit, delta, {})

        return forward_lambda()


@add_converter(operation_type='Range', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxRange(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
