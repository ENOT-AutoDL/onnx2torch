__all__ = ['OnnxRange']

from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxRange(nn.Module, OnnxToTorchModule):

    def __init__(self):
        super().__init__()
        self.register_buffer('dummy_buffer', torch.Tensor(), persistent=False)

    @staticmethod
    def _get_scalar(value) -> Union[float, int]:
        if isinstance(value, torch.Tensor):
            return value.item()

        return value

    def forward(
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


@add_converter(operation_type='Range', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxRange(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
