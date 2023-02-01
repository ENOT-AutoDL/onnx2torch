__all__ = [
    'OnnxDropoutDynamic',
]

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxDropoutDynamic(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(  # pylint: disable=missing-function-docstring, unused-argument
        self,
        input_tensor: torch.Tensor,
        ratio: float = 0.5,
        training_mode: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ignoring training_mode from ONNX and use the one from PyTorch
        return F.dropout(input_tensor, p=ratio, training=self.training)


@add_converter(operation_type='Dropout', version=10)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    ratio = node_attributes.get('ratio', 0.5)

    torch_module = nn.Dropout(p=ratio)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Dropout', version=12)
@add_converter(operation_type='Dropout', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    seed = node_attributes.get('seed')
    if seed is not None:
        raise NotImplementedError('Dropout nodes with seeds are not supported.')

    return OperationConverterResult(
        torch_module=OnnxDropoutDynamic(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
