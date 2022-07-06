__all__ = [
    'OnnxTopK',
]

from typing import Tuple
from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxTopK(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, dim: int = -1, largest: int = 1, sorted_: int = 1):
        super().__init__()
        self.dim = dim
        self.largest = largest == 1
        self.sorted = sorted_ == 1

    def forward(  # pylint: disable=missing-function-docstring, invalid-name
        self,
        input_tensor: torch.Tensor,
        k: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = k[0] if isinstance(k, torch.Tensor) else k

        top_k = torch.topk(
            input_tensor,
            k=k,
            dim=self.dim,
            largest=self.largest,
            sorted=self.sorted,
        )
        return top_k.values, top_k.indices


@add_converter(operation_type='TopK', version=1)
@add_converter(operation_type='TopK', version=10)
@add_converter(operation_type='TopK', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    axis = node_attributes.get('axis', -1)
    largest = node_attributes.get('largest', 1)
    sorted_ = node_attributes.get('sorted', 1)

    return OperationConverterResult(
        torch_module=OnnxTopK(dim=axis, largest=largest, sorted_=sorted_),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
