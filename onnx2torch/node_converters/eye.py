__all__ = [
    'OnnxEyeLike',
]

import torch
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node
from torch import nn


class OnnxEyeLike(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, k: int, dtype: torch.dtype = None):
        super().__init__()
        self.dtype = dtype
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(x.shape) != 2:
            raise ValueError('OnnxEyeLike only support 2-D tensor')

        n = x.size(dim=0)
        m = x.size(dim=1)
        if self.k > n:
            raise ValueError(f'Error EyeLike Attribute k value, the k value is {self.k}, but x shape is {(n,m)}')

        if self.k == 0:
            return torch.eye(n, m, dtype=self.dtype)

        k_tensor = torch.zeros(n, self.k, dtype=self.dtype)
        eye_tensor = torch.eye(n, m - self.k, dtype=self.dtype)
        return torch.concat([k_tensor, eye_tensor], axis=1)


@add_converter(operation_type='EyeLike', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    k = node_attributes.get('k', 0)
    dtype = node_attributes.get('dtype', torch.float32)
    return OperationConverterResult(
        torch_module=OnnxEyeLike(dtype=dtype, k=k),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
