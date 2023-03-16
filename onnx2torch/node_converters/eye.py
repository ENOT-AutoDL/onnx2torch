__all__ = [
    'OnnxEyeLike',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxEyeLike(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, eyelike_k: int, dtype: torch.dtype = None):
        super().__init__()
        self.dtype = dtype
        self.eyelike_k = eyelike_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(x.shape) != 2:
            raise ValueError(f'EyeLike only supports 2D tensors, got {len(x.shape)}')

        if self.dtype is None:
            self.dtype = x.dtype

        size_n = x.size(dim=0)
        size_m = x.size(dim=1)
        if self.eyelike_k > size_n:
            raise ValueError(
                f'EyeLike attribute k should be less or equal than the zero dimension of input tensor,'
                f'got {self.eyelike_k} and {size_n}'
            )

        if self.eyelike_k == 0:
            return torch.eye(n=size_n, m=size_m, dtype=self.dtype)

        k_tensor = torch.zeros(size_n, self.eyelike_k, dtype=self.dtype)
        eye_tensor = torch.eye(n=size_n, m=size_m - self.eyelike_k, dtype=self.dtype)
        return torch.concat([k_tensor, eye_tensor], axis=1)


@add_converter(operation_type='EyeLike', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    eyelike_k = node_attributes.get('k', 0)
    dtype = node_attributes.get('dtype')
    return OperationConverterResult(
        torch_module=OnnxEyeLike(dtype=dtype, eyelike_k=eyelike_k),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
