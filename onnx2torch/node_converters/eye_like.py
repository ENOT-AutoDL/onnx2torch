__all__ = [
    'OnnxEyeLike',
]

from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.dtype import onnx_dtype_to_torch_dtype


class OnnxEyeLike(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, dtype: Optional[int] = None, k: int = 0):  # pylint: disable=invalid-name
        super().__init__()
        self.dtype = dtype
        self.k = k  # pylint: disable=invalid-name

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(x.shape) != 2:
            raise ValueError(f'EyeLike only supports 2D tensors, got {len(x.shape)}')

        dtype = x.dtype if self.dtype is None else onnx_dtype_to_torch_dtype(self.dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f'Expected type of dtype is torch.dtype, got {type(dtype)}')

        rows, cols = x.size()
        if self.k > rows:
            raise ValueError(
                f'EyeLike attribute k should be less or equal than the zero dimension of input tensor,'
                f'got {self.k} and {rows}'
            )

        if self.k == 0:
            return torch.eye(n=rows, m=cols, dtype=dtype)
        if self.k > 0:
            return torch.concat(
                [
                    torch.zeros(rows, self.k, dtype=dtype),
                    torch.eye(n=rows, m=(cols - self.k), dtype=dtype),
                ],
                dim=1,
            )
        return torch.concat(  # k < 0:
            [
                torch.zeros(-self.k, cols, dtype=dtype),
                torch.eye(n=(rows + self.k), m=cols, dtype=dtype),
            ],
            dim=0,
        )


@add_converter(operation_type='EyeLike', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    k = node_attributes.get('k', 0)  # pylint: disable=invalid-name
    dtype = node_attributes.get('dtype', None)
    return OperationConverterResult(
        torch_module=OnnxEyeLike(dtype=dtype, k=k),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
