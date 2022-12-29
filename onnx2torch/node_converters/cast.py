__all__ = [
    'OnnxCast',
]

import torch
from onnx import TensorProto  # pylint: disable=no-name-in-module
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

# pylint: disable=no-member
TENSOR_TYPE_TO_TORCH_TYPE = {
    int(TensorProto.FLOAT): torch.float32,
    int(TensorProto.UINT8): torch.uint8,
    int(TensorProto.INT8): torch.int8,
    int(TensorProto.INT16): torch.int16,
    int(TensorProto.INT32): torch.int32,
    int(TensorProto.INT64): torch.int64,
    int(TensorProto.BOOL): torch.bool,
    int(TensorProto.FLOAT16): torch.float16,
    int(TensorProto.DOUBLE): torch.float64,
    int(TensorProto.COMPLEX64): torch.complex64,
    int(TensorProto.COMPLEX128): torch.complex128,
}
# pylint: enable=no-member


class OnnxCast(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, onnx_dtype: int):
        super().__init__()
        try:
            self.torch_dtype = TENSOR_TYPE_TO_TORCH_TYPE[onnx_dtype]
        except KeyError as exc:
            raise NotImplementedError(f'Conversion to "{onnx_dtype}" is not implemented') from exc

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return input_tensor.to(self.torch_dtype)


@add_converter(operation_type='Cast', version=9)
@add_converter(operation_type='Cast', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    onnx_dtype = node_attributes.get('to', None)

    return OperationConverterResult(
        torch_module=OnnxCast(onnx_dtype),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
