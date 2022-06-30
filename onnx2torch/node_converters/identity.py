import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxCopyIdentity(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, x: torch.Tensor):  # pylint: disable=missing-function-docstring
        return x.clone()


@add_converter(operation_type='Identity', version=16)
@add_converter(operation_type='Identity', version=14)
@add_converter(operation_type='Identity', version=13)
@add_converter(operation_type='Identity', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    # We need copy identity because in onnx identity create new tensor.
    # Pytorch identity simply returns the same tensor.
    # Which ruin quantization logic, because we should mark quantized tensors.
    # For example, input quantization node will be supressed if input tensor is already quantized.
    return OperationConverterResult(
        torch_module=OnnxCopyIdentity(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
