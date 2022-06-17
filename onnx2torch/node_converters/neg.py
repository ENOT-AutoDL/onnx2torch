__all__ = ['OnnxNeg']

from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxNeg(nn.Module, OnnxToTorchModule):

    def forward(self, x):
        return -x


@add_converter(operation_type='Neg', version=1)
@add_converter(operation_type='Neg', version=6)
@add_converter(operation_type='Neg', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxNeg(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
