__all__ = []

from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


@add_converter(operation_type='LRN', version=13)
@add_converter(operation_type='LRN', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    size = node.attributes.get('size')
    alpha = node.attributes.get('alpha', 0.0001)
    beta = node.attributes.get('beta', 0.75)
    k = node.attributes.get('bias', 1)  # pylint: disable=invalid-name

    return OperationConverterResult(
        torch_module=nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
