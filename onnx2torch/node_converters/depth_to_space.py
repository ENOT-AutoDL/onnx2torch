__all__ = [
    'OnnxDepthToSpace',
]

import torch
from torch import nn

from onnx2torch.node_converters.base_element_wise import OnnxBaseElementWise
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import OnnxToTorchModule


class OnnxDepthToSpace(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, blocksize):
        super().__init__()
        self._upscale_factor = blocksize

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.pixel_shuffle(input_tensor, self._upscale_factor);


@add_converter(operation_type='DepthToSpace', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node.attributes.get("mode") != "CRD":
        raise NotImplementedError('DepthToSpace for mode other than CRD is not implemented')
    return OperationConverterResult(
        torch_module=OnnxDepthToSpace(node.attributes.get("blocksize")),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )