__all__ = ['OnnxDepthToSpace']

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxDepthToSpace(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, blocksize: int):
        super().__init__()
        self._upscale_factor = blocksize

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.pixel_shuffle(input_tensor, upscale_factor=self._upscale_factor)


@add_converter(operation_type='DepthToSpace', version=11)
@add_converter(operation_type='DepthToSpace', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph

    blocksize: int = node.attributes['blocksize']  # required
    mode: str = node.attributes.get('mode', 'DCR')

    if mode != 'CRD':
        raise NotImplementedError('DepthToSpace for mode other than CRD is not implemented')

    return OperationConverterResult(
        torch_module=OnnxDepthToSpace(blocksize=blocksize),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
