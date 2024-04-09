__all__ = ['OnnxDepthToSpace']

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OnnxToTorchModule
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import onnx_mapping_from_node


class OnnxDepthToSpace(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, blocksize: int, mode: str):
        super().__init__()
        self.blocksize = blocksize
        self.mode = mode

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if self.mode == "DCR":
            b, c, h, w = input_tensor.shape
            tmp = torch.reshape(input_tensor, [b, self.blocksize, self.blocksize, c // (self.blocksize**2), h, w])
            tmp = torch.permute(tmp, [0, 3, 4, 1, 5, 2])
            return torch.reshape(tmp, [b, c // (self.blocksize**2), h * self.blocksize, w * self.blocksize])
        return torch.pixel_shuffle(input_tensor, upscale_factor=self.blocksize)


@add_converter(operation_type='DepthToSpace', version=11)
@add_converter(operation_type='DepthToSpace', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph

    blocksize: int = node.attributes['blocksize']  # required
    mode: str = node.attributes.get('mode', 'DCR')

    return OperationConverterResult(
        torch_module=OnnxDepthToSpace(blocksize=blocksize, mode=mode),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
