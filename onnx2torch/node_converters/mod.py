__all__ = [
    'OnnxMod',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxMod(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, fmod: int):
        super().__init__()
        self.fmod = fmod

        if self.fmod not in [0, 1]:
            raise ValueError(f'OnnxMod fom must be 0 or 1, but get {self.fmod}')

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.fmod(x, y) if self.fmod else torch.remainder(x, y)


@add_converter(operation_type='Mod', version=10)
@add_converter(operation_type='Mod', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    fmod = node_attributes.get('fmod', 0)
    return OperationConverterResult(
        torch_module=OnnxMod(fmod=fmod),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
