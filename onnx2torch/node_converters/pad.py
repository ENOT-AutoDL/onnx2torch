__all__ = [
    'OnnxPadStatic',
    'OnnxPadDynamic',
]

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


def _onnx_padding_to_torch(pads: List[int]) -> List[int]:
    middle = len(pads) // 2
    onnx_pad_begin, onnx_pad_end = pads[:middle], pads[middle:]
    onnx_pad_begin, onnx_pad_end = onnx_pad_begin[::-1], onnx_pad_end[::-1]
    torch_pads = []
    for begin, end in zip(onnx_pad_begin, onnx_pad_end):
        torch_pads.extend([begin, end])

    return torch_pads

class OnnxPadStatic(nn.Module, OnnxToTorchModule):

    def __init__(
        self,
        pads: List[int],
        mode: str = 'constant',
        constant_value: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.pads = pads
        self.constant_value = constant_value

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(
            input_tensor,
            mode=self.mode,
            pad=self.pads,
            value=self.constant_value,
        )


class OnnxPadDynamic(nn.Module, OnnxToTorchModule):

    def __init__(self, mode: str = 'constant'):
        super().__init__()
        self.mode = mode

    def forward(
        self,
        input_tensor: torch.Tensor,
        pads: torch.Tensor,
        constant_value: Optional[float] = 0.0,
    ) -> torch.Tensor:

        torch_pads = _onnx_padding_to_torch(pads.tolist())
        return torch.nn.functional.pad(input_tensor, mode=self.mode, pad=torch_pads, value=constant_value)


@add_converter(operation_type='Pad', version=11)
@add_converter(operation_type='Pad', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    mode = node.attributes.get('mode', 'constant')
    if mode != 'constant':
        raise NotImplementedError(f'"{mode}" pad mode is not implemented.')
    torch_module = OnnxPadDynamic(
        mode=mode,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=node.input_values,
            outputs=node.output_values,
        ),
    )


@add_converter(operation_type='Pad', version=2)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    mode = node.attributes.get('mode', 'constant')

    if mode != 'constant':
        raise NotImplementedError(f'"{mode}" pad mode is not implemented.')

    pads = node.attributes.get('pads')
    torch_pads = onnx_padding_to_torch(pads)

    constant_value = node.attributes.get('constant_value', 0.0)
    torch_module = OnnxPadStatic(
        mode=mode,
        pads=torch_pads,
        constant_value=constant_value,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
