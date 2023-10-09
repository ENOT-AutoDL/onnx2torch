__all__ = [
    'OnnxPadStatic',
    'OnnxPadDynamic',
]

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

_ONNX_TO_TORCH_MODE = {
    'constant': 'constant',
    'reflect': 'reflect',
    'edge': 'replicate',
}


def _onnx_to_torch_mode(mode: str) -> str:
    try:
        return _ONNX_TO_TORCH_MODE[mode]
    except KeyError as exc:
        raise NotImplementedError(f'{mode} mode is not implemented') from exc


def _torch_padding_to_mode_format(pads: List[int], mode: str) -> List[int]:
    if mode in ('replicate', 'reflect'):
        batch_channel_pads = pads[-4:]
        if set(batch_channel_pads) == {0}:
            return pads[:-4]

        raise RuntimeError(
            f'{mode} padding is implemented for padding the last 3 dimensions of 5D input tensor, '
            f'or the last 2 dimensions of 4D input tensor, or the last dimension of 3D input tensor.'
        )

    return pads


def _onnx_padding_to_torch(pads: List[int]) -> List[int]:
    # Convert padding from onnx format to torch format
    # onnx format: [x1_begin, x2_begin, ... , x1_end, x2_end, ...]
    # torch format [xn_begin, xn_end, ... , x2_begin, x2_end, x1_begin, x1_end]
    middle = len(pads) // 2
    onnx_pad_begin, onnx_pad_end = pads[:middle], pads[middle:]
    onnx_pad_begin, onnx_pad_end = onnx_pad_begin[::-1], onnx_pad_end[::-1]
    torch_pads = []
    for begin, end in zip(onnx_pad_begin, onnx_pad_end):
        torch_pads.extend([begin, end])

    return torch_pads


class OnnxPadStatic(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(
        self,
        pads: Union[Tuple[int, ...], List[int]],
        mode: str = 'constant',
        constant_value: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.pads = pads
        self.constant_value = constant_value

    @classmethod
    def create_from_onnx_params(  # pylint: disable=missing-function-docstring
        cls,
        onnx_pads: Union[Tuple[int, ...], List[int]],
        onnx_mode: str = 'constant',
        constant_value: float = 0.0,
    ) -> 'OnnxPadStatic':
        torch_mode = _onnx_to_torch_mode(onnx_mode)
        torch_padding = _onnx_padding_to_torch(onnx_pads)
        torch_padding = _torch_padding_to_mode_format(torch_padding, torch_mode)
        return cls(pads=torch_padding, mode=torch_mode, constant_value=constant_value)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return F.pad(  # pylint: disable=not-callable
            input_tensor,
            mode=self.mode,
            pad=self.pads,
            value=self.constant_value,
        )


class OnnxPadDynamic(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, mode: str = 'constant'):
        super().__init__()
        self.mode = mode

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        pads: torch.Tensor,
        constant_value: Optional[float] = 0.0,
    ) -> torch.Tensor:
        torch_pads = _onnx_padding_to_torch(pads.tolist())
        torch_pads = _torch_padding_to_mode_format(torch_pads, self.mode)

        return F.pad(input_tensor, mode=self.mode, pad=torch_pads, value=constant_value)  # pylint: disable=not-callable


@add_converter(operation_type='Pad', version=11)
@add_converter(operation_type='Pad', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    mode = node.attributes.get('mode', 'constant')
    mode = _onnx_to_torch_mode(mode)

    return OperationConverterResult(
        torch_module=OnnxPadDynamic(mode=mode),
        onnx_mapping=OnnxMapping(
            inputs=node.input_values,
            outputs=node.output_values,
        ),
    )


@add_converter(operation_type='Pad', version=2)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    torch_module = OnnxPadStatic.create_from_onnx_params(
        onnx_mode=node.attributes.get('mode', 'constant'),
        onnx_pads=node.attributes.get('pads'),
        constant_value=node.attributes.get('constant_value', 0.0),
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
