__all__ = [
    'OnnxPadStatic',
    'OnnxPadDynamic',
]

from typing import List
from typing import Optional

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


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

        if torch.unique(pads).shape[0] > 1:
            raise NotImplementedError(f'Only uniform padding on all axes is implemented ({pads})')

        return torch.nn.functional.pad(input_tensor, mode=self.mode, pad=tuple(pads), value=constant_value)


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

    if len(set(pads)) > 1:
            raise NotImplementedError(f'Only uniform padding on all axes is implemented ({pads})')

    constant_value = node.attributes.get('constant_value', 0.0)
    torch_module = OnnxPadStatic(
        mode=mode,
        pads=pads,
        constant_value=constant_value,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
