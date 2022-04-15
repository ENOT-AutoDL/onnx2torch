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


class OnnxPadDynamic(nn.Module, OnnxToTorchModuleWithCustomExport):

    def __init__(self, mode: str = 'constant'):
        super().__init__()
        self.mode = mode

    @staticmethod
    def _do_forward(
        input_tensor: torch.Tensor,
        mode: str,
        pads: torch.Tensor,
        constant_value: Optional[float],
    ) -> torch.Tensor:

        if constant_value is not None:
            return torch.nn.functional.pad(input_tensor, mode=mode, pad=pads, value=constant_value)

        return torch.nn.functional.pad(input_tensor, mode=mode, pad=pads)

    def forward(
        self,
        input_tensor: torch.Tensor,
        pads: torch.Tensor,
        constant_value: Optional[float] = None,
    ) -> torch.Tensor:

        if torch.unique(pads).shape[0] > 1:
            raise NotImplementedError(f'Only uniform padding on all axes is implemented ({pads})')

        pads = tuple(pads.tolist())
        output = self._do_forward(input_tensor, self.mode, pads, constant_value)
        if torch.onnx.is_in_onnx_export():
            args = [self.mode, input_tensor, torch.tensor(pads, dtype=torch.int64)]
            if constant_value is not None:
                args.append(constant_value)
            return _OnnxPadDynamicExportToOnnx.set_output_and_apply(output, *args)

        return output


class _OnnxPadDynamicExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        mode = args[0]
        args = args[1:]
        return graph.op('Pad', *args, mode_s=mode, outputs=1)


@add_converter(operation_type='Pad', version=11)
@add_converter(operation_type='Pad', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    mode = node.attributes.get('mode', 'constant')
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
    pads = tuple(node.attributes.get('pads'))
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
