__all__ = [
    'OnnxSqueezeStaticAxes',
    'OnnxSqueezeDynamicAxes',
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
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxSqueezeStaticAxes(nn.Module, OnnxToTorchModule):

    def __init__(self, axes: Optional[List[int]] = None):
        super().__init__()
        if axes is not None:
            axes = sorted(axes, reverse=True)

        self.axes = axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not self.axes:
            return torch.squeeze(input_tensor)

        result = input_tensor
        for axes_id in self.axes:
            result = torch.squeeze(result, dim=axes_id)

        return result


class OnnxSqueezeDynamicAxes(nn.Module, OnnxToTorchModuleWithCustomExport):

    @staticmethod
    def _do_forward(input_tensor: torch.Tensor, axes: Optional[torch.Tensor]) -> torch.Tensor:
        if axes is None or axes.nelement() == 0:
            return torch.squeeze(input_tensor)

        result = input_tensor
        for axes_id in torch.sort(axes, descending=True).values:
            result = torch.squeeze(result, dim=axes_id)

        return result

    def forward(self, input_tensor: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self._do_forward(input_tensor, axes)
        if torch.onnx.is_in_onnx_export():
            args = [input_tensor]
            if axes is not None:
                args.append(axes)

            return _SqueezeExportToOnnx.set_output_and_apply(output, *args)

        return output


class _SqueezeExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Squeeze', *args, outputs=1)


@add_converter(operation_type='Squeeze', version=1)
@add_converter(operation_type='Squeeze', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axes = node.attributes.get('axes', None)
    return OperationConverterResult(
        torch_module=OnnxSqueezeStaticAxes(axes=axes),
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type='Squeeze', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if len(node.input_values) == 2:
        try:
            axes = get_const_value(node.input_values[1], graph)
            return OperationConverterResult(
                torch_module=OnnxSqueezeStaticAxes(axes=axes),
                onnx_mapping=OnnxMapping(
                    inputs=(node.input_values[0],),
                    outputs=node.output_values,
                ),
            )
        except KeyError:
            pass

    return OperationConverterResult(
        torch_module=OnnxSqueezeDynamicAxes(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
