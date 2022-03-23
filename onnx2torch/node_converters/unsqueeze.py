__all__ = [
    'OnnxUnsqueezeStaticAxes',
    'OnnxUnsqueezeDynamicAxes',
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


class OnnxUnsqueezeStaticAxes(nn.Module, OnnxToTorchModule):

    def __init__(self, axes: Optional[List[int]] = None):
        super().__init__()
        self.axes = sorted(axes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        result = input_tensor
        for axes_id in self.axes:
            result = torch.unsqueeze(result, dim=axes_id)

        return result


class OnnxUnsqueezeDynamicAxes(nn.Module, OnnxToTorchModuleWithCustomExport):

    @staticmethod
    def _do_forward(input_tensor: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        result = input_tensor
        for axes_id in torch.sort(axes).values:
            result = torch.unsqueeze(result, dim=axes_id)

        return result

    def forward(self, input_tensor: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        output = self._do_forward(input_tensor, axes)
        if torch.onnx.is_in_onnx_export():
            return _UnsqueezeExportToOnnx.set_output_and_apply(output, input_tensor, axes)

        return output


class _UnsqueezeExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Unsqueeze', *args, outputs=1)


@add_converter(operation_type='Unsqueeze', version=1)
@add_converter(operation_type='Unsqueeze', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axes = node.attributes['axes']
    return OperationConverterResult(
        torch_module=OnnxUnsqueezeStaticAxes(axes=axes),
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type='Unsqueeze', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    try:
        axes = get_const_value(node.input_values[1], graph)
        return OperationConverterResult(
            torch_module=OnnxUnsqueezeStaticAxes(axes=axes),
            onnx_mapping=OnnxMapping(
                inputs=(node.input_values[0],),
                outputs=node.output_values,
            ),
        )
    except KeyError:
        pass

    return OperationConverterResult(
        torch_module=OnnxUnsqueezeDynamicAxes(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
