__all__ = [
    'OnnxUnsqueezeStaticAxes',
    'OnnxUnsqueezeDynamicAxes',
]

from typing import List

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxUnsqueezeStaticAxes(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, axes: List[int]):
        super().__init__()
        self._axes = sorted(axes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        result = input_tensor
        for axes_id in self._axes:
            result = torch.unsqueeze(result, dim=axes_id)

        return result


class OnnxUnsqueezeDynamicAxes(  # pylint: disable=missing-class-docstring
    nn.Module,
    OnnxToTorchModuleWithCustomExport,
):
    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        axes: torch.Tensor,
    ) -> torch.Tensor:
        def _forward():
            result = input_tensor
            for axes_id in torch.sort(axes).values:
                result = torch.unsqueeze(result, dim=axes_id)

            return result

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'Unsqueeze', input_tensor, axes, {})

        return _forward()


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
        axes = axes.tolist()
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
