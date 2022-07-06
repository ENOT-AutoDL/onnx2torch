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
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxSqueezeStaticAxes(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, axes: Optional[List[int]] = None):
        super().__init__()
        if axes is not None:
            axes = sorted(axes, reverse=True)

        self.axes = axes

    @staticmethod
    def _do_forward(input_tensor: torch.Tensor, axes: Optional[List[int]]) -> torch.Tensor:
        if not axes:
            return torch.squeeze(input_tensor)

        result = input_tensor
        for axes_id in axes:
            result = torch.squeeze(result, dim=axes_id)

        return result

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        output = self._do_forward(input_tensor, self.axes)
        if torch.onnx.is_in_onnx_export() and get_onnx_version() >= 13:
            args = [input_tensor]
            if self.axes:
                axes = torch.tensor(self.axes, device=input_tensor.device, dtype=torch.int64)
                args.append(axes)
            return _SqueezeDynamicAxesExportToOnnx.set_output_and_apply(output, *args)

        return output


class OnnxSqueezeDynamicAxes(  # pylint: disable=missing-class-docstring
    nn.Module,
    OnnxToTorchModuleWithCustomExport,
):
    @staticmethod
    def is_empty_axes(axes: torch.Tensor) -> bool:  # pylint: disable=missing-function-docstring
        return axes is None or axes.nelement() == 0

    @staticmethod
    def _do_forward(input_tensor: torch.Tensor, axes: Optional[torch.Tensor]) -> torch.Tensor:
        if OnnxSqueezeDynamicAxes.is_empty_axes(axes):
            return torch.squeeze(input_tensor)

        result = input_tensor
        for axes_id in torch.sort(axes, descending=True).values:
            result = torch.squeeze(result, dim=axes_id)

        return result

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        axes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self._do_forward(input_tensor, axes)
        if torch.onnx.is_in_onnx_export():
            args = [input_tensor]
            if not self.is_empty_axes(axes):
                args.append(axes)

            return _SqueezeDynamicAxesExportToOnnx.set_output_and_apply(output, *args)

        return output


class _SqueezeDynamicAxesExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
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
    return OperationConverterResult(
        torch_module=OnnxSqueezeDynamicAxes(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
