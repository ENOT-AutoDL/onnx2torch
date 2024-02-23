__all__ = [
    'OnnxShape',
]

from typing import Any
from typing import Dict
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxShape(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, start: int = 0, end: Optional[int] = None):
        super().__init__()
        self._start = start
        self._end = end

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        if opset_version < 15:
            if self._start != 0:
                raise ValueError(f'Shape from opset < 15 does not support start != 0, got {self._start}')
            if self._end is not None:
                raise ValueError(f'Shape from opset < 15 does not support end != None, got {self._end}')
            return {}

        onnx_attrs: Dict[str, Any] = {'start_i': self._start}
        if self._end:
            onnx_attrs['end_i'] = self._end

        return onnx_attrs

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        def _forward():
            return torch.tensor(
                input_tensor.shape[self._start : self._end],
                device=input_tensor.device,
            )

        if torch.onnx.is_in_onnx_export():
            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(_forward, 'Shape', input_tensor, onnx_attrs)

        return _forward()


@add_converter(operation_type='Shape', version=1)
@add_converter(operation_type='Shape', version=13)
@add_converter(operation_type='Shape', version=15)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxShape(
            start=node.attributes.get('start', 0),
            end=node.attributes.get('end', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
