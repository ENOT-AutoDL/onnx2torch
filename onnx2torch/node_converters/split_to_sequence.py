__all__ = ['OnnxSplitToSequence']

from typing import Any
from typing import Dict
from typing import List
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


class OnnxSplitToSequence(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, axis: int = 0, keepdims: int = 1):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        del opset_version
        return {'axis_i': self.axis, 'keepdims_i': self.keepdims}

    @staticmethod
    def _split_to_sequence(
        inputs: torch.Tensor,
        split: Optional[torch.IntTensor] = None,
        axis: int = 0,
        keepdims: int = 1,
    ) -> List[torch.Tensor]:
        del keepdims

        split_size_or_sections = split.tolist() if split is not None else 1
        return torch.split(tensor=inputs, split_size_or_sections=split_size_or_sections, dim=axis)

    # pylint: disable=missing-function-docstring
    def forward(self, inputs: torch.Tensor, split: Optional[torch.IntTensor] = None) -> List[torch.Tensor]:
        if torch.onnx.is_in_onnx_export():

            def _stub_forward():
                return torch.Tensor()

            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(_stub_forward, 'SplitToSequence', inputs, split, onnx_attrs)

        return self._split_to_sequence(inputs=inputs, split=split, axis=self.axis, keepdims=self.keepdims)


@add_converter(operation_type='SplitToSequence', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    del graph

    axis: int = node.attributes.get('axis', 0)
    keepdims: int = node.attributes.get('keepdims', 1)

    if len(node.input_values) == 1 and keepdims == 0:  # no split and keepdim = 0
        raise NotImplementedError('SplitToSequence without split argument and keepdims == 0 is not implemented')

    return OperationConverterResult(
        torch_module=OnnxSplitToSequence(axis=axis, keepdims=keepdims),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
