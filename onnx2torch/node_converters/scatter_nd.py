__all__ = [
    'OnnxScatterND',
]

from enum import Enum
from typing import Any
from typing import Dict

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


class ReductionOnnxAttr(Enum):
    """
    Representation of new attribute in 16 opset: `reduction`.

    Type of reduction to apply: none (default), add, mul.
    - `none`: no reduction applied.
    - `add`: reduction using the addition operation.
    - `mul`: reduction using the multiplication operation.
    """

    NONE = 'none'
    ADD = 'add'
    MUL = 'mul'


class OnnxScatterND(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, reduction: ReductionOnnxAttr):
        super().__init__()

        if reduction != ReductionOnnxAttr.NONE:
            raise NotImplementedError(f'ScatterND with reduction attribute "{reduction.value}" is not implemented')

        self._reduction = reduction

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        onnx_attrs: Dict[str, Any] = {}

        if opset_version < 16:
            if self._reduction != ReductionOnnxAttr.NONE:
                raise ValueError(
                    'ScatterND from opset < 16 does not support'
                    f'reduction attribute != {ReductionOnnxAttr.NONE.value},'
                    f'got {self._reduction.value}'
                )
            return onnx_attrs

        onnx_attrs['reduction_s'] = self._reduction.value
        return onnx_attrs

    def forward(  # pylint: disable=missing-function-docstring
        self,
        data: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        def _forward():
            # There is no scatter nd for torch, use following formula:
            # https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterND
            output = data.clone()

            ind_dim = indices.dim()
            # last dimension is a partial-index into data
            output_indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
            # update.shape = indices.shape[0:ind_dim-1] ++ data.shape[indices.shape[-1]:data.dim()-1]
            output_updates = updates.reshape((-1, *updates.shape[ind_dim - 1 :]))
            output[output_indices] = output_updates

            return output

        if torch.onnx.is_in_onnx_export():
            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(_forward, 'ScatterND', data, indices, updates, onnx_attrs)

        return _forward()


@add_converter(operation_type='ScatterND', version=11)
@add_converter(operation_type='ScatterND', version=13)
@add_converter(operation_type='ScatterND', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    reduction = ReductionOnnxAttr(node_attributes.get('reduction', 'none'))
    return OperationConverterResult(
        torch_module=OnnxScatterND(reduction=reduction),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
