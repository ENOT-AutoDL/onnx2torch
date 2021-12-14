__all__ = ['OnnxScatterND']

from typing import Optional

import numpy as np
import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.common import skip_torch_tracing
from onnx2torch.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxScatterND(nn.Module):
    def __init__(self, reduction: Optional[str] = None):
        super().__init__()
        self.reduction = reduction

    def _do_forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        # There is no scatter nd for torch, use following formula:
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterND
        output = data.clone()

        update_indices = indices.shape[:-1]
        for idx in np.ndindex(update_indices):
            if self.reduction == 'sum':
                output[indices[idx]] += updates[idx]
            elif self.reduction == 'mul':
                output[indices[idx]] *= updates[idx]
            elif self.reduction is None:
                output[indices[idx]] = updates[idx]

        return output

    def forward(self, *args) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            with skip_torch_tracing():
                output = self._do_forward(*args)

                export_kwargs = {}
                if self.reduction is not None:
                    export_kwargs['reduction'] = self.reduction

                return _ScatterNDExportToOnnx.set_output_and_apply(output, *args, **export_kwargs)

        return self._do_forward(*args)


class _ScatterNDExportToOnnx(CustomExportToOnnx):
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args, **kwargs) -> torch_C.Value:
        return graph.op('ScatterND', *args, **kwargs, outputs=1)


@add_converter(operation_type='ScatterND', version=11)
@add_converter(operation_type='ScatterND', version=13)
@add_converter(operation_type='ScatterND', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    reduction = node_attributes.get('reduction', None)

    return OperationConverterResult(
        torch_module=OnnxScatterND(reduction=reduction),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
