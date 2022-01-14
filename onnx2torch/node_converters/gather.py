__all__ = ['OnnxGather']

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxGather(nn.Module):
    """ONNX gather implementation (or numpy.take implementation)"""

    def __init__(self, axis: Optional[int] = 0):
        super().__init__()
        self.axis = axis

    @staticmethod
    def slice_from_axis(
            input_tensor: torch.Tensor,
            axis: int,
            indices: torch.Tensor,
    ) -> Tuple[Union[slice, torch.Tensor], ...]:
        axis = input_tensor.dim() + axis if axis < 0 else axis
        skip_axis: List[Union[slice, torch.Tensor]] = [slice(None)] * axis
        skip_axis.append(indices)
        return tuple(skip_axis)

    def forward(self, input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # pytorch Gather differs from onnx Gather, onnx gather work like numpy.take
        # But torch.take does not support different axis. So we make it by yourself
        # numpy.take is input_data[:, :, indices] where we pass NONE slices AXIS time
        slice_for_take = self.slice_from_axis(input_tensor, self.axis, indices)
        return input_tensor[slice_for_take]


@add_converter(operation_type='Gather', version=1)
@add_converter(operation_type='Gather', version=11)
@add_converter(operation_type='Gather', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    torch_module = OnnxGather(
        axis=axis,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
