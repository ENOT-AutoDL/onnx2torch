from copy import deepcopy

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


def _arbitrary_dim_shift_and_insert_zero(
        input_tensor: torch.Tensor,
        insert_dim: int,
) -> torch.Tensor:

    t_shape = input_tensor.shape
    insert_dim = torch.arange(start=0, end=len(t_shape), dtype=torch.int64)[insert_dim]
    insert_dim_size = t_shape[insert_dim]

    rh_tensor = torch.index_select(
        input=input_tensor,
        dim=insert_dim,
        index=torch.arange(0, insert_dim_size-1, dtype=torch.int32, device=input_tensor.device),
    )

    insert_index = torch.arange(1, insert_dim_size, dtype=torch.int64, device=input_tensor.device)
    index_shape = [1] * len(t_shape)
    index_shape[insert_dim] = insert_dim_size - 1

    insert_index = torch.reshape(insert_index, index_shape)
    insert_index = insert_index + torch.zeros_like(rh_tensor, dtype=torch.int64, device=input_tensor.device)

    input_tensor = torch.scatter(
        input=input_tensor,
        dim=insert_dim,
        index=insert_index,
        src=rh_tensor,
    )

    shuffled_input_tensor = torch.transpose(input_tensor, len(t_shape) - 1, insert_dim)
    shuffled_input_tensor[..., 0] = 0
    input_tensor = torch.transpose(shuffled_input_tensor, len(t_shape) - 1, insert_dim)
    return input_tensor


class OnnxCumSum(nn.Module, OnnxToTorchModule):
    def __init__(
            self,
            exclusive: bool = False,
            reverse: bool = False,
    ):
        super().__init__()
        self.exclusive = exclusive
        self.reverse = reverse

    def forward(self, input_tensor: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        axis = axis.item()
        if self.reverse:
            input_tensor = torch.flip(input_tensor, dims=(axis,))

        if self.exclusive:
            input_tensor = _arbitrary_dim_shift_and_insert_zero(input_tensor, insert_dim=axis)

        input_tensor = torch.cumsum(input_tensor, dim=axis)

        if self.reverse:
            input_tensor = torch.flip(input_tensor, dims=(axis,))

        return input_tensor


@add_converter(operation_type='CumSum', version=11)
@add_converter(operation_type='CumSum', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    exclusive = bool(node_attributes.get('exclusive', 0))
    reverse = bool(node_attributes.get('reverse', 1))

    return OperationConverterResult(
        torch_module=OnnxCumSum(exclusive, reverse),
        onnx_mapping=onnx_mapping_from_node(node),
    )
