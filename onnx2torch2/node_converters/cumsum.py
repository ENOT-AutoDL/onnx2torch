__all__ = [
    'OnnxCumSum',
]
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
    # single item shift
    slice_index, insertion = [[slice(None)] * len(input_tensor.shape)] * 2
    insert_dim_size = input_tensor.shape[insert_dim]

    slice_index[insert_dim] = slice(0, -1)
    slice_index = tuple(slice_index)
    tensor_slice = input_tensor[slice_index]

    insert_index = torch.arange(start=1, end=insert_dim_size, dtype=torch.int64, device=input_tensor.device)
    index_shape = [1] * len(input_tensor.shape)
    index_shape[insert_dim] = insert_dim_size - 1

    insert_index = torch.reshape(insert_index, index_shape)
    insert_index = insert_index + torch.zeros_like(tensor_slice, dtype=torch.int64, device=input_tensor.device)

    input_tensor = torch.scatter(
        input=input_tensor,
        dim=insert_dim,
        index=insert_index,
        src=tensor_slice,
    )

    insertion[insert_dim] = slice(0, 1)
    insertion = tuple(insertion)
    input_tensor[insertion] = 0

    return input_tensor


class OnnxCumSum(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(
        self,
        exclusive: bool = False,
        reverse: bool = False,
    ):
        super().__init__()
        self.exclusive = exclusive
        self.reverse = reverse

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        axis: torch.Tensor,
    ) -> torch.Tensor:
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
