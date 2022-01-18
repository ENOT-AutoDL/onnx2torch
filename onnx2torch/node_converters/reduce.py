__all__ = [
    'OnnxReduceSumDynamicAxes',
    'OnnxReduceSumStaticAxes',
    'OnnxReduceStaticAxes',
]

from functools import partial
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.common import OnnxMapping
from onnx2torch.common import OperationConverterResult
from onnx2torch.common import SkipTorchTracing
from onnx2torch.common import get_const_value
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


def _initialize_none_dim(dim: Optional[Union[int, Tuple[int, ...]]], input_dim: int):
    if dim is None:
        return list(range(input_dim))

    return dim


def _log_sum(
        input_tensor: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
):
    dim = _initialize_none_dim(dim, input_tensor.dim())
    return torch.log(torch.sum(input_tensor, dim=dim, keepdim=keepdim))


def _log_sum_exp(
        input_tensor: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
):
    dim = _initialize_none_dim(dim, input_tensor.dim())
    return torch.logsumexp(input_tensor, dim=dim, keepdim=keepdim)


def _sum_square(
        input_tensor: torch.Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
):
    dim = _initialize_none_dim(dim, input_tensor.dim())
    return torch.sum(torch.square(input_tensor), dim=dim, keepdim=keepdim)


_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'ReduceL1': partial(torch.norm, p=1),
    'ReduceL2': partial(torch.norm, p=2),
    'ReduceLogSum': _log_sum,
    'ReduceLogSumExp': _log_sum_exp,
    'ReduceMax': torch.max,
    'ReduceMean': torch.mean,
    'ReduceMin': torch.min,
    'ReduceProd': torch.prod,
    'ReduceSum': torch.sum,
    'ReduceSumSquare': _sum_square,
}


class OnnxReduceSumDynamicAxes(nn.Module):

    def __init__(self, keepdims: int = 1, noop_with_empty_axes: int = 0):
        super().__init__()

        self.keepdims = keepdims == 1
        self.noop_with_empty_axes = noop_with_empty_axes == 1

    def _do_forward(self, input_tensor: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        if axes is None or axes.nelement() == 0:
            if self.noop_with_empty_axes:
                return input_tensor

            if not self.keepdims:
                return torch.sum(input_tensor)

            axes = list(range(input_tensor.dim()))
        else:
            axes = torch.sort(axes).values.tolist()

        return torch.sum(input_tensor, dim=axes, keepdim=self.keepdims)

    def forward(self, input_tensor: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            with SkipTorchTracing():
                args = [input_tensor, axes]
                output = self._do_forward(*args)
                if axes is None:
                    args.pop()

                return _ReduceSumExportToOnnx.set_output_and_apply(
                    output,
                    *args,
                    int(self.keepdims),
                    int(self.noop_with_empty_axes),
                )

        return self._do_forward(input_tensor, axes)


class _ReduceSumExportToOnnx(CustomExportToOnnx):

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        *args, keepdims, noop_with_empty_axes = args
        return graph.op(
            'ReduceSum',
            *args,
            noop_with_empty_axes_i=noop_with_empty_axes,
            keepdims_i=keepdims,
            outputs=1,
        )


class OnnxReduceSumStaticAxes(nn.Module):

    def __init__(
            self,
            axes: List[int],
            keepdims: int = 1,
            noop_with_empty_axes: int = 0,
    ):
        super().__init__()
        if axes is not None:
            axes = sorted(axes)

        self.keepdims = keepdims == 1
        self.noop_with_empty_axes = noop_with_empty_axes == 1
        self.axes = axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.axes is None or len(self.axes) == 0:
            if self.noop_with_empty_axes:
                return input_tensor

            if not self.keepdims:
                return self.math_op_function(input_tensor)

            self.axes = list(range(input_tensor.dim()))
            
        return torch.sum(input_tensor, dim=self.axes, keepdim=self.keepdims)


class OnnxReduceStaticAxes(nn.Module):

    def __init__(
            self,
            operation_type: str,
            axes: List[int],
            keepdims: int = 1,
    ):
        super().__init__()
        self.operation_type = operation_type
        self.math_op_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

        if axes is not None:
            axes = sorted(axes)

        self.keepdims = keepdims == 1
        self.axes = axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.axes is None or len(self.axes) == 0:
            if not self.keepdims:
                return self.math_op_function(input_tensor)

            self.axes = list(range(input_tensor.dim()))

        if self.operation_type not in ['ReduceMax', 'ReduceMin', 'ReduceProd']:
            return self.math_op_function(input_tensor, dim=self.axes, keepdim=self.keepdims)

        result = input_tensor
        for passed_dims, axis in enumerate(self.axes):
            result = self.math_op_function(
                result,
                dim=axis if self.keepdims else axis - passed_dims,
                keepdim=self.keepdims,
            )
            if isinstance(result, tuple):
                result = result[0]

        return result


@add_converter(operation_type='ReduceL1', version=1)
@add_converter(operation_type='ReduceL1', version=11)
@add_converter(operation_type='ReduceL1', version=13)
@add_converter(operation_type='ReduceL2', version=1)
@add_converter(operation_type='ReduceL2', version=11)
@add_converter(operation_type='ReduceL2', version=13)
@add_converter(operation_type='ReduceLogSum', version=1)
@add_converter(operation_type='ReduceLogSum', version=11)
@add_converter(operation_type='ReduceLogSum', version=13)
@add_converter(operation_type='ReduceLogSumExp', version=1)
@add_converter(operation_type='ReduceLogSumExp', version=11)
@add_converter(operation_type='ReduceLogSumExp', version=13)
@add_converter(operation_type='ReduceMax', version=1)
@add_converter(operation_type='ReduceMax', version=11)
@add_converter(operation_type='ReduceMax', version=12)
@add_converter(operation_type='ReduceMax', version=13)
@add_converter(operation_type='ReduceMean', version=1)
@add_converter(operation_type='ReduceMean', version=11)
@add_converter(operation_type='ReduceMean', version=13)
@add_converter(operation_type='ReduceMin', version=1)
@add_converter(operation_type='ReduceMin', version=11)
@add_converter(operation_type='ReduceMin', version=12)
@add_converter(operation_type='ReduceMin', version=13)
@add_converter(operation_type='ReduceProd', version=1)
@add_converter(operation_type='ReduceProd', version=11)
@add_converter(operation_type='ReduceProd', version=13)
@add_converter(operation_type='ReduceSum', version=1)
@add_converter(operation_type='ReduceSum', version=11)
@add_converter(operation_type='ReduceSumSquare', version=1)
@add_converter(operation_type='ReduceSumSquare', version=11)
@add_converter(operation_type='ReduceSumSquare', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    axes = node_attributes.get('axes', None)
    keepdims = node_attributes.get('keepdims', 1)

    return OperationConverterResult(
        torch_module=OnnxReduceStaticAxes(
            operation_type=node.operation_type,
            axes=axes,
            keepdims=keepdims,
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='ReduceSum', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    keepdims = node.attributes.get('keepdims', 1)
    noop_with_empty_axes = node.attributes.get('noop_with_empty_axes', 0)

    if len(node.input_values) == 2:
        try:
            axes = get_const_value(node.input_values[1], graph)
            return OperationConverterResult(
                torch_module=OnnxReduceSumStaticAxes(
                    axes=axes,
                    keepdims=keepdims,
                    noop_with_empty_axes=noop_with_empty_axes,
                ),
                onnx_mapping=OnnxMapping(
                    inputs=(node.input_values[0],),
                    outputs=node.output_values,
                ),
            )
        except KeyError:
            pass

    return OperationConverterResult(
        torch_module=OnnxReduceSumDynamicAxes(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes),
        onnx_mapping=onnx_mapping_from_node(node),
    )

