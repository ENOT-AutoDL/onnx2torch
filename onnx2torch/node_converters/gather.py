__all__ = [
    'OnnxGather',
    'OnnxGatherElements',
    'OnnxGatherND',
    'OnnxGatherNDOpset11',
]

from typing import List
from typing import Tuple
from typing import Union

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.indices import upcast_indices


class OnnxGatherElements(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.gather(input_tensor, dim=self.axis, index=upcast_indices(indices))


class OnnxGather(nn.Module, OnnxToTorchModuleWithCustomExport):
    """ONNX gather implementation (or numpy.take implementation)"""

    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    @staticmethod
    def slice_from_axis(  # pylint: disable=missing-docstring
        input_tensor: torch.Tensor,
        axis: int,
        indices: torch.Tensor,
    ) -> Tuple[Union[slice, torch.Tensor], ...]:
        axis = input_tensor.dim() + axis if axis < 0 else axis
        skip_axis: List[Union[slice, torch.Tensor]] = [slice(None)] * axis
        skip_axis.append(upcast_indices(indices))
        return tuple(skip_axis)

    def forward(  # pylint: disable=missing-function-docstring
        self, input_tensor: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        def _forward():
            # pytorch Gather differs from onnx Gather, onnx gather work like numpy.take
            # But torch.take does not support different axis. So we make it by yourself
            # numpy.take is input_data[:, :, indices] where we pass NONE slices AXIS time
            slice_for_take = self.slice_from_axis(input_tensor, self.axis, indices)
            return input_tensor[slice_for_take]

        if torch.onnx.is_in_onnx_export():
            return _GatherExportToOnnx.set_forward_and_apply(_forward, input_tensor, indices, self.axis)

        return _forward()


class _GatherExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        input_tensor, indices, axis = args
        return graph.op('Gather', input_tensor, indices, axis_i=axis, outputs=1)


class OnnxGatherND(nn.Module, OnnxToTorchModuleWithCustomExport):
    """ONNX GatherND implementation."""

    def __init__(self, batch_dims: int = 0):
        super().__init__()
        self.batch_dims: int = batch_dims

    def forward(self, input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:  # pylint: disable=C0116
        def _forward():
            return self._gather_nd(data=input_tensor, indices=indices, batch_dims=self.batch_dims)

        if torch.onnx.is_in_onnx_export():
            return _GatherNDExportToOnnx.set_forward_and_apply(_forward, input_tensor, indices, self.batch_dims)

        return _forward()

    @staticmethod
    def _gather_nd(data: torch.Tensor, indices: torch.Tensor, batch_dims: int) -> torch.Tensor:
        if batch_dims != 0:
            raise NotImplementedError('GatherND for batch_dims != 0 is not implemented')

        r, m = len(data.shape), indices.shape[-1]  # pylint: disable=C0103
        if m > r or m < 1:
            raise ValueError(
                f'The last dimension of indices should have a value between 1 (inclusive) and data rank (inclusive), '
                f'got {m} and {r} respectively'
            )

        total_samples = indices.shape[:-1].numel()
        output_shape = indices.shape[:-1] + data.shape[m:]
        _indices = torch.split(
            tensor=indices.reshape(total_samples, m).transpose(0, 1),
            split_size_or_sections=1,
            dim=0,
        )

        return data[_indices].reshape(output_shape).contiguous()


class _GatherNDExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        input_tensor, indices, batch_dims = args
        return graph.op('GatherND', input_tensor, indices, batch_dims_i=batch_dims, outputs=1)


class OnnxGatherNDOpset11(OnnxGatherND):
    """ONNX GatherND implementation for opset 11 (without batch_dims parameter)."""

    def __init__(self):
        super().__init__(batch_dims=0)

    def forward(self, input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:  # pylint: disable=C0116
        def _forward():
            return self._gather_nd(data=input_tensor, indices=indices, batch_dims=0)

        if torch.onnx.is_in_onnx_export():
            return _GatherNDOpset11ExportToOnnx.set_forward_and_apply(_forward, input_tensor, indices)

        return _forward()


class _GatherNDOpset11ExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        input_tensor, indices = args
        return graph.op('GatherND', input_tensor, indices, outputs=1)


@add_converter(operation_type='Gather', version=1)
@add_converter(operation_type='Gather', version=11)
@add_converter(operation_type='Gather', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    torch_module = OnnxGather(
        axis=axis,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='GatherElements', version=11)
@add_converter(operation_type='GatherElements', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    torch_module = OnnxGatherElements(
        axis=axis,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='GatherND', version=12)
@add_converter(operation_type='GatherND', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    batch_dims = node.attributes.get('batch_dims', 0)
    torch_module = OnnxGatherND(
        batch_dims=batch_dims,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='GatherND', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    torch_module = OnnxGatherNDOpset11()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
