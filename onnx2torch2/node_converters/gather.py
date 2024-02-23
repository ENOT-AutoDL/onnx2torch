__all__ = [
    'OnnxGather',
    'OnnxGatherElements',
    'OnnxGatherND',
]

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport
from onnx2torch.utils.indices import upcast_indices


class OnnxGatherElements(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int = 0):
        super().__init__()
        self._axis = axis

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.gather(input_tensor, dim=self._axis, index=upcast_indices(indices))


class OnnxGather(nn.Module, OnnxToTorchModuleWithCustomExport):
    """ONNX Gather implementation (or numpy.take implementation)."""

    def __init__(self, axis: int = 0):
        super().__init__()
        self._axis = axis

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        return {'axis_i': self._axis}

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
            slice_for_take = self.slice_from_axis(input_tensor, self._axis, indices)
            return input_tensor[slice_for_take]

        if torch.onnx.is_in_onnx_export():
            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(_forward, 'Gather', input_tensor, indices, onnx_attrs)

        return _forward()


class OnnxGatherND(nn.Module, OnnxToTorchModuleWithCustomExport):
    """ONNX GatherND implementation."""

    def __init__(self, batch_dims: int = 0):
        super().__init__()
        self._batch_dims: int = batch_dims

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        onnx_attrs: Dict[str, Any] = {}

        if opset_version == 11:
            if self._batch_dims != 0:
                raise ValueError(f'GatherND from opset 11 does not support batch_dims != 0, got {self._batch_dims}')
            return onnx_attrs

        onnx_attrs['batch_dims_i'] = self._batch_dims
        return onnx_attrs

    def forward(self, input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:  # pylint: disable=C0116
        def _forward():
            return self._gather_nd(data=input_tensor, indices=indices, batch_dims=self._batch_dims)

        if torch.onnx.is_in_onnx_export():
            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(_forward, 'GatherND', input_tensor, indices, onnx_attrs)

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
        indices_ = torch.split(
            tensor=indices.reshape(total_samples, m).transpose(0, 1),
            split_size_or_sections=1,
            dim=0,
        )

        return data[indices_].reshape(output_shape).contiguous()


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


@add_converter(operation_type='GatherND', version=11)
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
