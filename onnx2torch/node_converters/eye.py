__all__ = [
    'OnnxEyeLike',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class OnnxEyeLike(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, eyelike_k: int, dtype: torch.dtype = None):
        super().__init__()
        self.dtype = dtype
        self.eyelike_k = eyelike_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(x.shape) != 2:
            raise ValueError('OnnxEyeLike only support 2-D tensor')

        input_value_n = x.size(dim=0)
        input_value_m = x.size(dim=1)
        if self.eyelike_k > input_value_n:
            raise ValueError(
                f'Error EyeLike Attribute k value, the k value is {self.eyelike_k},'
                'but x shape is {(input_value_n, input_value_m)}'
            )

        if self.eyelike_k == 0:
            return torch.eye(input_value_n, input_value_m, dtype=self.dtype)

        k_tensor = torch.zeros(input_value_n, self.eyelike_k, dtype=self.dtype)
        eye_tensor = torch.eye(input_value_n, input_value_m - self.eyelike_k, dtype=self.dtype)
        return torch.concat([k_tensor, eye_tensor], axis=1)


@add_converter(operation_type='EyeLike', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    eyelike_k = node_attributes.get('k', 0)
    dtype = node_attributes.get('dtype', torch.float32)
    return OperationConverterResult(
        torch_module=OnnxEyeLike(dtype=dtype, eyelike_k=eyelike_k),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
