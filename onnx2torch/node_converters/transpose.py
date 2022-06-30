__all__ = [
    'OnnxTranspose',
]

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult


class OnnxTranspose(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, perm: Optional[List[int]] = None):
        super().__init__()
        self.perm = perm

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if self.perm is None:
            self.perm = list(range(input_tensor.dim()))[::-1]

        return input_tensor.permute(self.perm)


@add_converter(operation_type='Transpose', version=1)
@add_converter(operation_type='Transpose', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    input_values = [node.input_values[0]]
    perm_value_name = node.input_values[1] if len(node.input_values) > 1 else None

    if perm_value_name is not None:
        perm = graph.initializers[perm_value_name].to_torch().tolist()
    else:
        perm = node.attributes.get('perm', None)
        if perm is not None:
            perm = torch.tensor(perm, dtype=torch.long).tolist()

    return OperationConverterResult(
        torch_module=OnnxTranspose(perm=perm),
        onnx_mapping=OnnxMapping(
            inputs=tuple(input_values),
            outputs=node.output_values,
        ),
    )
