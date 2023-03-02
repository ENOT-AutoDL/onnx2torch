__all__ = [
    'OnnxGemm',
]

from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxGemm(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, alpha: float, beta: float, trans_a: bool, trans_b: bool):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.trans_a = trans_a
        self.trans_b = trans_b

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        input_c: Optional[torch.Tensor] = None,
    ):
        if self.trans_a:
            input_a = torch.transpose(input_a, dim0=0, dim1=1)
        if self.trans_b:
            input_b = torch.transpose(input_b, dim0=0, dim1=1)

        output = input_a @ input_b * self.alpha
        if input_c is not None:
            output += input_c * self.beta

        return output


@add_converter(operation_type='Gemm', version=9)
@add_converter(operation_type='Gemm', version=11)
@add_converter(operation_type='Gemm', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    a_name = node.input_values[0]
    b_name = node.input_values[1]
    c_name = node.input_values[2] if len(node.input_values) > 2 else None

    node_attributes = node.attributes
    alpha = node_attributes.get('alpha', 1.0)
    beta = node_attributes.get('beta', 1.0)
    trans_a = node_attributes.get('transA', 0) != 0
    trans_b = node_attributes.get('transB', 0) != 0

    if not trans_a and b_name in graph.initializers and (c_name is None or c_name in graph.initializers):
        if c_name is None:
            bias = None
        else:
            bias = graph.initializers[c_name]
            bias = bias.to_torch()

        if bias is None or bias.dim() == 1:
            weights = graph.initializers[b_name]
            weights = weights.to_torch()
            if not trans_b:
                weights = weights.T

            in_features, out_features = weights.shape[1], weights.shape[0]
            torch_module = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias is not None,
            )

            with torch.no_grad():
                weights = weights * alpha
                torch_module.weight.data = weights
                if bias is not None:
                    bias = bias * beta
                    torch_module.bias.data = bias

            return OperationConverterResult(
                torch_module=torch_module,
                onnx_mapping=OnnxMapping(
                    inputs=(a_name,),
                    outputs=node.output_values,
                ),
            )

    return OperationConverterResult(
        torch_module=OnnxGemm(alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b),
        onnx_mapping=onnx_mapping_from_node(node),
    )
