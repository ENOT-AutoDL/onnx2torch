__all__ = ['OnnxGeneralLinear']

import torch
import torch.nn.functional as F
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult


class OnnxGeneralLinear(nn.Linear, OnnxToTorchModule):
    """General Linear layer with functionality of ONNX GEMM node.

    For additional info https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            trans_a: int,
    ):

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        # If != 0 transpose input before matmul
        self.trans_a = trans_a

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        input_tensor = torch.transpose(input_tensor, 0, 1) if self.trans_a != 0 else input_tensor
        return F.linear(input_tensor, self.weight, self.bias)

    @classmethod
    def maybe_create_simple_linear(
            cls,
            in_features: int,
            out_features: int,
            bias: bool,
            trans_a: int,
    ):
        if trans_a == 0:
            return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        return OnnxGeneralLinear(in_features, out_features, bias, trans_a)


@add_converter(operation_type='Gemm', version=9)
@add_converter(operation_type='Gemm', version=11)
@add_converter(operation_type='Gemm', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    weights_value_name = node.input_values[1]
    weights = graph.initializers[weights_value_name]
    weights = weights.to_torch()

    # An empty string may be used in the place of an actual argument's name to indicate a missing argument.
    # See ONNX documentation
    if len(node.input_values) == 3 and node.input_values[2] != '':
        bias_value_name = node.input_values[2]
        bias = graph.initializers[bias_value_name]
        bias = bias.to_torch()
    else:
        bias = None

    node_attributes = node.attributes
    alpha = node_attributes.get('alpha', 1.0)
    beta = node_attributes.get('beta', 1.0)
    trans_a = node_attributes.get('transA', 0)
    trans_b = node_attributes.get('transB', 0)

    if trans_b == 0:
        in_features, out_features = weights.shape[0], weights.shape[1]
    else:
        in_features, out_features = weights.shape[1], weights.shape[0]

    torch_module = OnnxGeneralLinear.maybe_create_simple_linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias is not None,
        trans_a=trans_a,
    )

    with torch.no_grad():
        # In pytorch weights are transposed by default (see documentation)
        # So we transpose weights before matmul if trans_b == 0
        weights = torch.transpose(weights, 0, 1) if trans_b == 0 else weights
        weights = weights * alpha
        torch_module.weight.data = weights
        if bias is not None:
            bias = bias * beta
            torch_module.bias.data = bias

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
