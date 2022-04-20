__all__ = ['OnnxErf', 'OnnxHardSigmoid', 'OnnxSoftmaxV1V11']

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxErf(nn.Module, OnnxToTorchModule):

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.erf(input_tensor)


class OnnxHardSigmoid(nn.Module, OnnxToTorchModule):
    def __init__(self, alpha: float = 0.2, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.clip(self.alpha * input_tensor + self.beta, min=0.0, max=1.0)


class OnnxSoftmaxV1V11(nn.Module, OnnxToTorchModule):
    def __init__(self, axis: int = 1, is_log: bool = False):
        super().__init__()
        self.axis = axis
        self.is_log = is_log

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        shape = input_tensor.shape
        result = torch.flatten(input_tensor, start_dim=self.axis)
        result = torch.log_softmax(result, -1) if self.is_log else torch.softmax(result, -1)

        return torch.reshape(result, shape)


@add_converter(operation_type='Erf', version=9)
@add_converter(operation_type='Erf', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxErf(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='HardSigmoid', version=1)
@add_converter(operation_type='HardSigmoid', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    alpha = node.attributes.get('alpha', 0.2)
    beta = node.attributes.get('beta', 0.5)

    return OperationConverterResult(
        torch_module=OnnxHardSigmoid(alpha=alpha, beta=beta),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='HardSwish', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=nn.Hardswish(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='LeakyRelu', version=1)
@add_converter(operation_type='LeakyRelu', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    alpha = node.attributes.get('alpha', 0.01)

    return OperationConverterResult(
        torch_module=nn.LeakyReLU(negative_slope=alpha),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='LogSoftmax', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    dim = node.attributes.get('axis', -1)

    return OperationConverterResult(
        torch_module=nn.LogSoftmax(dim=dim),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='LogSoftmax', version=1)
@add_converter(operation_type='LogSoftmax', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 1)

    return OperationConverterResult(
        torch_module=OnnxSoftmaxV1V11(axis=axis, is_log=True),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Relu', version=6)
@add_converter(operation_type='Relu', version=13)
@add_converter(operation_type='Relu', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=nn.ReLU(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Elu', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    alpha = node.attributes.get('alpha', 1.0)

    return OperationConverterResult(
        torch_module=nn.ELU(alpha=alpha),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Celu', version=12)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    alpha = node.attributes.get('alpha', 1.0)

    return OperationConverterResult(
        torch_module=nn.CELU(alpha=alpha),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Selu', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:   # pylint: disable=unused-argument
    default_alpha = 1.67326319217681884765625
    default_gamma = 1.05070102214813232421875

    alpha = node.attributes.get('alpha', default_alpha)
    gamma = node.attributes.get('gamma', default_gamma)

    if alpha != default_alpha:
        raise ValueError(f'alpha parameter must be {default_alpha}, not {alpha}')
    if gamma != default_gamma:
        raise ValueError(f'gamma parameter must be {default_gamma}, not {gamma}')

    return OperationConverterResult(
        torch_module=nn.SELU(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Sigmoid', version=1)
@add_converter(operation_type='Sigmoid', version=6)
@add_converter(operation_type='Sigmoid', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=nn.Sigmoid(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Softmax', version=1)
@add_converter(operation_type='Softmax', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 1)

    return OperationConverterResult(
        torch_module=OnnxSoftmaxV1V11(axis=axis),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Softmax', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    dim = node.attributes.get('axis', -1)

    return OperationConverterResult(
        torch_module=torch.nn.Softmax(dim=dim),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Softsign', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=torch.nn.Softsign(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )

@add_converter(operation_type='Softplus', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    beta = node.attributes.get('beta', 1.0)
    threshold = node.attributes.get('threshold', 20.0)

    return OperationConverterResult(
        torch_module=torch.nn.Softplus(beta=beta, threshold=threshold),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
