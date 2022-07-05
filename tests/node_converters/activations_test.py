from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_activation(  # pylint: disable=missing-function-docstring
    activation: str, data: np.ndarray, opset_version, **kwargs
) -> None:
    test_inputs = {'input_tensor': data}

    node = onnx.helper.make_node(op_type=activation, inputs=['input_tensor'], outputs=['y'], **kwargs)
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-6,
        atol_torch_cpu_cuda=10**-6,
    )


@pytest.mark.parametrize(
    'activation,input_shape,opset_version',
    (
        ('Erf', [8, 3, 32, 32], 11),
        ('HardSigmoid', [8, 3, 32, 32], 11),
        ('HardSwish', [8, 3, 32, 32], 14),
        ('LeakyRelu', [8, 3, 32, 32], 11),
        ('LogSoftmax', [8, 3, 32, 32], 11),
        ('Softsign', [8, 3, 32, 32], 1),
        ('Softplus', [8, 3, 32, 32], 1),
        ('Relu', [8, 3, 32, 32], 11),
        ('Elu', [8, 3, 32, 32], 6),
        ('Celu', [8, 3, 32, 32], 12),
        ('Selu', [8, 3, 32, 32], 6),
        ('Sigmoid', [8, 3, 32, 32], 11),
    ),
)
def test_common_activations(  # pylint: disable=missing-function-docstring
    activation: str,
    input_shape: List[int],
    opset_version: int,
) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    _test_activation(activation, data=data, opset_version=opset_version)


@pytest.mark.parametrize(
    'input_shape,axis,opset_version',
    (
        ([8, 3, 32, 32], None, 9),
        ([8, 3, 32, 32], None, 11),
        ([8, 3, 32, 32], None, 13),
        ([8, 3, 32, 32], 0, 9),
        ([8, 3, 32, 32], 0, 11),
        ([8, 3, 32, 32], 0, 13),
        ([8, 3, 32, 32], 1, 9),
        ([8, 3, 32, 32], 1, 11),
        ([8, 3, 32, 32], 1, 13),
        ([8, 3, 32, 32], -1, 9),
        ([8, 3, 32, 32], -1, 11),
        ([8, 3, 32, 32], -1, 13),
    ),
)
@pytest.mark.parametrize('activation', ('Softmax', 'LogSoftmax'))
def test_softmax(  # pylint: disable=missing-function-docstring
    activation: str,
    input_shape: List[int],
    axis: Optional[int],
    opset_version: int,
) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    if axis is None:
        _test_activation(activation, data=data, opset_version=opset_version)
    else:
        _test_activation(activation, data=data, opset_version=opset_version, axis=axis)


@pytest.mark.parametrize(
    'input_shape,opset_version',
    (
        ([8, 3, 32, 32], 7),
        ([2, 64, 16, 16], 11),
        ([1, 16, 8, 8], 7),
        ([4, 32, 8, 8], 9),
    ),
)
def test_prelu(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    opset_version: int,
) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    slope = np.random.randn(input_shape[1], 1, 1).astype(np.float32)
    test_inputs = {'input_tensor': data, 'slope': slope}

    node = onnx.helper.make_node(op_type='PRelu', inputs=['input_tensor', 'slope'], outputs=['y'])
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-6,
        atol_torch_cpu_cuda=10**-6,
    )
