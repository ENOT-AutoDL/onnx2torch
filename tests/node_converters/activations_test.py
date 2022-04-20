from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_activation(activation: str, data: np.ndarray, opset_version, **kwargs) -> None:
    test_inputs = {'input_tensor': data}

    node = onnx.helper.make_node(op_type=activation, inputs=['input_tensor'], outputs=['y'], **kwargs)
    model = make_model_from_nodes(
        nodes=node, initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'activation,input_shape',
    (
            ('Erf', [8, 3, 32, 32]),
            ('HardSigmoid', [8, 3, 32, 32]),
            ('LeakyRelu', [8, 3, 32, 32]),
            ('LogSoftmax', [8, 3, 32, 32]),
            ('Relu', [8, 3, 32, 32]),
            ('Sigmoid', [8, 3, 32, 32]),
    ),
)
def test_common_activations(activation: str, input_shape: List[int]) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    _test_activation(activation, data=data, opset_version=11)


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
def test_softmax(activation: str, input_shape: List[int], axis: Optional[int], opset_version: int) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    if axis is None:
        _test_activation(activation, data=data, opset_version=opset_version)
    else:
        _test_activation(activation, data=data, opset_version=opset_version, axis=axis)
