from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_functions(function: str, data: np.ndarray, opset_version, **kwargs) -> None:
    test_inputs = {'input_tensor': data}

    node = onnx.helper.make_node(op_type=function, inputs=['input_tensor'], outputs=['y'], **kwargs)
    model = make_model_from_nodes(
        nodes=node, initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'function,input_shape',
    (
            ('Ceil', [8, 3, 32, 32]),
            ('Floor', [8, 3, 32, 32]),
            ('Round', [8, 3, 32, 32]),
    ),
)
def test_roundings(function: str, input_shape: List[int]) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    _test_functions(function, data=data, opset_version=11)


@pytest.mark.parametrize(
    'function,input_shape',
    (
            ('Abs', [8, 3, 32, 32]),
            ('Cos', [8, 3, 32, 32]),
            ('Exp', [8, 3, 32, 32]),
            ('Log', [8, 3, 32, 32]),
            ('Sign', [8, 3, 32, 32]),
            ('Sin', [8, 3, 32, 32]),
            ('Tan', [8, 3, 32, 32])
    ),
)
def test_common_functions(function: str, input_shape: List[int]) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    if function == 'Log':
        data[data <= 0] = 10**-4
    _test_functions(function, data=data, opset_version=11)


@pytest.mark.parametrize(
    'function,input_shape',
    (
            ('Acos', [8, 3, 32, 32]),
            ('Asin', [8, 3, 32, 32]),
            ('Atan', [8, 3, 32, 32]),
    ),
)
def test_arc_functions(function: str, input_shape: List[int]) -> None:
    if function in ['Acos', 'Asin']:
        data = np.random.uniform(-1, 1, input_shape).astype(np.float32)
    else:
        data = np.random.randn(*input_shape).astype(np.float32)

    _test_functions(function, data=data, opset_version=11)


@pytest.mark.parametrize(
    'function,input_shape',
    (
            ('Tanh', [8, 3, 32, 32]),
    ),
)
def test_hyperbolic_functions(function: str, input_shape: List[int]) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    _test_functions(function, data=data, opset_version=11)
