from typing import Dict
from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_pool_op(
    op_type,
    input_shape: List[int],
    atol_onnx_torch: float = 0.0,
    **kwargs,
) -> None:
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(
        op_type,
        inputs=['x'],
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs)
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=atol_onnx_torch,
    )


@pytest.mark.parametrize(
    'op',
    (
        'MaxPool',
        'AveragePool',
    ),
)
@pytest.mark.parametrize(
    'input_shape,kernel_shape,optional_attrs',
    (
        # 1d
        ([2, 3, 16], [2], {}),
        ([2, 3, 16], [1], {}),
        ([2, 3, 16], [3], {}),
        ([2, 3, 16], [2], {'strides': [3]}),
        ([2, 3, 16], [2], {'ceil_mode': 1}),
        # 2d
        ([2, 3, 16, 16], [2, 2], {}),
        ([2, 3, 16, 16], [1, 2], {}),
        ([2, 3, 16, 16], [3, 2], {}),
        ([2, 3, 16, 16], [2, 2], {'strides': [2, 3]}),
        ([2, 3, 16, 16], [2, 2], {'ceil_mode': 1}),
        # 3d
        ([2, 3, 16, 16, 16], [2, 2, 2], {}),
        ([2, 3, 16, 16, 16], [1, 2, 3], {}),
        ([2, 3, 16, 16, 16], [3, 2, 1], {}),
        ([2, 3, 16, 16, 16], [2, 2, 2], {'strides': [1, 2, 3]}),
        ([2, 3, 16, 16, 16], [2, 2, 2], {'ceil_mode': 1}),
    ),
)
def test_max_pool_average_pool(  # pylint: disable=missing-function-docstring
    op: str,  # pylint: disable=invalid-name
    input_shape: List[int],
    kernel_shape: List[int],
    optional_attrs: Dict,
) -> None:
    if op == 'AveragePool':
        optional_attrs['atol_onnx_torch'] = 10**-7

    _test_pool_op(op, input_shape=input_shape, kernel_shape=kernel_shape, **optional_attrs)


@pytest.mark.parametrize(
    'input_shape,kernel_shape,optional_attrs',
    (
        # 1d
        ([2, 3, 16], [2], {'pads': [1] * 2}),
        ([2, 3, 16], [3], {'pads': [0, 1]}),
        ([2, 3, 16], [3], {'pads': [2, 0]}),
        # 2d
        ([2, 3, 16, 16], [2, 2], {'pads': [1] * 4}),
        ([2, 3, 16, 16], [2, 2], {'pads': [0] * 2 + [1] * 2}),
        ([2, 3, 16, 16], [3, 3], {'pads': [0, 1, 1, 0]}),
        # 3d
        ([2, 3, 16, 16, 16], [2, 2, 2], {'pads': [1] * 6}),
        ([2, 3, 16, 16, 16], [2, 2, 2], {'pads': [0] * 3 + [1] * 3}),
        ([2, 3, 16, 16, 16], [3, 3, 3], {'pads': [0, 1, 2, 2, 1, 0]}),
    ),
)
def test_max_pool_padding(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    kernel_shape: List[int],
    optional_attrs: Dict,
) -> None:
    _test_pool_op('MaxPool', input_shape=input_shape, kernel_shape=kernel_shape, **optional_attrs)
