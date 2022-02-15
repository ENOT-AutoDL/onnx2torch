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
        atol_torch_cpu_cuda=0,
        atol_onnx_torch2onnx=0,
    )


@pytest.mark.parametrize(
    'op',
    (
        'MaxPool',
        'AveragePool',
    )
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
def test_max_pool_average_pool(op: str, input_shape: List[int], kernel_shape: List[int], optional_attrs: Dict) -> None:
    if op == 'AveragePool':
        optional_attrs['atol_onnx_torch'] = 10**-7

    _test_pool_op(op, input_shape=input_shape, kernel_shape=kernel_shape, **optional_attrs)
