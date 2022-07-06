from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'input_shape',
    (
        [2, 3, 16, 16, 16],
        [2, 3, 16, 16],
        [2, 3, 16],
    ),
)
def test_global_avg_pool(input_shape: List[int]) -> None:  # pylint: disable=missing-function-docstring
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(
        op_type='GlobalAveragePool',
        inputs=['x'],
        outputs=['y'],
    )
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs)
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-7,
        atol_torch_cpu_cuda=10**-7,
    )
