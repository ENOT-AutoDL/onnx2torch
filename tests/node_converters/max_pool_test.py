from typing import List

import numpy as np
import onnx

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_max_pool(
        input_shape: List[int],
        **kwargs,
) -> None:

    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs)
    check_model(model, test_inputs)


def test_max_pool() -> None:
    _test_max_pool(input_shape=[2, 3, 16, 16], kernel_shape=[2, 2], strides=[2, 2])
    _test_max_pool(input_shape=[2, 3, 16, 16, 16], kernel_shape=[2, 2, 2], strides=[2, 2, 2])
    _test_max_pool(input_shape=[2, 3, 16, 16], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
