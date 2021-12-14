from typing import List

import numpy as np
import onnx

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_flatten(
        input_shape: List[int],
        **kwargs,
) -> None:

    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(
        op_type='Flatten',
        inputs=['x'],
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs)
    check_model(model, test_inputs)


def test_flatten() -> None:
    _test_flatten(input_shape=[2, 3, 16, 16, 16])
    _test_flatten(input_shape=[2, 3, 16, 16], axis=2)
    _test_flatten(input_shape=[2, 3, 16], axis=-1)
