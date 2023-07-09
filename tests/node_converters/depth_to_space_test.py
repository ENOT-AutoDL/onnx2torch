from typing import List

import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_depth_to_space(
    input_shape: List[int],
    **kwargs,
) -> None:
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(
        op_type='DepthToSpace',
        inputs=['x'],
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs, opset_version=13)
    check_onnx_model(model, test_inputs)


def test_depth_to_space() -> None:  # pylint: disable=missing-function-docstring
    _test_depth_to_space(input_shape=[1, 12, 3, 3], blocksize=2, mode='CRD')