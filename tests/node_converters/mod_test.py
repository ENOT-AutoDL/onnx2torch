from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes

def test_mod() -> None:  # pylint: disable=missing-function-docstring
    x_variants = [
        np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)
    ]
    x_variants = np.random.randn(12, 1).astype(np.float32)
    y_variants = np.random.randn(12, 1).astype(np.float32)

    # y_variants = [
        # np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
    # ]

    test_inputs = {'x': x_variants, 'y': y_variants}
    initializers = {}
    node = onnx.helper.make_node(
        op_type='Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=1
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)

