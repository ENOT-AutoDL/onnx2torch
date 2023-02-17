from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'dividend',
    [
        [-4, 7, 5, 4, -7, 8],
        [-4.3, 7.2, 5.0, 4.3, -7.2, 8.0],
    ],
)
@pytest.mark.parametrize(
    'divisor',
    [
        [2, -3, 8, -2, 3, 5],
        [2.1, -3.4, 8.0, -2.1, 3.4, 5.0],
    ],
)
@pytest.mark.parametrize('fmod', [0, 1])
def test_mod(  # pylint: disable=missing-function-docstring
    dividend: List[float],
    divisor: List[float],
    fmod: int,
) -> None:
    x_variants = np.array(dividend).astype(np.float32 if fmod else np.int32)
    y_variants = np.array(divisor).astype(np.float32 if fmod else np.int32)

    test_inputs = {'x': x_variants, 'y': y_variants}

    node = onnx.helper.make_node(op_type='Mod', inputs=['x', 'y'], outputs=['z'], fmod=fmod)
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)
