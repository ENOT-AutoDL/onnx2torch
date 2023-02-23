from typing import Tuple

import numpy as np
import onnx
import pytest
from tests.utils.common import check_onnx_model, make_model_from_nodes


@pytest.mark.parametrize('dtype', ((None),))
@pytest.mark.parametrize('eyelike_k', [0, 1, 2])
@pytest.mark.parametrize(
    'input_shapes',
    (
        ((2, 3)),
        ((3, 4)),
        ((3, 3)),
    ),
)
def test_eye(  # pylint: disable=missing-function-docstring
    input_shapes: Tuple[int],
    dtype: str,
    eyelike_k: int,
) -> None:
    input_values = np.random.randint(0, 100, size=input_shapes)

    test_inputs = {'x': input_values}

    node = onnx.helper.make_node(op_type='EyeLike', inputs=['x'], outputs=['z'], k=eyelike_k, dtype=dtype)
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)
