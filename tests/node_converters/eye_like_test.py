from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize('dtype', [None, 1, 6, 7, 11])
@pytest.mark.parametrize('k', [-2, -1, 0, 1, 2])
@pytest.mark.parametrize('shape', [[2, 3], [3, 4], [3, 3]])
def test_eye_like(  # pylint: disable=missing-function-docstring
    shape: Tuple[int],
    dtype: Optional[int],
    k: int,  # pylint: disable=invalid-name
) -> None:
    input_values = np.random.randn(*shape).astype(np.float32)
    test_inputs = {'x': input_values}

    node = onnx.helper.make_node(op_type='EyeLike', inputs=['x'], outputs=['z'], dtype=dtype, k=k)
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=[make_tensor_value_info(name='z', elem_type=dtype, shape=shape)] if dtype else None,
    )
    check_onnx_model(model, test_inputs)
