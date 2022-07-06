from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_comparison(op_type: str, x: np.ndarray, y: np.ndarray, opset_version: int = 13) -> None:
    test_inputs = {'x': x, 'y': y}

    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=list(test_inputs),
        outputs=['out'],
    )
    outputs_info = [
        make_tensor_value_info(
            name='out',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype('bool')],
            shape=x.shape,
        ),
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=opset_version,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'op_type,x_shape,y_shape',
    (
        ('Equal', [3, 4, 5], [5]),
        ('Equal', [3, 4, 5], [3, 4, 5]),
        ('Less', [3, 4, 5], [5]),
        ('Less', [3, 4, 5], [3, 4, 5]),
        ('Greater', [3, 4, 5], [5]),
        ('Greater', [3, 4, 5], [3, 4, 5]),
        ('LessOrEqual', [3, 4, 5], [5]),
        ('LessOrEqual', [3, 4, 5], [3, 4, 5]),
        ('GreaterOrEqual', [3, 4, 5], [5]),
        ('GreaterOrEqual', [3, 4, 5], [3, 4, 5]),
    ),
)
def test_comparison(  # pylint: disable=missing-function-docstring
    op_type: str,
    x_shape: List[int],
    y_shape: List[int],
) -> None:
    _test_comparison(
        op_type=op_type,
        x=np.random.randn(*x_shape),
        y=np.random.randn(*y_shape),
    )
