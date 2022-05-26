from typing import Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_cumsum(
        input_tensor: np.ndarray,
        axis: int,
        exclusive: int,
        reverse: int,
) -> None:
    test_inputs = {'x': input_tensor, 'axis': np.array(axis)}
    node = onnx.helper.make_node(
        op_type='CumSum',
        inputs=list(test_inputs.keys()),
        outputs=['y'],
        exclusive=exclusive,
        reverse=reverse,
    )

    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[input_tensor.dtype],
            shape=input_tensor.shape,
        ),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(model, test_inputs)


# @pytest.mark.parametrize(
#     'input_tensor,axis',
#     (
#             (np.random.randint(-10, 10, size=(10, )), 0),
#             (np.random.randint(-10, 10, size=(10, 10)), 0),
#             (np.random.randint(-10, 10, size=(10, 10)), 1),
#             (np.random.randint(-10, 10, size=(10, 10)), -1),
#             (np.random.randint(-10, 10, size=(10, 10)), -2),
#             (np.random.randint(-10, 10, size=(10, 10, 5)), 0),
#             (np.random.randint(-10, 10, size=(10, 10, 5)), 1),
#             (np.random.randint(-10, 10, size=(10, 10, 5)), 2),
#             (np.random.randint(-10, 10, size=(10, 10, 5)), -1),
#             (np.random.randint(-10, 10, size=(10, 10, 5)), -2),
#             (np.random.randint(-10, 10, size=(10, 10, 5)), -3),
#      ),
# )
# @pytest.mark.parametrize(
#     'exclusive,reverse',
#     (
#             # (0, 0),
#             # (0, 1),
#             (1, 0),
#             (1, 1),
#     ),
# )
# def test_cumsum(input_tensor, axis, exclusive, reverse) -> None:
#     _test_cumsum(
#         input_tensor=input_tensor,
#         axis=axis,
#         exclusive=exclusive,
#         reverse=reverse,
#     )


if __name__ == '__main__':
    _test_cumsum(
        input_tensor=np.random.randint(-10, 10, size=(12, 12)),
        axis=0,
        exclusive=1,
        reverse=0,
    )