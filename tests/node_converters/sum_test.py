from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_sum(
    data_list: List[np.ndarray],
) -> None:
    test_inputs = {f'data_{i}': data for i, data in enumerate(data_list)}

    node = onnx.helper.make_node(op_type='Sum', inputs=list(test_inputs), outputs=['y'])
    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[data_list[0].dtype],
            shape=None,
        ),
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'input_shapes',
    (
        ([],),
        ([2, 3, 4],),
        ([3, 1], [2, 1, 6]),
        ([3, 1], [3, 4]),
    ),
)
def test_sum(input_shapes: List[List[int]]) -> None:  # pylint: disable=missing-function-docstring
    input_tensors = [np.random.normal(size=i_shape).astype(np.float32) for i_shape in input_shapes]
    _test_sum(data_list=input_tensors)
