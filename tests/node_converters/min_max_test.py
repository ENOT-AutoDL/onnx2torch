from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_min_max(
    data_list: List[np.ndarray],
    operation_type: str,
) -> None:
    test_inputs = {f'data_{i}': data for i, data in enumerate(data_list)}

    node = onnx.helper.make_node(op_type=operation_type, inputs=list(test_inputs), outputs=['y'])
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
@pytest.mark.parametrize('operation_type', ['Min', 'Max'])
def test_min_amx(  # pylint: disable=missing-function-docstring
    input_shapes: List[List[int]],
    operation_type: str,
) -> None:
    input_tensors = [np.random.normal(size=i_shape).astype(np.float32) for i_shape in input_shapes]

    _test_min_max(
        data_list=input_tensors,
        operation_type=operation_type,
    )
