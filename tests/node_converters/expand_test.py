from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_expand(
        data: np.ndarray,
        shape: List[int],
) -> None:
    test_inputs = {
        'x': data,
        'shape': np.array(shape, dtype=np.int64),
    }

    node = onnx.helper.make_node(op_type='Expand', inputs=list(test_inputs), outputs=['y'])
    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[data.dtype],
            shape=[],
        ),
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_model(model, test_inputs)


@pytest.mark.parametrize(
    'src_shape,dst_shape',
    (
        ([3, 1], [2, 1, 6]),
        ([3, 1], [3, 4]),
    ),
)
def test_expand(src_shape: List[int], dst_shape: List[int]) -> None:
    data = np.reshape(np.arange(1, np.prod(src_shape) + 1, dtype=np.float32), src_shape)
    _test_expand(
        data=data,
        shape=dst_shape,
    )
