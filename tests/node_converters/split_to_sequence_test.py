from typing import Optional

import numpy as np
import pytest
from onnx.helper import make_node
from onnx.helper import make_tensor_sequence_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_split(
    inputs: np.ndarray,
    split: Optional[np.ndarray],
    axis: Optional[int],
    keepdims: Optional[int],
) -> None:
    test_inputs = {'input': inputs}
    if split is not None:
        test_inputs['split'] = split

    node = make_node(
        op_type='SplitToSequence',
        inputs=[*test_inputs],
        outputs=['output'],
        axis=axis,
        keepdims=keepdims,
    )

    outputs_info = [
        make_tensor_sequence_value_info(
            name='output',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[inputs.dtype],
            shape=None,
        )
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'inputs, split, axis, keepdims',
    [
        (
            np.arange(18).reshape((3, 6)).astype(np.float32),
            np.array(2, dtype=np.int64),
            1,
            None,
        ),
        (
            np.arange(18).reshape((3, 6)).astype(np.float32),
            None,
            1,
            1,
        ),
    ],
)
def test_split(  # pylint: disable=missing-function-docstring
    inputs: np.ndarray,
    split: Optional[np.ndarray],
    axis: Optional[int],
    keepdims: Optional[int],
) -> None:
    _test_split(
        inputs=inputs,
        split=split,
        axis=axis,
        keepdims=keepdims,
    )
