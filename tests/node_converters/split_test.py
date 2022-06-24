from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_split(
        x: np.ndarray,
        expected_output: List[np.ndarray],
        opset_version: int,
        **kwargs,
) -> None:
    inputs = ['x', ]
    test_inputs = {'x': x}

    if opset_version >= 13 and kwargs.get('split') is not None:
        split = kwargs.pop('split')
        test_inputs['split'] = split
        inputs.append('split')

    node = onnx.helper.make_node(
        op_type='Split',
        inputs=inputs,
        outputs=[f'output_{i}' for i, _ in enumerate(expected_output)],
        **kwargs,
    )

    outputs_info = [
        make_tensor_value_info(
            name=f'output_{i}',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[out.dtype],
            shape=out.shape,
        )
        for i, out in enumerate(expected_output)
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=opset_version,
    )
    check_onnx_model(model, test_inputs)


INPUT_1D = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)
INPUT_2D = np.array([
    [1., 2., 3., 4., 5., 6.],
    [7., 8., 9., 10., 11., 12.]
]).astype(np.float32)

EMPTY_INPUT = np.array([]).astype(np.float32)
EXPECTED_EMPTY_OUT = [np.array([]).astype(np.float32), np.array([]).astype(np.float32), np.array([]).astype(np.float32)]


@pytest.mark.parametrize(
    'input_array,expected_out,axis,split',
    (
            (INPUT_1D, np.split(INPUT_1D, 3), None, None),
            (INPUT_1D, np.split(INPUT_1D, 3), 0, None),
            (INPUT_1D, np.split(INPUT_1D, [2]), None, np.array([2, 4]).astype(np.int64)),
            (INPUT_1D, np.split(INPUT_1D, [2]), 0, np.array([2, 4]).astype(np.int64)),
            (INPUT_2D, np.split(INPUT_2D, 2, axis=1), 1, None),
            (INPUT_2D, np.split(INPUT_2D, [2], axis=1), 1, np.array([2, 4]).astype(np.int64)),
            (EMPTY_INPUT, EXPECTED_EMPTY_OUT, None, np.array([0, 0, 0]).astype(np.int64))
    ),
)
@pytest.mark.parametrize('opset_version', (13, 11, 2))
def test_split(
    input_array: np.ndarray,
    expected_out: List[np.ndarray],
    axis: Optional[int],
    split: Optional[np.ndarray],
    opset_version: int,
) -> None:
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    if split is not None:
        kwargs['split'] = split

    _test_split(input_array, expected_out, opset_version=opset_version, **kwargs)
