# pylint: disable=missing-function-docstring
from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_depth_to_space(
    input_shape: List[int],
    blocksize: int,
    mode: str,
    opset: int,
) -> None:
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(  # type: ignore
        op_type='DepthToSpace',
        inputs=['x'],
        outputs=['y'],
        blocksize=blocksize,
        mode=mode,
    )
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs, opset_version=opset)
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'input_shape, blocksize',
    [
        ([1, 12, 3, 3], 2),
        ([5, 75, 3, 3], 5),
        ([7, 588, 3, 4], 7),
    ],
)
@pytest.mark.parametrize('opset', [11, 13])
def test_depth_to_space(input_shape: List[int], blocksize: int, opset: int) -> None:
    _test_depth_to_space(input_shape=input_shape, blocksize=blocksize, mode='CRD', opset=opset)
