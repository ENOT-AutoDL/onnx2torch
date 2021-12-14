from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_unsqueeze(
        input_shape: List[int],
        axes: List[int],
        opset_version: int,
        **kwargs,
) -> None:
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}
    initializers = {}

    if opset_version >= 13:
        initializers['axes'] = np.array(axes, dtype=np.int64)
    else:
        kwargs['axes'] = axes

    node = onnx.helper.make_node(
        op_type='Unsqueeze',
        inputs=list(test_inputs) + list(initializers),
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_model(model, test_inputs)


# Known warning. Shape Inference do not work properly in opset_version=9 and negative indices.
# [W:onnxruntime:, execution_frame.cc:721 VerifyOutputSizes]
# Expected shape from model of {2,3,16,16} does not match actual shape of {2,1,3,16,1,16} for output y
@pytest.mark.parametrize(
    'input_shape,axes,opset_version',
    (
            ([2, 3, 16, 16], [0, 1], 9),
            ([2, 3, 16, 16], [1, 5], 9),
            ([2, 3, 16, 16], [1, -2], 9),
            ([2, 3, 16, 16], [0, 1], 13),
            ([2, 3, 16, 16], [1, 5], 13),
            ([2, 3, 16, 16], [1, -3], 13),
    ),
)
def test_unsqueeze(input_shape: List[int], axes: List[int], opset_version: int) -> None:
    _test_unsqueeze(input_shape=input_shape, axes=axes, opset_version=opset_version)
