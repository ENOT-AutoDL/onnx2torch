from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_squeeze(
        input_tensor: np.ndarray,
        axes: np.ndarray,
        opset_version: int,
        **kwargs,

) -> None:
    test_inputs = {'input_tensor': input_tensor}
    initializers = {}

    if opset_version >= 13:
        initializers['axes'] = np.asarray(axes).astype(np.int64)
    else:
        kwargs['axes'] = axes

    node = onnx.helper.make_node(
        op_type='Squeeze',
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


@pytest.mark.parametrize(
    'shape,axes,opset_version',
    (
            ([1, 3, 4, 5], [0], 11),
            ([1, 3, 1, 5], [-2], 11),
            ([1, 3, 4, 5], [0], 13),
            ([1, 3, 1, 5], [-2], 13),
    ),
)
def test_squeeze(shape: List[int], axes: List[int], opset_version: int) -> None:
    x = np.random.randn(*shape).astype(np.float32)
    axes = np.array(axes, dtype=np.int64)
    _test_squeeze(input_tensor=x, axes=axes, opset_version=opset_version)
