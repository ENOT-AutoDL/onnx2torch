from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_reshape(
        input_shape: List[int],
        output_shape: List[int],
        opset_version: int,
        **kwargs,
) -> None:
    test_inputs = {'x': np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)}
    initializers = {'output_shape': np.asarray(output_shape, dtype=np.int64)}

    node = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['x', 'output_shape'],
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


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'input_shape,output_shape,opset_version',
    (
            ([2, 3, 16, 16], [2, -1, 3], 9),
            ([2, 3, 16, 16], [2, 0, -1], 9),
            ([2, 3, 16, 16], [2, 0, 1, 1, 1, 1, 1, 1, -1], 9),
            ([2, 3, 16, 16], [-1, 1, 1, 2, 1, 1, 1, 2, 1, 1], 14),
    ),
)
def test_reshape(input_shape: List[int], output_shape: List[int], opset_version: int) -> None:
    _test_reshape(
        input_shape=input_shape,
        output_shape=output_shape,
        opset_version=opset_version,
    )
