import itertools

import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_transpose(data: np.ndarray, **kwargs) -> None:
    test_inputs = {'input_tensor': data}
    node = onnx.helper.make_node(
        op_type='Transpose',
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)


def test_transpose() -> None:
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    permutations = list(itertools.permutations(np.arange(len(shape))))
    for permutation in permutations:
        _test_transpose(
            data=data,
            perm=np.array(permutation, dtype=np.int64),
        )

    _test_transpose(data=data)
