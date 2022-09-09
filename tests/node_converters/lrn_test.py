from random import randrange

import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_lrn(data: np.ndarray, alpha: float, beta: float, bias: float, size: int) -> None:
    test_inputs = {'input_tensor': data}
    node = onnx.helper.make_node(
        op_type='LRN',
        inputs=list(test_inputs),
        outputs=['y'],
        alpha=alpha,  # ONNX attributes are passed as regular keyword arguments.
        beta=beta,
        bias=bias,
        size=size,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)


def test_lrn() -> None:  # pylint: disable=missing-function-docstring
    shape = (1, 3, 227, 227)
    data = np.random.random_sample(shape).astype(np.float32)
    alpha = np.random.uniform(low=0.0, high=1.0)
    beta = np.random.uniform(low=0.0, high=1.0)
    bias = np.random.uniform(low=1.0, high=5.0)
    size = randrange(start=1, stop=10, step=2)  # diameter of channels, not radius, must be odd
    _test_lrn(data=data, alpha=alpha, beta=beta, bias=bias, size=size)
