import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def test_matmul() -> None:
    a_variants = [
        np.random.randn(3, 4).astype(np.float32),
        np.random.randn(2, 3, 4).astype(np.float32),
        np.random.randn(1, 2, 3, 4).astype(np.float32),
    ]

    b_variants = [
        np.random.randn(4, 3).astype(np.float32),
        np.random.randn(2, 4, 3).astype(np.float32),
        np.random.randn(1, 2, 4, 3).astype(np.float32),
    ]

    for a, b in zip(a_variants, b_variants):
        test_inputs = {'a': a, 'b': b}
        initializers = {}
        node = onnx.helper.make_node(
            op_type='MatMul',
            inputs=['a', 'b'],
            outputs=['z'],
        )

        model = make_model_from_nodes(
            nodes=node,
            initializers=initializers,
            inputs_example=test_inputs,
        )
        check_onnx_model(model, test_inputs)
