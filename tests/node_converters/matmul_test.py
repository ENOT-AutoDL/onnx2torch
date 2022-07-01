import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def test_matmul() -> None:  # pylint: disable=missing-function-docstring
    x_variants = [
        np.random.randn(3, 4).astype(np.float32),
        np.random.randn(2, 3, 4).astype(np.float32),
        np.random.randn(1, 2, 3, 4).astype(np.float32),
    ]

    y_variants = [
        np.random.randn(4, 3).astype(np.float32),
        np.random.randn(2, 4, 3).astype(np.float32),
        np.random.randn(1, 2, 4, 3).astype(np.float32),
    ]

    for x, y in zip(x_variants, y_variants):
        test_inputs = {'x': x, 'y': y}
        initializers = {}
        node = onnx.helper.make_node(
            op_type='MatMul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        model = make_model_from_nodes(
            nodes=node,
            initializers=initializers,
            inputs_example=test_inputs,
        )
        check_onnx_model(
            model,
            test_inputs,
            atol_onnx_torch=10**-6,
            atol_torch_cpu_cuda=10**-6,
        )
