import numpy as np
import onnx
import pytest

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'op_type',
    ('Add', 'Sub', 'Mul', 'Div', 'Pow'),
)
def test_math_binary_operation(op_type) -> None:
    input_shape = [10, 3, 128, 128]
    if op_type == 'Pow':
        x_variants = [
            np.random.uniform(low=0.0, high=4.0, size=input_shape).astype(np.float32),
            np.random.uniform(low=-4.0, high=4.0, size=input_shape).astype(np.float32),
            np.random.uniform(low=-4.0, high=0.001, size=input_shape).astype(np.float32),
            np.random.uniform(low=-4.0, high=4.0, size=input_shape).astype(np.float32),
        ]

        y_variants = [
            np.random.uniform(low=-3.0, high=3.0, size=1).astype(np.float32),
            np.random.randint(low=0, high=4, size=[1] * len(input_shape)).astype(np.float32),
            np.random.randint(low=-4, high=0, size=input_shape).astype(np.float32),
            np.array([0.0], dtype=np.float32),
        ]
    else:
        y_variants = [
            np.random.uniform(low=-1.0, high=1.0, size=1).astype(np.float32),
            np.random.uniform(low=-1.0, high=1.0, size=[1] * len(input_shape)).astype(np.float32),
            np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32),
            np.array([0.0], dtype=np.float32),
        ]
        x_variants = [np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32), ] * len(y_variants)

    for x, y in zip(x_variants, y_variants):
        test_inputs = {'x': x, 'y': y}
        initializers = {}
        node = onnx.helper.make_node(
            op_type=op_type,
            inputs=['x', 'y'],
            outputs=['z'],
        )

        model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
        check_model(model, test_inputs)
