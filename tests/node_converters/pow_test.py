import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def test_pow() -> None:
    input_shape = [10, 3, 128, 128]
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

    for x, y in zip(x_variants, y_variants):
        test_inputs = {'x': x, 'y': y}
        initializers = {}
        node = onnx.helper.make_node(
            op_type='Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        model = make_model_from_nodes(
            nodes=node, 
            initializers=initializers,
            inputs_example=test_inputs,
        )
        check_onnx_model(model, test_inputs)


def test_sqrt() -> None:
    input_shape = [10, 3, 128, 128]
    x = np.random.uniform(low=0.0, high=10.0, size=input_shape).astype(np.float32)

    test_inputs = {'x': x}
    initializers = {}
    node = onnx.helper.make_node(
        op_type='Sqrt',
        inputs=['x'],
        outputs=['z'],
    )

    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    check_onnx_model(model, test_inputs)
