import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'op_type',
    ('Or', 'And', 'Xor'),
)
def test_logical_operation(op_type: str) -> None:
    x = (np.random.randn(10, 1, 64, 128) > 0)
    y_variants = (
        (np.random.randn(128) > 0),
        (np.random.randn(64, 128) > 0),
        (np.random.randn(1, 64, 128) > 0),
        (np.random.randn(1, 3, 1, 128) > 0),
        (np.random.randn(10, 1, 64, 128) > 0),
    )
    for y in y_variants:
        test_inputs = {'x': x, 'y': y}
        initializers = {}
        node = onnx.helper.make_node(
            op_type=op_type,
            inputs=['x', 'y'],
            outputs=['z'],
        )

        model = make_model_from_nodes(
            nodes=node,
            initializers=initializers,
            inputs_example=test_inputs,
        )
        check_onnx_model(model, test_inputs)


def test_not() -> None:
    x_variants = (
        (np.random.randn(128) > 0),
        (np.random.randn(64, 128) > 0),
        (np.random.randn(1, 64, 128) > 0),
        (np.random.randn(10, 1, 64, 128) > 0),
    )

    for x in x_variants:
        test_inputs = {'x': x}
        initializers = {}
        node = onnx.helper.make_node(
            op_type='Not',
            inputs=['x'],
            outputs=['z'],
        )

        model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
        check_onnx_model(model, test_inputs)
