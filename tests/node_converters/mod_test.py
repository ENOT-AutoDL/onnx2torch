import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize('fmod_type', [0, 1])
def test_mod(  # pylint: disable=missing-function-docstring
    fmod_type: int,
) -> None:
    x_variants = np.random.randn(12, 1).astype(np.float32)
    y_variants = np.random.randn(12, 1).astype(np.float32)

    if fmod_type == 0:
        x_variants = np.random.randint(10, 15, size=(12, 1))
        y_variants = np.random.randint(10, 15, size=(12, 1))

    test_inputs = {'x': x_variants, 'y': y_variants}

    node = onnx.helper.make_node(op_type='Mod', inputs=['x', 'y'], outputs=['z'], fmod=fmod_type)
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_onnx_model(model, test_inputs)
