from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_layer_norm(
    x: np.ndarray,
    parameters_as_inputs: bool,
) -> None:
    normalized_shape = calculate_normalized_shape(x.shape, -1)
    scale = np.random.randn(*normalized_shape).astype(np.float32)
    bias = np.random.randn(*normalized_shape).astype(np.float32)

    inputs = {'input': x}
    parameters = {'scale': scale, 'bias': bias}
    initializers = {}

    if parameters_as_inputs:
        inputs.update(parameters)
    else:
        initializers.update(parameters)

    node = onnx.helper.make_node(
        op_type='LayerNormalization',
        inputs=['input', 'scale', 'bias'],
        outputs=['y'],
    )
    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=inputs, opset_version=17)
    check_onnx_model(onnx_model=model, onnx_inputs=inputs, atol_onnx_torch=1e-6, atol_torch_cpu_cuda=1e-6)


# @pytest.mark.parametrize(
# 'parameters_as_inputs',
# (True, False),
# )
@pytest.mark.parametrize(
    'input_shape',
    (
        # 1d
        [2, 3, 16],
        [2, 1, 7],
        # # 2d
        [2, 3, 16, 16],
        [2, 1, 7, 16],
        # # 3d
        [2, 3, 16, 16, 16],
        [2, 1, 16, 7, 16],
    ),
)
def test_layer_norm(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    parameters_as_inputs: bool = False,
) -> None:
    x = np.random.randn(*input_shape).astype(np.float32)

    _test_layer_norm(x=x, parameters_as_inputs=parameters_as_inputs)


def calculate_normalized_shape(x_shape, axis):  # pylint: disable=missing-function-docstring
    x_rank = len(x_shape)
    if axis < 0:
        axis = axis + x_rank
    return x_shape[axis:]
