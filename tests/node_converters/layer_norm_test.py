# pylint: disable=missing-function-docstring
from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_layer_norm(
    x: np.ndarray,
    scale: np.ndarray,
    bias: Optional[np.ndarray],
    axis: int,
    parameters_as_inputs: bool,
) -> None:
    inputs = {'input': x}
    parameters = {'scale': scale}
    if bias is not None:
        parameters['bias'] = bias

    initializers = {}

    if parameters_as_inputs:
        inputs.update(parameters)
    else:
        initializers.update(parameters)

    node = onnx.helper.make_node(
        op_type='LayerNormalization',
        inputs=['input', 'scale', 'bias'] if bias is not None else ['input', 'scale'],
        outputs=['y'],
        axis=axis,
    )
    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=inputs, opset_version=17)
    check_onnx_model(
        onnx_model=model,
        onnx_inputs=inputs,
        atol_onnx_torch=1e-5,
        atol_torch_cpu_cuda=1e-5,
        atol_onnx_torch2onnx=1e-5,
    )


@pytest.mark.parametrize('parameters_as_inputs', (True, False))
@pytest.mark.parametrize(
    'input_shape',
    (
        [2, 3, 16],
        [3, 1, 224],
        [4, 3, 16, 16],
        [5, 1, 32, 32],
        [6, 3, 16, 16, 8],
        [7, 1, 7, 7, 16],
    ),
)
def test_layer_norm(input_shape: List[int], parameters_as_inputs: bool) -> None:
    x = np.random.randn(*input_shape).astype(np.float32)

    for axis in [*range(len(input_shape))] + [-1]:
        normalized_shape = input_shape[axis:]

        scale = np.random.randn(*normalized_shape).astype(np.float32)
        bias = np.random.randn(*normalized_shape).astype(np.float32)

        for bias_ in [bias, None]:
            _test_layer_norm(
                x=x,
                scale=scale,
                bias=bias_,
                axis=axis,
                parameters_as_inputs=parameters_as_inputs,
            )
