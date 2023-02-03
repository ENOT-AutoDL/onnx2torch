from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize('parameters_as_inputs', (True, False))
@pytest.mark.parametrize(
    'input_shape',
    (
        # 1d
        [2, 3, 16],
        [2, 1, 7],
        # 2d
        [2, 3, 16, 16],
        [2, 1, 7, 16],
        # 3d
        [2, 3, 16, 16, 16],
        [2, 1, 16, 7, 16],
    ),
)
def test_instance_norm(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    parameters_as_inputs: bool,
) -> None:
    num_features = input_shape[1]
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    scale = np.random.uniform(low=0.0, high=1.0, size=num_features).astype(np.float32)
    bias = np.random.uniform(low=-1.0, high=1.0, size=num_features).astype(np.float32)

    inputs = {'input': x}
    parameters = {'scale': scale, 'bias': bias}
    initializers = {}

    if parameters_as_inputs:
        inputs.update(parameters)
    else:
        initializers.update(parameters)

    node = onnx.helper.make_node(op_type='InstanceNormalization', inputs=['input', 'scale', 'bias'], outputs=['y'])

    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=inputs)
    check_onnx_model(onnx_model=model, onnx_inputs=inputs, atol_onnx_torch=1e-6, atol_torch_cpu_cuda=1e-6)
