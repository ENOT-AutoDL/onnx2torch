from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'parameters_as_inputs',
    (True, False),
)
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
def test_batch_norm(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    parameters_as_inputs: bool,
) -> None:
    num_features = input_shape[1]
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    scale = np.random.uniform(low=0.0, high=1.0, size=num_features).astype(np.float32)
    bias = np.random.uniform(low=-1.0, high=1.0, size=num_features).astype(np.float32)
    mean = np.random.uniform(low=-1.0, high=1.0, size=num_features).astype(np.float32)
    var = np.random.uniform(low=0.001, high=0.5, size=num_features).astype(np.float32)

    test_inputs = {'x': x}
    initializers = {}
    parameters = {
        'scale': scale,
        'bias': bias,
        'mean': mean,
        'var': var,
    }
    if parameters_as_inputs:
        initializers.update(parameters)
    else:
        test_inputs.update(parameters)

    node = onnx.helper.make_node(
        op_type='BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'var'],
        outputs=['y'],
    )

    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-6,
        atol_torch_cpu_cuda=10**-6,
    )
