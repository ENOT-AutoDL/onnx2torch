import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_batch_norm(
        input_shape,
        num_features,
        **kwargs,
) -> None:

    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    scale = np.random.uniform(low=0.5, high=1.0, size=num_features).astype(np.float32)
    bias = np.random.uniform(low=0.5, high=1.0, size=num_features).astype(np.float32)
    mean = np.random.uniform(low=0.5, high=1.0, size=num_features).astype(np.float32)
    var = np.random.uniform(low=0.5, high=1.0, size=num_features).astype(np.float32)

    test_inputs = {'x': x}
    initializers = {
        'scale': scale,
        'bias': bias,
        'mean': mean,
        'var': var,
    }
    node = onnx.helper.make_node(
        op_type='BatchNormalization',
        inputs=['x', 'scale', 'bias', 'mean', 'var'],
        outputs=['y'],
        **kwargs,
    )

    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-7,
        atol_torch_cpu_cuda=10**-7,
        atol_onnx_torch2onnx=10**-7,
    )


def test_batch_norm():
    _test_batch_norm([2, 3, 4, 5], 3)
    _test_batch_norm([2, 3, 4, 5, 6], 3)
    _test_batch_norm([2, 3], 3)
