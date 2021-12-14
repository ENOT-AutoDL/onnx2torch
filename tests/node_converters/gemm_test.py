from typing import Tuple

import numpy as np
import onnx

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_gemm(
        input_shape: Tuple[int, int],
        weights_shape: Tuple[int, int],
        use_bias: bool,
        **kwargs,
) -> None:

    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    weights = np.random.uniform(low=-1.0, high=1.0, size=weights_shape).astype(np.float32)
    bias_shape = weights_shape[1] if kwargs.get("transB", 0) == 0 else weights_shape[0]
    bias = np.random.uniform(low=-1.0, high=1.0, size=(bias_shape, )).astype(np.float32) if use_bias else None

    test_inputs = {'x': x}
    initializers = {'weights': weights}
    gemm_inputs = ['x', 'weights']
    if bias is not None:
        initializers['bias'] = bias
        gemm_inputs.append('bias')

    node = onnx.helper.make_node(
        op_type='Gemm',
        inputs=gemm_inputs,
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-5,
        atol_torch_cpu_cuda=10**-5,
        atol_onnx_torch2onnx=10**-5,
    )


def test_base_gemm() -> None:
    _test_gemm(input_shape=(2, 16), weights_shape=(16, 32), use_bias=False)
    _test_gemm(input_shape=(2, 16), weights_shape=(16, 32), use_bias=True)
    _test_gemm(input_shape=(2, 16), weights_shape=(32, 16), use_bias=True, transB=1)


def test_advanced_gemm() -> None:
    _test_gemm(
        input_shape=(16, 2),
        weights_shape=(32, 16),
        use_bias=True,
        transA=1,
        transB=1,
        alpha=0.25,
        beta=0.5,
    )
