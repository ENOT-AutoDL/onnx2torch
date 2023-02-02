from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_gemm(
    input_a_shape: Tuple[int, int],
    input_b_shape: Tuple[int, int],
    has_input_c: bool,
    abc_as_initializers: Tuple[bool, bool, bool],
    **kwargs,
) -> None:
    input_a = np.random.uniform(low=-1.0, high=1.0, size=input_a_shape).astype(np.float32)
    input_b = np.random.uniform(low=-1.0, high=1.0, size=input_b_shape).astype(np.float32)
    input_c_shape = input_b_shape[1] if kwargs.get('transB', 0) == 0 else input_b_shape[0]
    input_c = np.random.uniform(low=-1.0, high=1.0, size=(input_c_shape,)).astype(np.float32) if has_input_c else None

    output_shape = [None] * 2
    output_shape[0] = input_a_shape[0 if kwargs.get('transA', 0) == 0 else 1]
    output_shape[1] = input_b_shape[1 if kwargs.get('transB', 0) == 0 else 0]

    test_inputs = {}
    initializers = {}
    gemm_inputs = ['a', 'b']

    if abc_as_initializers[0]:
        initializers['a'] = input_a
    else:
        test_inputs['a'] = input_a

    if abc_as_initializers[1]:
        initializers['b'] = input_b
    else:
        test_inputs['b'] = input_b

    if has_input_c:
        gemm_inputs.append('c')
        if abc_as_initializers[2]:
            initializers['c'] = input_c
        else:
            test_inputs['c'] = input_c

    outputs_info = [
        make_tensor_value_info(
            name='output',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(np.float32)],
            shape=output_shape,
        ),
    ]
    node = onnx.helper.make_node(
        op_type='Gemm',
        inputs=gemm_inputs,
        outputs=['output'],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-5,
        atol_torch_cpu_cuda=10**-5,
    )


@pytest.mark.parametrize(
    'abc_as_initializers',
    (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ),
)
@pytest.mark.parametrize(
    'has_input_c',
    (False, True),
)
@pytest.mark.parametrize(
    'input_a_shape,input_b_shape,trans_a,trans_b,alpha,beta',
    (
        ([3, 4], [4, 3], False, False, None, None),
        ([3, 4], [4, 3], False, False, None, None),
        ([4, 3], [4, 3], True, False, None, None),
        ([3, 4], [3, 4], False, True, None, None),
        ([3, 4], [4, 3], True, True, None, None),
        ([3, 4], [4, 3], False, False, 3.1415926, 2.71828),
    ),
)
def test_gemm(  # pylint: disable=missing-function-docstring
    input_a_shape: Tuple[int, int],
    input_b_shape: Tuple[int, int],
    has_input_c: bool,
    abc_as_initializers: Tuple[bool, bool, bool],
    trans_a: Optional[bool],
    trans_b: Optional[bool],
    alpha: Optional[float],
    beta: Optional[float],
) -> None:
    kwargs = {
        'transA': trans_a,
        'transB': trans_b,
        'alpha': alpha,
        'beta': beta,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    _test_gemm(
        input_a_shape=input_a_shape,
        input_b_shape=input_b_shape,
        has_input_c=has_input_c,
        abc_as_initializers=abc_as_initializers,
        **kwargs,
    )
