from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_reduce(input_tensor: np.ndarray, op_type: str, tol: float, **kwargs) -> None:
    test_inputs = {'input_tensor': input_tensor}
    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
    )
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=tol,
        atol_torch_cpu_cuda=tol,
        atol_onnx_torch2onnx=0.0,
    )


def _test_reduce_sum(
        input_tensor: np.ndarray,
        axes: Optional[List[int]],
        keepdims: Optional[int] = 1,
        noop_with_empty_axes: Optional[int] = 0,
) -> None:
    test_inputs = {'input_tensor': input_tensor}
    kwargs = {}

    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    else:
        keepdims = 1

    if noop_with_empty_axes is not None:
        kwargs['noop_with_empty_axes'] = noop_with_empty_axes == 1
    else:
        noop_with_empty_axes = 0

    if axes is not None and len(axes) > 0:
        test_inputs['axes'] = np.array(axes, dtype=np.int64)
        output_shape = np.sum(input_tensor, axis=tuple(axes), keepdims=(keepdims == 1)).shape
    else:
        test_inputs['axes'] = np.array([], dtype=np.int64)
        if noop_with_empty_axes == 0:
            output_shape = np.sum(input_tensor, keepdims=(keepdims == 1)).shape
        else:
            output_shape = input_tensor.shape

    node = onnx.helper.make_node(
        op_type='ReduceSum',
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=13,
        outputs_info=(make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[input_tensor.dtype],
            shape=output_shape,
        ),),
    )
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -5,
        atol_torch_cpu_cuda=10 ** -5,
        atol_onnx_torch2onnx=0.0,
    )


@pytest.mark.parametrize(
    'op_type,tol',
    (
        ('ReduceL1', 10 ** -5),
        ('ReduceL2', 10 ** -5),
        ('ReduceLogSum', 10 ** -5),
        ('ReduceLogSumExp', 10 ** -5),
        ('ReduceMax', 0),
        ('ReduceMin', 0),
        ('ReduceMean', 10 ** -5),
        ('ReduceSum', 10 ** -5),
        ('ReduceProd', 10 ** -5),
        ('ReduceSumSquare', 10 ** -5),
    )
)
@pytest.mark.parametrize(
    'shape,axes,keepdims',
    (
            ((1, 3, 8, 8), None, None),
            ((1, 3, 8, 8), None, 0),
            ((1, 3, 8, 8), None, 1),
            ((1, 3, 8, 8), [1], 0),
            ((1, 3, 8, 8), [1], 1),
            ((1, 3, 8, 8), [1], None),
            ((1, 3, 8, 8), [-2], 1),
            ((2, 3, 8, 8), [-2, -4], 1),
            ((2, 3, 8, 8), [1, 3], 1),
    ),
)
def test_reduce(
        op_type: str,
        tol: float,
        shape: Tuple[int],
        axes: Optional[int],
        keepdims: Optional[int],
) -> None:
    if op_type == 'ReduceLogSum':
        left_boundary = 10 ** -5
    else:
        left_boundary = -10

    test_kwargs = dict(
        input_tensor=np.random.uniform(left_boundary, 10, shape).astype(np.float32),
        op_type=op_type,
        tol=tol,
    )
    if axes is not None:
        test_kwargs['axes'] = axes
    if keepdims is not None:
        test_kwargs['keepdims'] = keepdims

    _test_reduce(**test_kwargs)


@pytest.mark.parametrize(
    'shape,axes,keepdims,noop_with_empty_axes',
    (
            ((1, 3, 8, 8), None, None, None),
            ((1, 3, 8, 8), None, 0, 0),
            ((1, 3, 8, 8), None, 1, 0),
            ((1, 3, 8, 8), None, 1, 1),
            ((1, 3, 8, 8), None, 1, 0),
            ((1, 3, 8, 8), [1], 0, 0),
            ((1, 3, 8, 8), [1], 1, 0),
            ((1, 3, 8, 8), [1], None, 0),
            ((1, 3, 8, 8), [-2], 1, 0),
            ((2, 3, 8, 8), [-2, -4], 1, 0),
            ((2, 3, 8, 8), [1, 3], 1, 0),
    ),
)
def test_reduce_sum(
        shape: Tuple[int],
        axes: Optional[List[int]],
        keepdims: Optional[int],
        noop_with_empty_axes: Optional[int],
) -> None:
    _test_reduce_sum(
        input_tensor=np.random.uniform(-10, 10, shape).astype(np.float32),
        axes=axes,
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )
