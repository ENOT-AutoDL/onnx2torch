import numpy as np
import onnx
import pytest

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes

REDUCE_OPERATIONS = ('ReduceMax', 'ReduceMin', 'ReduceMean', 'ReduceSum', 'ReduceProd')


def _test_reduce(input_tensor: np.ndarray, op_type: str, **kwargs) -> None:
    tol = 0.0
    if op_type in REDUCE_OPERATIONS[2:]:
        tol = 10 ** -5

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
        atol_onnx_torch2onnx=tol,
    )


@pytest.mark.parametrize('op_type', REDUCE_OPERATIONS)
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
def test_reduce(op_type, shape, axes, keepdims) -> None:
    test_kwargs = dict(
        input_tensor=np.random.uniform(-10, 10, shape).astype(np.float32),
        op_type=op_type,
    )
    if axes is not None:
        test_kwargs['axes'] = axes
    if keepdims is not None:
        test_kwargs['keepdims'] = keepdims

    _test_reduce(**test_kwargs)
