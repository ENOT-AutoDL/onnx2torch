import numpy as np
import onnx
import pytest

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes

REDUCE_OPERATIONS_WITH_TOLERANCE = (
    ('ReduceMax', 0),
    ('ReduceMin', 0),
    ('ReduceMean', 10 ** -5),
    ('ReduceSum', 10 ** -5),
    ('ReduceProd', 10 ** -5),
)


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
        atol_onnx_torch2onnx=tol,
    )


@pytest.mark.parametrize('op_type,tol', REDUCE_OPERATIONS_WITH_TOLERANCE)
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
def test_reduce(op_type, tol, shape, axes, keepdims) -> None:
    test_kwargs = dict(
        input_tensor=np.random.uniform(-10, 10, shape).astype(np.float32),
        op_type=op_type,
        tol=tol,
    )
    if axes is not None:
        test_kwargs['axes'] = axes
    if keepdims is not None:
        test_kwargs['keepdims'] = keepdims

    _test_reduce(**test_kwargs)
