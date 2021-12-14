from typing import Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_slice(
        input_tensor: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        output_shape: np.ndarray,
        axes: Optional[np.ndarray] = None,
        steps: Optional[np.ndarray] = None,
) -> None:
    test_inputs = {'input_tensor': input_tensor}

    initializers = {'starts': starts, 'ends': ends}
    if axes is not None:
        initializers['axes'] = axes
    if steps is not None:
        initializers['steps'] = steps

    node = onnx.helper.make_node(
        op_type='Slice',
        inputs=list(test_inputs.keys()) + list(initializers.keys()),
        outputs=['y'],
    )
    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[input_tensor.dtype],
            shape=output_shape,
        ),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_model(model, test_inputs)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_slice() -> None:
    x = np.random.randn(20, 10, 5).astype(np.float32)

    _test_slice(
        input_tensor=x,
        starts=np.array([0, 0], dtype=np.int64),
        ends=np.array([3, 10], dtype=np.int64),
        axes=np.array([0, 1], dtype=np.int64),
        steps=np.array([1, 1], dtype=np.int64),
        output_shape=x[0:3, 0:10].shape,
    )

    _test_slice(
        input_tensor=x,
        starts=np.array([0, 0, 3], dtype=np.int64),
        ends=np.array([20, 10, 4], dtype=np.int64),
        output_shape=x[:, :, 3:4].shape,
    )

    _test_slice(
        input_tensor=x,
        starts=np.array([1], dtype=np.int64),
        ends=np.array([1000], dtype=np.int64),
        axes=np.array([1], dtype=np.int64),
        steps=np.array([1], dtype=np.int64),
        output_shape=x[:, 1:1000].shape,
    )

    _test_slice(
        input_tensor=x,
        starts=np.array([0], dtype=np.int64),
        ends=np.array([-1], dtype=np.int64),
        axes=np.array([1], dtype=np.int64),
        steps=np.array([1], dtype=np.int64),
        output_shape=x[:, 0:-1].shape,
    )

    _test_slice(
        input_tensor=x,
        starts=np.array([20, 10, 4], dtype=np.int64),
        ends=np.array([0, 0, 1], dtype=np.int64),
        axes=np.array([0, 1, 2], dtype=np.int64),
        steps=np.array([-1, -3, -2], dtype=np.int64),
        output_shape=x[20:0:-1, 10:0:-3, 4:1:-2].shape,
    )

    _test_slice(
        input_tensor=x,
        starts=np.array([0, 0, 3], dtype=np.int64),
        ends=np.array([20, 10, 4], dtype=np.int64),
        axes=np.array([0, -2, -1], dtype=np.int64),
        output_shape=x[:, :, 3:4].shape,
    )
