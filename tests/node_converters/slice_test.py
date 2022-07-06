from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_slice(
    input_tensor: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
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
            shape=None,
        ),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    # onnx checker in torch 1.12 has problems with negative steps in Slice, so we disable it
    ignore_export_checker = steps is not None and np.any(steps < 0)
    check_onnx_model(model, test_inputs, ignore_export_checker=ignore_export_checker)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'input_shape,starts,ends,axes,steps',
    (
        ((20, 10, 15), [0, 0], [3, 10], [0, 1], [1, 1]),
        ((20, 10, 15), [0, 0, 3], [20, 10, 4], None, None),
        ((20, 10, 15), [1], [1000], [1], [1]),
        ((20, 10, 15), [0], [-1], [1], [1]),
        ((20, 10, 15), [20, 10, 4], [0, 0, 1], [0, 1, 2], [-1, -3, -2]),
        ((20, 10, 15), [0, 0, 3], [20, 10, 4], [0, -2, -1], None),
    ),
)
def test_slice(  # pylint: disable=missing-function-docstring
    input_shape: Tuple[int, ...],
    starts: List[int],
    ends: List[int],
    axes: Optional[List[int]],
    steps: Optional[List[int]],
) -> None:
    x = np.random.randn(*input_shape).astype(np.float32)
    _test_slice(
        input_tensor=x,
        starts=np.array(starts, dtype=np.int64),
        ends=np.array(ends, dtype=np.int64),
        axes=np.array(axes, dtype=np.int64) if axes is not None else None,
        steps=np.array(steps, dtype=np.int64) if steps is not None else None,
    )
