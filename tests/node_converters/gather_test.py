import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_gather(
    op_type: str,
    input_array: np.ndarray,
    indices: np.ndarray,
    opset_version: int,
    **kwargs,
) -> None:
    test_inputs = {
        'x': input_array,
        'indices': indices,
    }

    node = onnx.helper.make_node(
        op_type,
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'op_type,axis,opset_version',
    (
        ('Gather', 0, 13),
        ('Gather', 0, 11),
        ('Gather', 0, 9),
        ('Gather', 1, 13),
        ('Gather', 1, 11),
        ('Gather', 1, 9),
        ('GatherElements', 0, 13),
        ('GatherElements', 0, 11),
        ('GatherElements', 1, 13),
        ('GatherElements', 1, 11),
    ),
)
def test_gather(op_type: str, axis: int, opset_version: int) -> None:  # pylint: disable=missing-function-docstring

    input_tensor = np.asarray(
        [
            [1.0, 1.2, 1.9],
            [2.3, 3.4, 3.9],
            [4.5, 5.7, 5.9],
        ],
        dtype=np.float32,
    )
    indices = np.asarray(
        [
            [1, 0],
        ],
        dtype=np.int64,
    )
    _test_gather(op_type=op_type, input_array=input_tensor, indices=indices, axis=axis, opset_version=opset_version)
