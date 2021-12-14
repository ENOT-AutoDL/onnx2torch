import numpy as np
import onnx

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_gather(
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
        'Gather',
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
    check_model(model, test_inputs)


def test_gather() -> None:
    input_tensor = np.asarray(
        [
            [1.0, 1.2, 1.9],
            [2.3, 3.4, 3.9],
            [4.5, 5.7, 5.9],
        ],
        dtype=np.float32
    )
    indices = np.asarray(
        [
            [1, 0],
        ],
        dtype=np.int64,
    )
    _test_gather(input_array=input_tensor, indices=indices, axis=0, opset_version=9)
    _test_gather(input_array=input_tensor, indices=indices, axis=1, opset_version=9)
    _test_gather(input_array=input_tensor, indices=indices, axis=0, opset_version=13)
    _test_gather(input_array=input_tensor, indices=indices, axis=1, opset_version=13)
