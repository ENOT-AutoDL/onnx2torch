import numpy as np
import onnx
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def where_test(
        condition: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
) -> None:
    test_inputs = {'condition': condition, 'x': x, 'y': y}
    node = onnx.helper.make_node(
        op_type='Where',
        inputs=list(test_inputs),
        outputs=['z'],
    )
    outputs_info = [
        make_tensor_value_info(
            name='z',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[x.dtype],
            shape=None,
        )
    ]
    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs, outputs_info=outputs_info)
    check_model(model, test_inputs)


def test_where() -> None:
    where_test(
        condition=np.array([[1, 0], [1, 1]], dtype=bool),
        x=np.array([[1, 2], [3, 4]], dtype=np.int64),
        y=np.array([[9, 8], [7, 6]], dtype=np.int64),
    )

    where_test(
        condition=np.array([[1, 0], [1, 1]], dtype=bool),
        x=np.array([[1, 2], [3, 4]], dtype=np.float32),
        y=np.array([[9, 8], [7, 6]], dtype=np.float32),
    )

    where_test(
        condition=np.array([[1, 0], [1, 1]], dtype=bool),
        x=np.array([[1, ], [3, ]], dtype=np.float32),
        y=np.array([[9, 8], [7, 6]], dtype=np.float32),
    )
