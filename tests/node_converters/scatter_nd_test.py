import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_scatter_nd(
        data: np.ndarray,
        indices: np.ndarray,
        updates: np.ndarray,
        **kwargs,
) -> None:
    test_inputs = {'data': data, 'indices': indices, 'updates': updates}

    node = onnx.helper.make_node(
        op_type='ScatterND',
        inputs=['data', 'indices', 'updates'],
        outputs=['y'],
        **kwargs,
    )

    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs)
    check_onnx_model(model, test_inputs)


def test_scatter_nd() -> None:
    data = np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
    ], dtype=np.float32)

    indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    updates = np.array([1232, 5463], dtype=np.float32)
    _test_scatter_nd(
        data=data,
        indices=indices,
        updates=updates,
    )

    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    updates = np.array([
        [8, 7, 6, 5],
        [4, 3, 2, 1],
    ], dtype=np.float32)

    _test_scatter_nd(
        data=data,
        indices=indices,
        updates=updates,
    )

    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array([
        [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
    ], dtype=np.float32)
    _test_scatter_nd(
        data=data,
        indices=indices,
        updates=updates,
    )
