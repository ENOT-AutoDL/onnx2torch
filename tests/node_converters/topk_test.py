import numpy as np
import onnx
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_topk(data: np.ndarray, k: np.ndarray, **kwargs) -> None:
    test_inputs = {'input_tensor': data, 'k': k}

    node = onnx.helper.make_node(
        op_type='TopK',
        inputs=list(test_inputs),
        outputs=['y_0', 'y_1'],
        **kwargs,
    )
    outputs_info = [
        make_tensor_value_info(name='y_0', elem_type=NP_TYPE_TO_TENSOR_TYPE[data.dtype], shape=None),
        make_tensor_value_info(name='y_1', elem_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')], shape=None),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_model(model, test_inputs)


def test_topk() -> None:
    x = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=np.float32)

    _test_topk(data=x, k=np.array([3], dtype=np.int64), axis=1, largest=1)
    _test_topk(data=x, k=np.array([3], dtype=np.int64), axis=-1, largest=1)
    _test_topk( data=x, k=np.array([3], dtype=np.int64), axis=1, largest=1, sorted=1)
