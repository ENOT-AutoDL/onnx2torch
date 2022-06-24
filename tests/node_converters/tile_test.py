import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_tile(
        data: np.ndarray,
        repeats: np.ndarray,
        desire_out: np.ndarray,
) -> None:
    test_inputs = {'input_tensor': data, 'repeats': repeats}
    node = onnx.helper.make_node(
        op_type='Tile',
        inputs=list(test_inputs),
        outputs=['y'],
    )
    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[data.dtype],
            shape=desire_out.shape
        ),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_tile() -> None:
    data = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(data),)).astype(np.int64)
    _test_tile(
        data=data,
        repeats=repeats,
        desire_out=np.tile(data, repeats),
    )

    data = np.array([
        [0, 1],
        [2, 3]
    ], dtype=np.float32)

    repeats = np.array([2, 2], dtype=np.int64)
    _test_tile(
        data=data,
        repeats=repeats,
        desire_out=np.array(
            [
                [0, 1, 0, 1],
                [2, 3, 2, 3],
                [0, 1, 0, 1],
                [2, 3, 2, 3]
            ], dtype=np.float32),
    )
