import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_range(
        start: np.ndarray,
        limit: np.ndarray,
        delta: np.ndarray,
) -> None:
    test_inputs = dict(start=start, limit=limit, delta=delta)
    node = onnx.helper.make_node(op_type='Range', inputs=list(test_inputs), outputs=['y'])

    num_elements = int(max(np.ceil((limit - start) / delta), 0))
    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[delta.dtype],
            shape=[num_elements],
        ),
    ]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_model(model, test_inputs)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_range() -> None:
    _test_range(
        start=np.array(1, dtype=np.int32),
        limit=np.array(5, dtype=np.int32),
        delta=np.array(2, dtype=np.int32),
    )
    _test_range(
        start=np.array(10.0, dtype=np.float32),
        limit=np.array(6.0, dtype=np.float32),
        delta=np.array(-2.3, dtype=np.float32),
    )
    _test_range(
        start=np.array(1, dtype=np.int64),
        limit=np.array(60, dtype=np.int64),
        delta=np.array(7, dtype=np.int64),
    )
