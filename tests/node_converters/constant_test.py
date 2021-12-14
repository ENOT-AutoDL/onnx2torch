from typing import Tuple

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_constant_as_tensor(shape: Tuple[int, ...], dtype: np.dtype) -> None:
    values = np.random.randn(*shape).astype(dtype)
    onnx_type = NP_TYPE_TO_TENSOR_TYPE[values.dtype]
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['values'],
        value=numpy_helper.from_array(values, name='const_tensor'),
    )

    outputs_info = [make_tensor_value_info(name='values', elem_type=onnx_type, shape=values.shape)]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example={},
        outputs_info=outputs_info,
    )
    check_model(model, onnx_inputs={})


@pytest.mark.filterwarnings('ignore:No input args')
def test_constant() -> None:
    _test_constant_as_tensor((16, 16, 16), np.dtype('int32'))
    _test_constant_as_tensor((16, 16, 16), np.dtype('int32'))
    _test_constant_as_tensor((16, 16, 16), np.dtype('float32'))
