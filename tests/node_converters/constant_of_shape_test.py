import random

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_constant_of_shape(shape: np.ndarray, value: np.ndarray) -> None:
    test_inputs = {'shape': shape}
    onnx_type = NP_TYPE_TO_TENSOR_TYPE[value.dtype]

    node = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=list(test_inputs),
        outputs=['output'],
        value=numpy_helper.from_array(value, name='value'),
    )

    outputs_info = [make_tensor_value_info(name='output', elem_type=onnx_type, shape=shape.tolist())]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_constant_of_shape() -> None:
    for _ in range(10):
        size = random.randint(1, 6)
        shape = np.random.randint(low=1, high=2, size=(size,))
        value = np.random.uniform(low=-10000, high=10000, size=(1,))
        _test_constant_of_shape(shape, value)
