from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_shape(
        input_shape: List[int],
        opset_version: int,
        **kwargs,
) -> None:
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    test_inputs = {'x': x}

    node = onnx.helper.make_node(
        op_type='Shape',
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )
    onnx_type = NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')]
    outputs_info = [make_tensor_value_info(name='y', elem_type=onnx_type, shape=None)]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=opset_version,
    )
    check_model(model, test_inputs)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_shape() -> None:
    _test_shape(input_shape=[2, 3, 16, 16, 16], opset_version=9)
    _test_shape(input_shape=[2, 3, 16, 16], opset_version=9)
    _test_shape(input_shape=[2, 3, 16], opset_version=9)
