from itertools import product
from typing import List

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_concat(
        input_arrays_shapes: List[List[int]],
        opset_version: int,
        **kwargs,
) -> None:
    test_inputs = {}
    for i, input_array_shape in enumerate(input_arrays_shapes):
        x = np.random.uniform(low=-1.0, high=1.0, size=input_array_shape).astype(np.float32)
        node_name = f'x_{i}'
        test_inputs[node_name] = x

    node = onnx.helper.make_node(
        'Concat',
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )

    onnx_type = NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')]
    outputs_info = [make_tensor_value_info(name='y', elem_type=onnx_type, shape=None)]
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=opset_version,
    )
    check_onnx_model(model, test_inputs)


def test_concat() -> None:
    opset_variants = (9, 13)
    axis_variants = (0, 1)
    for opset_version, axis in product(opset_variants, axis_variants):
        _test_concat(
            input_arrays_shapes=[[1, 3, 16, 16], [1, 3, 16, 16], [1, 3, 16, 16]],
            axis=axis,
            opset_version=opset_version,
        )
