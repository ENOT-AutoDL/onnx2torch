from typing import List
from typing import Tuple

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'equation,input_shapes,output_shape',
    (
        ('...ii ->...i', [(3, 5, 5)], (3, 5)),
        ('i,i', [(5,), (5,)], None),
        ('ij->i', [(3, 4)], (3,)),
        ('ij->ji', [(3, 4)], (4, 3)),
    ),
)
def test_einsum(  # pylint: disable=missing-function-docstring
    equation: str,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
) -> None:
    test_inputs = {f'input_{index}': np.random.randn(*shape) for index, shape in enumerate(input_shapes)}

    node = onnx.helper.make_node(
        op_type='Einsum',
        inputs=list(test_inputs),
        outputs=['out'],
        equation=equation,
    )
    outputs_info = [
        make_tensor_value_info(
            name='out',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype('float')],
            shape=output_shape,
        ),
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=13,
    )
    check_onnx_model(model, test_inputs)
