from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_dropout(data: np.ndarray, opset_version: int, **kwargs) -> None:
    test_inputs = {'input_tensor': data}

    if opset_version >= 12:
        if 'ratio' in kwargs:
            test_inputs['ratio'] = np.array(kwargs.pop('ratio'), dtype=np.float16)
        if 'training_mode' in kwargs:
            test_inputs['training_mode'] = np.array(kwargs.pop('training_mode'), dtype=bool)

    node = onnx.helper.make_node(op_type='Dropout', inputs=list(test_inputs), outputs=['y'], **kwargs)
    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'input_shape,ratio,training_mode,opset_version',
    (
        ([3, 32, 32], None, None, 10),
        ([3, 32, 32], None, None, 12),
        ([3, 32, 32], None, None, 13),
        ([3, 32, 32], 0.8, None, 10),
        ([3, 32, 32], 0.8, None, 12),
        ([3, 32, 32], 0.8, None, 13),
        ([3, 32, 32], 0.8, False, 13),
        ([3, 32, 32], 0.8, False, 13),
        ([8, 3, 32, 32], None, None, 10),
        ([8, 3, 32, 32, 32], None, None, 10),
    ),
)
def test_dropout(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    ratio: Optional[float],
    training_mode: Optional[bool],
    opset_version: int,
) -> None:
    data = np.random.randn(*input_shape).astype(np.float32)
    kwargs = {}
    if ratio is not None:
        kwargs['ratio'] = ratio
    if training_mode is not None:
        kwargs['training_mode'] = training_mode
    _test_dropout(data=data, opset_version=opset_version, **kwargs)
