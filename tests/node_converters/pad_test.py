from typing import List

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_pad(
    input_array: np.ndarray,
    opset_version: int,
    **kwargs,
) -> None:

    test_inputs = {
        'x': input_array,
    }

    if opset_version != 2:
        test_inputs['pads'] = np.array(kwargs.pop('pads'), dtype=np.int64)

    node = onnx.helper.make_node(
        'Pad',
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'input_shape,pads,mode',
    (
        ([1, 1, 1, 3, 3], [0, 1, 1, 1, 1, 0, 0, 0, 1, 1], 'constant'),
        ([1, 1, 1, 3, 3], [0, 0, 5, 3, 7, 0, 0, 2, 3, 11], 'edge'),
        ([1, 1, 3, 3, 3], [0, 0, 1, 2, 1, 0, 0, 1, 2, 1], 'reflect'),
        ([1, 1, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0], 'constant'),
        ([1, 1, 3, 3], [0, 1, 1, 1, 1, 0, 0, 0], 'constant'),
        ([1, 1, 3, 3], [0, 2, 0, 2, 0, 2, 0, 2], 'constant'),
        ([1, 1, 3, 3], [1, 2, 4, 2, 5, 4, 4, 2], 'constant'),
        ([1, 1, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0], 'edge'),
        ([1, 1, 3, 3], [0, 0, 2, 3, 0, 0, 2, 3], 'edge'),
        ([1, 1, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0], 'reflect'),
        ([1, 1, 3, 3], [0, 0, 2, 1, 0, 0, 2, 1], 'reflect'),
        ([1, 3, 3], [0, 4, 0, 1, 0, 1], 'constant'),
        ([1, 3, 3], [0, 0, 3, 0, 0, 3], 'edge'),
        ([1, 3, 3], [0, 0, 1, 0, 0, 1], 'reflect'),
        # negative padding
        ([3, 3, 3, 3, 3], [0, -1, 1, -1, 1, 0, 0, 0, 1, 1], 'constant'),
        ([3, 3, 3, 3], [0, -1, -1, -1, -1, 0, 0, 0], 'constant'),
        ([5, 7, 6], [0, -4, 0, -1, 0, 1], 'constant'),
    ),
)
@pytest.mark.parametrize('opset_version', (2, 11, 13))
def test_pad(  # pylint: disable=missing-function-docstring
    input_shape: List[int],
    pads: List[int],
    mode: str,
    opset_version: int,
) -> None:
    input_array = np.random.random(size=input_shape).astype(np.float32)
    print(len(input_array.shape), len(pads))
    _test_pad(input_array=input_array, mode=mode, opset_version=opset_version, pads=pads)
