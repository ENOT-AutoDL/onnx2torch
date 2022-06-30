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
    'pads,mode',
    (
        ([0, 1, 1, 1, 1, 0, 0, 0, 1, 1], 'constant'),
        ([0, 0, 5, 3, 7, 0, 0, 2, 3, 11], 'edge'),
        ([0, 0, 1, 2, 1, 0, 0, 1, 2, 1], 'reflect'),
        ([0, 0, 0, 0, 0, 0, 0, 0], 'constant'),
        ([0, 1, 1, 1, 1, 0, 0, 0], 'constant'),
        ([0, 2, 0, 2, 0, 2, 0, 2], 'constant'),
        ([1, 2, 4, 2, 5, 4, 4, 2], 'constant'),
        ([0, 0, 0, 0, 0, 0, 0, 0], 'edge'),
        ([0, 0, 2, 3, 0, 0, 2, 3], 'edge'),
        ([0, 0, 0, 0, 0, 0, 0, 0], 'reflect'),
        ([0, 0, 2, 1, 0, 0, 2, 1], 'reflect'),
        ([0, 4, 0, 1, 0, 1], 'constant'),
        ([0, 0, 3, 0, 0, 3], 'edge'),
        ([0, 0, 1, 0, 0, 1], 'reflect'),
    ),
)
@pytest.mark.parametrize('opset_version', (2, 11, 13))
def test_pad(pads: np.array, mode: str, opset_version: int) -> None:  # pylint: disable=missing-function-docstring

    input_tensor = np.asarray(
        [
            [
                [1.0, 1.2],
                [2.3, 3.4],
                [4.5, 5.6],
            ],
            [
                [0.1, 2.1],
                [3.2, 4.3],
                [3.4, 4.5],
            ],
        ],
        dtype=np.float32,
    )

    dims = list(range(len(pads) // 2 - input_tensor.ndim))
    input_tensor = np.expand_dims(input_tensor, dims)

    _test_pad(input_array=input_tensor, mode=mode, opset_version=opset_version, pads=pads)
