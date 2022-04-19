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
        ([0,0,0,0,0,0,0,0], 'constant'),
        ([0,1,1,1,1,0,0,0], 'constant'),
        ([0,2,0,2,0,2,0,2], 'constant'),
        ([1,2,4,2,5,4,4,2], 'constant'),
        ([0,0,0,0,0,0,0,0], 'edge'),
        ([0,0,2,3,0,0,2,3], 'edge'),
        ([0,0,0,0,0,0,0,0], 'reflect'),
        ([0,0,2,1,0,0,2,1], 'reflect'),
    )
)
def test_pad(pads: np.array, mode: str) -> None:

    input_tensor = np.asarray(
        [
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
                ]
            ]
        ],
        dtype=np.float32
    )

    opset_version_variants = (2, 11, 13)

    for opset_version in opset_version_variants:
        _test_pad(input_array=input_tensor, mode=mode, opset_version=opset_version, pads=pads)
