from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_squeeze(
        input_tensor: np.ndarray,
        axes: Optional[List[int]],
        opset_version: int,
        **kwargs,

) -> None:
    test_inputs: Dict[str, Any] = {'input_tensor': input_tensor}

    if axes is not None and len(axes) > 0:
        if opset_version >= 13:
            test_inputs['axes'] = np.array(axes, dtype=np.int64)
        else:
            kwargs['axes'] = axes

        output_shape = np.squeeze(
            input_tensor,
            axis=tuple(a for a in axes if input_tensor.shape[a] == 1)
        ).shape
    else:
        output_shape = np.squeeze(input_tensor).shape

    node = onnx.helper.make_node(
        op_type='Squeeze',
        inputs=list(test_inputs),
        outputs=['y'],
        **kwargs,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
        outputs_info=(make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[input_tensor.dtype],
            shape=output_shape,
        ),),
    )
    check_model(model, test_inputs)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'shape,axes,opset_version',
    (
            ([1, 3, 4, 5], [0], 11),
            ([1, 3, 1, 5], [-2], 11),
            ([1, 3, 1, 5], [0, 2], 11),
            ([1, 3, 1, 5], [2, 0], 11),
            ([1, 3, 1, 5], [0, -2], 11),
            ([1, 3, 1, 5], [-2, 0], 11),
            ([1, 3, 1, 5], None, 11),
            ([1, 1, 1, 1], None, 11),
            ([1], None, 11),
            ([3, 3, 3], None, 11),

            ([1, 3, 4, 5], [0], 13),
            ([1, 3, 1, 5], [-2], 13),
            ([1, 3, 1, 5], [0, 2], 13),
            ([1, 3, 1, 5], [2, 0], 13),
            ([1, 3, 1, 5], [0, -2], 13),
            ([1, 3, 1, 5], [-2, 0], 13),
            ([1, 3, 1, 5], None, 13),
            ([1, 1, 1, 1], None, 13),
            ([1], None, 13),
            ([3, 3, 3], None, 13),
    ),
)
def test_squeeze(shape: List[int], axes: List[int], opset_version: int) -> None:
    x = np.random.randn(*shape).astype(np.float32)
    _test_squeeze(input_tensor=x, axes=axes, opset_version=opset_version)
