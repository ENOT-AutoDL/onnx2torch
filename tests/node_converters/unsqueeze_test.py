from typing import Any
from typing import Dict
from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_unsqueeze(
        input_tensor: np.ndarray,
        axes: List[int],
        opset_version: int,
        **kwargs,

) -> None:
    test_inputs: Dict[str, Any] = {'input_tensor': input_tensor}

    if opset_version >= 13:
        test_inputs['axes'] = np.array(axes, dtype=np.int64)
    else:
        kwargs['axes'] = axes

    node = onnx.helper.make_node(
        op_type='Unsqueeze',
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
            shape=np.expand_dims(input_tensor, axis=axes).shape,
        ),),
    )
    check_onnx_model(model, test_inputs)


# Known warning. Shape Inference do not work properly in opset_version=9 and negative indices.
# [W:onnxruntime:, execution_frame.cc:721 VerifyOutputSizes]
# Expected shape from model of {2,3,16,16} does not match actual shape of {2,1,3,16,1,16} for output y
@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'shape,axes,opset_version',
    (
            ([2, 3, 16, 16], [0], 11),
            ([2, 3, 16, 16], [2], 11),
            ([2, 3, 16, 16], [-1], 11),
            ([2, 3, 16, 16], [-3], 11),
            ([2, 3, 16, 16], [0, 1], 11),
            ([2, 3, 16, 16], [1, 2, 3, 4, 5], 11),
            ([2, 3, 16, 16], [1, -2], 11),
            ([2, 3, 16, 16], [-2, 1], 11),

            ([2, 3, 16, 16], [0], 13),
            ([2, 3, 16, 16], [2], 13),
            ([2, 3, 16, 16], [-1], 13),
            ([2, 3, 16, 16], [-3], 13),
            ([2, 3, 16, 16], [0, 1], 13),
            ([2, 3, 16, 16], [1, 2, 3, 4, 5], 13),
            ([2, 3, 16, 16], [1, -2], 13),
            ([2, 3, 16, 16], [-2, 1], 13),
    ),
)
def test_unsqueeze(shape: List[int], axes: List[int], opset_version: int) -> None:
    x = np.random.randn(*shape).astype(np.float32)
    _test_unsqueeze(input_tensor=x, axes=axes, opset_version=opset_version)
