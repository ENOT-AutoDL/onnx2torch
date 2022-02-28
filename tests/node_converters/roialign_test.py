from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def get_roi_align_input_values():  # type: ignore
    x = np.array(
        [
            [
                [
                    [
                        0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
                    ],
                    [
                        0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
                    ],
                    [
                        0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
                    ],
                    [
                        0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
                    ],
                    [
                        0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
                    ],
                    [
                        0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
                    ],
                    [
                        0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
                    ],
                    [
                        0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
                    ],
                    [
                        0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
                    ],
                    [
                        0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502,
                    ],
                ]
            ]
        ],
        dtype=np.float32,
    )
    batch_indices = np.array([0, 0, 0], dtype=np.int64)
    rois = np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32)
    return x, batch_indices, rois


def _test_roi(
        input_tensor: np.ndarray,
        rois: np.ndarray,
        batch_indices: np.ndarray,
        **kwargs,
) -> None:
    test_inputs = {'X': input_tensor, 'rois': rois, 'batch_indices': batch_indices}

    node = onnx.helper.make_node(
        op_type='RoiAlign',
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
    )
    check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'spatial_scale,sampling_ratio,output_height,output_width',
    (
        (1.0, 2, 5, 5),
        (0.25, 0, 7, 7),
        (0.125, 0, 7, 7),
        (0.6, 0, 1, 1),
        (None, None, None, None),
    )
)
@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_roi(spatial_scale: float, sampling_ratio: int, output_height: int, output_width:int) -> None:
    x, batch_indices, rois = get_roi_align_input_values()
    kwargs = {}
    if spatial_scale is not None:
        kwargs['spatial_scale'] = spatial_scale
    if sampling_ratio is not None:
        kwargs['sampling_ratio'] = sampling_ratio
    if output_height is not None:
        kwargs['output_height'] = output_height
    if output_width is not None:
        kwargs['output_width'] = output_width
    _test_roi(
        input_tensor=x,
        rois=rois,
        batch_indices=batch_indices,
        **kwargs,
    )
