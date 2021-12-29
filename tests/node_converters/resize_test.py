import numpy as np
import onnx
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_resize(
        x: np.ndarray,
        scales: np.ndarray = None,
        sizes: np.ndarray = None,
        **kwargs,
) -> None:
    align_corners = kwargs.pop('align_corners', None)
    if align_corners:
        kwargs['coordinate_transformation_mode'] = 'align_corners'

    initializers = {}
    inputs = ['x', 'roi']
    test_inputs = {'x': x, 'roi': np.array([0.0])}
    if scales is not None:
        test_inputs['scales'] = scales
        inputs.append('scales')
    if sizes is not None:
        inputs.append('scales')
        test_inputs['scales'] = np.array([]).astype(np.float32)
        test_inputs['sizes'] = sizes
        inputs.append('sizes')

    node = onnx.helper.make_node(op_type='Resize', inputs=inputs, outputs=['y'], **kwargs)

    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[x.dtype],
            shape=None,
        ),
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -7,
        atol_torch_cpu_cuda=10 ** -7,
        atol_onnx_torch2onnx=10 ** -7,
    )


def test_resize() -> None:
    # Cubic
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)
    # upsample
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    _test_resize(x=data, scales=scales, mode='cubic')

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)
    # upsample
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    _test_resize(x=data, scales=scales, mode='cubic', align_corners=True)

    # linear
    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    _test_resize(x=data, scales=scales, mode='linear')

    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    _test_resize(x=data, scales=scales, mode='linear', align_corners=True)

    # nearest
    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    _test_resize(
        x=data,
        scales=scales,
        mode='nearest',
        nearest_mode='floor',
        coordinate_transformation_mode='asymmetric',
    )
