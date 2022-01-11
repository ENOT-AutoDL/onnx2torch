from typing import Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_resize(
        x: np.ndarray,
        scales: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        align_corners: Optional[bool] = None,
        **kwargs,
) -> None:
    if align_corners is not None:
        kwargs['coordinate_transformation_mode'] = 'align_corners'

    inputs = ['x', '']
    test_inputs = {'x': x}
    if scales is not None:
        test_inputs['scales'] = scales
        inputs.append('scales')
    else:
        inputs.append('')

    if sizes is not None:
        test_inputs['sizes'] = sizes
        inputs.append('sizes')
    else:
        inputs.append('')

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
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=13,
    )
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -7,
        atol_torch_cpu_cuda=10 ** -7,
        atol_onnx_torch2onnx=10 ** -7,
    )


def _test_resize_v10(
        x: np.ndarray,
        scales: np.ndarray = None,
        mode: str = 'nearest',
) -> None:
    test_inputs = {'x': x, 'scales': scales}

    node = onnx.helper.make_node(op_type='Resize', inputs=list(test_inputs), outputs=['y'], mode=mode)

    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[x.dtype],
            shape=None,
        ),
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=10,
    )
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -7,
        atol_torch_cpu_cuda=10 ** -7,
        atol_onnx_torch2onnx=10 ** -7,
    )


@pytest.mark.parametrize('mode', ('linear', 'nearest'))
def test_resize(mode) -> None:
    data = np.random.randint(0, 255, size=(1, 1, 20, 20), dtype=np.float32)
    if mode == 'nearest':
        # nearest scales
        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
        _test_resize(
            x=data,
            scales=scales,
            mode='nearest',
            nearest_mode='floor',
            coordinate_transformation_mode='asymmetric',
        )

        # nearest sizes
        sizes = np.array([1, 1, 10, 10], dtype=np.int64)
        _test_resize(
            x=data,
            sizes=sizes,
            mode='nearest',
            nearest_mode='floor',
            coordinate_transformation_mode='asymmetric',
        )
    else:
        # upsample
        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        _test_resize(x=data, scales=scales, mode=mode)
        _test_resize(x=data, scales=scales, mode=mode, align_corners=True)
        sizes = np.array([1, 1, 9, 10], dtype=np.int64)
        _test_resize(x=data, sizes=sizes, mode=mode)

        # downsample
        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
        _test_resize(x=data, scales=scales, mode=mode)
        _test_resize(x=data, scales=scales, mode=mode, align_corners=True)
        sizes = np.array([1, 1, 3, 3], dtype=np.int64)
        _test_resize(x=data, sizes=sizes, mode=mode)


@pytest.mark.parametrize('mode', ('linear', 'nearest'))
def test_resizeV10(mode: str) -> None:
    data = np.random.randint(0, 255, size=(1, 1, 20, 20), dtype=np.float32)
    # upsample
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    _test_resize_v10(x=data, scales=scales, mode=mode)

    # downsample
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
    _test_resize_v10(x=data, scales=scales, mode=mode)
