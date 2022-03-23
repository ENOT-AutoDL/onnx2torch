from typing import Optional

import numpy as np
import onnx
import pytest
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_resize(
        x: np.ndarray,
        scales: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        align_corners: bool = False,
        **kwargs,
) -> None:
    if align_corners:
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
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -6,
        atol_torch_cpu_cuda=10 ** -6,
        atol_onnx_torch2onnx=10 ** -6,
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
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -7,
        atol_torch_cpu_cuda=10 ** -7,
        atol_onnx_torch2onnx=10 ** -7,
    )


_UPSAMPLE_SIZES = np.array([1, 1, 500, 500]).astype(np.int64)
_UPSAMPLE_SCALES = np.array([1.0, 1.0, 2.0, 2.0]).astype(np.float32)

_DOWNSAMPLE_SIZES = np.array([1, 1, 125, 125]).astype(np.int64)
_DOWNSAMPLE_SCALES = np.array([1.0, 1.0, 0.5, 0.5]).astype(np.float32)

_DATA = np.random.normal(scale=3.0, size=[1, 1, 250, 250]).astype(np.float32)


@pytest.mark.parametrize(
    'sizes,scales,mode,coordinate_transformation_mode',
    (
            (_UPSAMPLE_SIZES, None, 'linear', 'half_pixel'),
            (None, _UPSAMPLE_SCALES, 'linear', 'half_pixel'),
            (_DOWNSAMPLE_SIZES, None, 'linear', 'half_pixel'),
            (None, _DOWNSAMPLE_SCALES, 'linear', 'half_pixel'),
            (_UPSAMPLE_SIZES, None, 'nearest', 'asymmetric'),
            (None, _UPSAMPLE_SCALES, 'nearest', 'asymmetric'),
            (_DOWNSAMPLE_SIZES, None, 'nearest', 'asymmetric'),
            (None, _DOWNSAMPLE_SCALES, 'nearest', 'asymmetric'),
            (_UPSAMPLE_SIZES, None, 'cubic', 'half_pixel'),
            (None, _UPSAMPLE_SCALES, 'cubic', 'half_pixel'),
            (_DOWNSAMPLE_SIZES, None, 'cubic', 'half_pixel'),
            (None, _DOWNSAMPLE_SCALES, 'cubic', 'half_pixel'),
    )
)
def test_resize(
        sizes: np.ndarray,
        scales: np.ndarray,
        mode: str,
        coordinate_transformation_mode: str,
) -> None:

    _test_resize(
        x=_DATA,
        sizes=sizes,
        scales=scales,
        mode=mode,
        nearest_mode='floor',
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


@pytest.mark.parametrize('mode', ('nearest',))
def test_resizeV10(mode: str) -> None:
    _test_resize_v10(x=_DATA, scales=_UPSAMPLE_SCALES, mode=mode)
    _test_resize_v10(x=_DATA, scales=_DOWNSAMPLE_SCALES, mode=mode)
