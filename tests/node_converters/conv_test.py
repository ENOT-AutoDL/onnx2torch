from itertools import chain
from itertools import product
from typing import Tuple

import numpy as np
import onnx

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_conv(
        in_channels: int,
        out_channels: int,
        kernel_shape: Tuple[int, int],
        input_hw: Tuple[int, int],
        **kwargs,
) -> None:
    group = kwargs.get('group', 1)

    x_shape = (2, in_channels) + input_hw
    x = np.random.uniform(low=-1.0, high=1.0, size=x_shape).astype(np.float32)
    weights_shape = (out_channels, in_channels//group) + kernel_shape
    weights = np.random.uniform(low=-1.0, high=1.0, size=weights_shape).astype(np.float32)

    test_inputs = {'x': x}
    initializers = {'weights': weights}
    node = onnx.helper.make_node(
        op_type='Conv',
        inputs=['x', 'weights'],
        outputs=['y'],
        kernel_shape=kernel_shape,
        **kwargs,
    )

    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
        atol_onnx_torch2onnx=10**-4,
    )


def test_conv2d_base_params() -> None:
    in_channels_variants = (1, 2, 3, 4, 16)
    out_channels_variants = (1, 2, 3, 4, 16)
    input_hw_variants = ((32, 32), (32, 31), (31, 31))
    kernel_shape_variants = tuple(chain(
        ((i, i) for i in range(1, 6)),
        ((1, 2), (1, 3), (1, 5)),
        ((2, 2), (2, 3), (2, 5)),
    ))
    all_variants = product(in_channels_variants, out_channels_variants, input_hw_variants, kernel_shape_variants)
    for in_channels, out_channels, input_hw, kernel_shape in all_variants:
        _test_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
        )

    in_out_channels_variants = (2, 3, 4, 16)
    all_variants = product(in_out_channels_variants, input_hw_variants, kernel_shape_variants)
    for in_out_channels, input_hw, kernel_shape in all_variants:
        _test_conv(
            in_channels=in_out_channels,
            out_channels=in_out_channels,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
            group=in_out_channels,
        )


def test_conv_stride_dilations_pads() -> None:
    input_hw_variants = ((32, 32), (32, 27), (27, 27))
    kernel_shape_variants = tuple(chain(
        ((i, i) for i in range(1, 4)),
        ((1, 2), (1, 3), (2, 3)),
    ))
    stride_variants = (
        (1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (1, 3), (3, 1),
    )
    dilations_variants = (
        (1, 1), (2, 2), (1, 2), (2, 1),
    )
    all_variants = product(input_hw_variants, kernel_shape_variants, stride_variants, dilations_variants)
    for input_hw, kernel_shape, strides, dilations in all_variants:
        _test_conv(
            in_channels=16,
            out_channels=16,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
        )
