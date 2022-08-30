from itertools import chain
from itertools import product
from typing import Tuple

import numpy as np
import onnx

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_conv(
    op_type: str,
    in_channels: int,
    out_channels: int,
    kernel_shape: Tuple[int, int],
    input_hw: Tuple[int, int],
    **kwargs,
) -> None:
    group = kwargs.get('group', 1)

    x_shape = (2, in_channels) + input_hw
    x = np.random.uniform(low=-1.0, high=1.0, size=x_shape).astype(np.float32)
    if op_type == 'Conv':
        weights_shape = (out_channels, in_channels // group) + kernel_shape
    elif op_type == 'ConvTranspose':
        weights_shape = (in_channels, out_channels // group) + kernel_shape
    weights = np.random.uniform(low=-1.0, high=1.0, size=weights_shape).astype(np.float32)

    test_inputs = {'x': x}
    initializers = {'weights': weights}
    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=['x', 'weights'],
        outputs=['y'],
        kernel_shape=kernel_shape,
        **kwargs,
    )

    model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
    )


def test_conv2d_base_params() -> None:  # pylint: disable=missing-function-docstring
    op_type_variants = ('ConvTranspose', 'Conv')
    in_channels_variants = (1, 2, 3, 4, 16)
    out_channels_variants = (1, 2, 3, 4, 16)
    input_hw_variants = ((32, 32), (32, 31), (31, 32), (31, 31))
    kernel_shape_variants = tuple(
        chain(
            ((i, i) for i in range(1, 6)),
            ((1, 2), (1, 3), (1, 5)),
            ((2, 2), (2, 3), (2, 5)),
        )
    )
    all_variants = product(
        op_type_variants, in_channels_variants, out_channels_variants, input_hw_variants, kernel_shape_variants
    )
    for op_type, in_channels, out_channels, input_hw, kernel_shape in all_variants:
        _test_conv(
            op_type=op_type,
            in_channels=in_channels,
            out_channels=out_channels,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
        )

    in_out_channels_variants = (2, 3, 4, 16)
    all_variants = product(op_type_variants, in_out_channels_variants, input_hw_variants, kernel_shape_variants)
    for op_type, in_out_channels, input_hw, kernel_shape in all_variants:
        _test_conv(
            op_type=op_type,
            in_channels=in_out_channels,
            out_channels=in_out_channels,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
            group=in_out_channels,
        )


def test_conv_stride_dilations_pads() -> None:  # pylint: disable=missing-function-docstring
    input_hw_variants = ((32, 32), (32, 27), (27, 32), (27, 27))
    kernel_shape_variants = tuple(
        chain(
            ((i, i) for i in range(1, 4)),
            ((1, 2), (1, 3), (2, 3)),
        )
    )
    stride_variants = (
        (1, 1),
        (2, 2),
        (3, 3),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
    )
    dilations_variants = (
        (1, 1),
        (2, 2),
        (1, 2),
        (2, 1),
    )
    pads = (
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [0, 2, 7, 0],
        [3, 0, 1, 2],
    )

    all_variants = product(
        input_hw_variants,
        kernel_shape_variants,
        stride_variants,
        dilations_variants,
        pads,
    )
    for input_hw, kernel_shape, strides, dilations, pads in all_variants:
        _test_conv(
            op_type='Conv',
            in_channels=16,
            out_channels=16,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            pads=pads,
        )

    pads = (
        [1, 1, 1, 1],
        [1, 2, 1, 2],
        [2, 2, 2, 2],
    )

    all_variants = product(
        input_hw_variants,
        kernel_shape_variants,
        stride_variants,
        dilations_variants,
        pads,
    )
    for input_hw, kernel_shape, strides, dilations, pads in all_variants:
        _test_conv(
            op_type='ConvTranspose',
            in_channels=16,
            out_channels=16,
            input_hw=input_hw,
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            pads=pads,
        )


def test_conv_transpose_output_pads() -> None:  # pylint: disable=missing-function-docstring
    input_hw_variants = ((5, 5), (6, 6), (7, 7))
    stride_variants = (
        (4, 4),
        (3, 4),
        (4, 3),
        (3, 3),
    )
    dilations_variants = (
        (3, 3),
        (2, 3),
        (3, 2),
    )
    output_pads_variants = (
        (1, 1),
        (2, 2),
        (1, 2),
    )

    all_variants = product(input_hw_variants, stride_variants, dilations_variants, output_pads_variants)
    for input_hw, strides, dilations, output_pads in all_variants:
        _test_conv(
            op_type='ConvTranspose',
            in_channels=16,
            out_channels=32,
            input_hw=input_hw,
            kernel_shape=(3, 3),
            strides=strides,
            dilations=dilations,
            output_padding=output_pads,
        )
