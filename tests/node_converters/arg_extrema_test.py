# pylint: disable=missing-docstring
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.helper import make_tensor_value_info

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    "op_type",
    (
        "ArgMax",
        "ArgMin",
    ),
)
@pytest.mark.parametrize(
    "opset_version",
    (
        11,
        12,
        13,
    ),
)
@pytest.mark.parametrize(
    "dims,axis",
    (
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ),
)
@pytest.mark.parametrize(
    "keepdims",
    (
        0,
        1,
    ),
)
@pytest.mark.parametrize(
    "select_last_index",
    (0, 1),
)
def test_arg_max_arg_min(
    op_type: str,
    opset_version: int,
    dims: int,
    axis: int,
    keepdims: int,
    select_last_index: int,
) -> None:
    input_shape = [3] * dims  # arbitrary magnitude in each dimension
    test_inputs = {"data": np.random.randn(*input_shape).astype(np.float32)}

    kwargs = {"keepdims": keepdims, "axis": axis}
    if opset_version >= 12:
        # since opset_version 12, we can specify whether to return the LAST index
        # of the max/min (respectively) occurance
        kwargs["select_last_index"] = select_last_index

    node = onnx.helper.make_node(op_type=op_type, inputs=["data"], outputs=["reduced"], **kwargs)

    # we need to specify outputs_info, since the required output type for arg max (int64)
    # is different than the input type
    outputs_info = [make_tensor_value_info(name="reduced", elem_type=onnx.TensorProto.INT64, shape=None)]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
        opset_version=opset_version,
    )

    check_onnx_model(model, test_inputs)

    # Test once again with input we know to all be the same.
    # This is a way to force the testing of the select_last_index attribute.
    # We need the min/max index to occur more than once.
    test_inputs2 = {"data": np.ones_like(test_inputs["data"])}
    check_onnx_model(model, test_inputs2)


class ArgMaxModel(torch.nn.Module):
    def __init__(self, axis: int, keepdims: bool):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.argmax(data, dim=self.axis, keepdim=self.keepdims)


class ArgMinModel(torch.nn.Module):
    def __init__(self, axis: int, keepdims: bool):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.argmin(data, dim=self.axis, keepdim=self.keepdims)


@pytest.mark.parametrize("op_type", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("opset_version", [11, 12, 13])
@pytest.mark.parametrize(
    "dims, axis",
    (
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ),
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_start_from_torch_module(
    op_type: str,
    opset_version: int,
    dims: int,
    axis: int,
    keepdims: bool,
    tmp_path: Path,
) -> None:
    """
    Test starting from a torch module, export to Onnx, then converting back to torch.
    """
    if op_type == "ArgMax":
        model = ArgMaxModel(axis=axis, keepdims=keepdims)
    else:
        model = ArgMinModel(axis=axis, keepdims=keepdims)

    input_shape = [3] * dims  # arbitrary magnitude in each dimension

    # export the pytorch model to onnx
    dummy_data = {"data": torch.randn(*input_shape)}
    input_names = ["data"]
    output_names = ["indices"]
    model_path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        (dummy_data,),
        str(model_path),
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=False,
        opset_version=opset_version,
    )

    # load the exported onnx file
    model = onnx.load(model_path)
    onnx.checker.check_model(model, False)

    test_inputs = {"data": np.random.randn(*input_shape).astype(np.float32)}
    check_onnx_model(model, test_inputs)
