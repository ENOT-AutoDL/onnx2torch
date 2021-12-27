import numpy as np
import onnx
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import calc_ort_outputs
from tests.utils.common import calc_torch_and_ort_outputs
from tests.utils.common import convert_onnx2torch2onnx
from tests.utils.common import make_model_from_nodes


def _test_resize(
        x: np.ndarray,
        scales: np.ndarray = None,
        sizes: np.ndarray = None,
        **kwargs,
) -> None:
    initializers = {}
    inputs = ['x', 'roi']
    test_inputs = {'x': x, 'roi': np.array([0.0])}
    if scales is not None:
        test_inputs['scales'] = scales
        inputs.append('scales')
    if sizes is not None:
        inputs.append('scales')
        test_inputs['scales'] = None
        test_inputs['sizes'] = sizes
        inputs.append('sizes')

    print(test_inputs, inputs)
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

    torch_outputs, ort_outputs = calc_torch_and_ort_outputs(model=model, test_inputs=test_inputs)
    (ort_outputs,) = ort_outputs

    onnx_backward_conversion_model = convert_onnx2torch2onnx(model, inputs=test_inputs)

    ort_backward_conversion_model_outputs = calc_ort_outputs(
        onnx_backward_conversion_model,
        inputs={graph_input.name: test_inputs[graph_input.name] for graph_input in
                onnx_backward_conversion_model.graph.input},
    )
    (ort_backward_conversion_model_outputs,) = ort_backward_conversion_model_outputs

    ok = ort_backward_conversion_model_outputs.shape == ort_outputs.shape
    assert ok, 'ort from onnx2torch2onnx model and ort outputs have significant difference'

    ok = np.all(torch_outputs == ort_outputs)
    assert ok, 'torch and ort outputs have significant difference'


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
    _test_resize(x=data, scales=scales, mode='nearest')

