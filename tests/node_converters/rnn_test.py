from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import make_tensor_value_info

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _make_lstm_model(
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    direction: str = 'forward',
    with_bias: bool = True,
    output_names: Optional[List[str]] = None,
    with_initial_h: bool = False,
    with_initial_c: bool = False,
):
    num_directions = 2 if direction == 'bidirectional' else 1
    W = np.random.randn(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    R = np.random.randn(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.randn(num_directions, 8 * hidden_size).astype(np.float32) if with_bias else None

    if output_names is None:
        output_names = ['Y', 'Y_h', 'Y_c']

    inputs_list = ['X', 'W', 'R', 'B' if with_bias else '']
    if with_initial_h or with_initial_c:
        inputs_list += ['', 'h0' if with_initial_h else '', 'c0' if with_initial_c else '']

    node = onnx.helper.make_node(
        op_type='LSTM',
        inputs=inputs_list,
        outputs=output_names,
        hidden_size=hidden_size,
        direction=direction,
    )

    initializers = {'W': W, 'R': R}
    if with_bias:
        initializers['B'] = B
    test_inputs: Dict[str, np.ndarray] = {'X': np.random.randn(seq_len, batch_size, input_size).astype(np.float32)}
    if with_initial_h:
        h0 = np.random.randn(num_directions, batch_size, hidden_size).astype(np.float32)
        test_inputs['h0'] = h0
        initializers.pop('h0', None)
    if with_initial_c:
        c0 = np.random.randn(num_directions, batch_size, hidden_size).astype(np.float32)
        test_inputs['c0'] = c0
        initializers.pop('c0', None)

    outputs_info = []
    for name in output_names:
        if not name:
            continue
        if name == 'Y':
            outputs_info.append(
                make_tensor_value_info(name, TensorProto.FLOAT, [seq_len, num_directions, batch_size, hidden_size])
            )
        else:
            outputs_info.append(
                make_tensor_value_info(name, TensorProto.FLOAT, [num_directions, batch_size, hidden_size])
            )

    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    return model, test_inputs


def _make_gru_model(
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    direction: str = 'forward',
    with_bias: bool = True,
    output_names: Optional[List[str]] = None,
):
    num_directions = 2 if direction == 'bidirectional' else 1
    W = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    R = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32) if with_bias else None

    if output_names is None:
        output_names = ['Y', 'Y_h']

    inputs_list = ['X', 'W', 'R', 'B' if with_bias else '']
    node = onnx.helper.make_node(
        op_type='GRU',
        inputs=inputs_list,
        outputs=output_names,
        hidden_size=hidden_size,
        direction=direction,
    )

    initializers = {'W': W, 'R': R}
    if with_bias:
        initializers['B'] = B
    test_inputs = {'X': np.random.randn(seq_len, batch_size, input_size).astype(np.float32)}

    outputs_info = []
    for name in output_names:
        if not name:
            continue
        if name == 'Y':
            outputs_info.append(
                make_tensor_value_info(name, TensorProto.FLOAT, [seq_len, num_directions, batch_size, hidden_size])
            )
        else:
            outputs_info.append(
                make_tensor_value_info(name, TensorProto.FLOAT, [num_directions, batch_size, hidden_size])
            )

    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    return model, test_inputs


def _make_rnn_model(
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    direction: str = 'forward',
    with_bias: bool = True,
    output_names: Optional[List[str]] = None,
):
    num_directions = 2 if direction == 'bidirectional' else 1
    W = np.random.randn(num_directions, hidden_size, input_size).astype(np.float32)
    R = np.random.randn(num_directions, hidden_size, hidden_size).astype(np.float32)
    B = np.random.randn(num_directions, 2 * hidden_size).astype(np.float32) if with_bias else None

    if output_names is None:
        output_names = ['Y', 'Y_h']

    inputs_list = ['X', 'W', 'R', 'B' if with_bias else '']
    node = onnx.helper.make_node(
        op_type='RNN',
        inputs=inputs_list,
        outputs=output_names,
        hidden_size=hidden_size,
        direction=direction,
    )

    initializers = {'W': W, 'R': R}
    if with_bias:
        initializers['B'] = B
    test_inputs = {'X': np.random.randn(seq_len, batch_size, input_size).astype(np.float32)}

    outputs_info = []
    for name in output_names:
        if not name:
            continue
        if name == 'Y':
            outputs_info.append(
                make_tensor_value_info(name, TensorProto.FLOAT, [seq_len, num_directions, batch_size, hidden_size])
            )
        else:
            outputs_info.append(
                make_tensor_value_info(name, TensorProto.FLOAT, [num_directions, batch_size, hidden_size])
            )

    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    return model, test_inputs


# ─── LSTM tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize('with_bias', [True, False])
@pytest.mark.parametrize(
    'seq_len, batch_size, input_size, hidden_size',
    [
        (5, 3, 4, 6),
        (1, 1, 8, 4),
        (10, 2, 16, 8),
    ],
)
def test_lstm_forward_all_outputs(  # pylint: disable=missing-function-docstring
    seq_len, batch_size, input_size, hidden_size, with_bias
):
    model, inputs = _make_lstm_model(seq_len, batch_size, input_size, hidden_size, with_bias=with_bias)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5, atol_torch_cpu_cuda=1e-5)


@pytest.mark.parametrize('seq_len, batch_size, input_size, hidden_size', [(5, 3, 4, 6)])
def test_lstm_forward_only_y(  # pylint: disable=missing-function-docstring
    seq_len, batch_size, input_size, hidden_size
):
    model, inputs = _make_lstm_model(seq_len, batch_size, input_size, hidden_size, output_names=['Y'])
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


@pytest.mark.parametrize('with_bias', [True, False])
def test_lstm_bidirectional(with_bias):  # pylint: disable=missing-function-docstring
    model, inputs = _make_lstm_model(5, 3, 4, 6, direction='bidirectional', with_bias=with_bias)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


def test_lstm_reverse():  # pylint: disable=missing-function-docstring
    model, inputs = _make_lstm_model(5, 3, 4, 6, direction='reverse')
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


def test_lstm_with_initial_h_and_c():  # pylint: disable=missing-function-docstring
    model, inputs = _make_lstm_model(5, 3, 4, 6, with_initial_h=True, with_initial_c=True)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


def test_lstm_with_initial_h_only():  # pylint: disable=missing-function-docstring
    model, inputs = _make_lstm_model(5, 3, 4, 6, with_initial_h=True, with_initial_c=False)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


# ─── GRU tests ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('with_bias', [True, False])
@pytest.mark.parametrize(
    'seq_len, batch_size, input_size, hidden_size',
    [
        (5, 3, 4, 6),
        (1, 1, 8, 4),
        (10, 2, 16, 8),
    ],
)
def test_gru_forward_all_outputs(  # pylint: disable=missing-function-docstring
    seq_len, batch_size, input_size, hidden_size, with_bias
):
    model, inputs = _make_gru_model(seq_len, batch_size, input_size, hidden_size, with_bias=with_bias)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5, atol_torch_cpu_cuda=1e-5, atol_onnx_torch2onnx=1e-4)


def test_gru_forward_only_y():  # pylint: disable=missing-function-docstring
    model, inputs = _make_gru_model(5, 3, 4, 6, output_names=['Y'])
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5, atol_onnx_torch2onnx=1e-4)


@pytest.mark.parametrize('with_bias', [True, False])
def test_gru_bidirectional(with_bias):  # pylint: disable=missing-function-docstring
    model, inputs = _make_gru_model(5, 3, 4, 6, direction='bidirectional', with_bias=with_bias)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5, atol_onnx_torch2onnx=1e-4)


def test_gru_reverse():  # pylint: disable=missing-function-docstring
    model, inputs = _make_gru_model(5, 3, 4, 6, direction='reverse')
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5, atol_onnx_torch2onnx=1e-4)


# ─── RNN tests ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('with_bias', [True, False])
@pytest.mark.parametrize(
    'seq_len, batch_size, input_size, hidden_size',
    [
        (5, 3, 4, 6),
        (1, 1, 8, 4),
        (10, 2, 16, 8),
    ],
)
def test_rnn_forward_all_outputs(  # pylint: disable=missing-function-docstring
    seq_len, batch_size, input_size, hidden_size, with_bias
):
    model, inputs = _make_rnn_model(seq_len, batch_size, input_size, hidden_size, with_bias=with_bias)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5, atol_torch_cpu_cuda=1e-5)


def test_rnn_forward_only_y():  # pylint: disable=missing-function-docstring
    model, inputs = _make_rnn_model(5, 3, 4, 6, output_names=['Y'])
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


@pytest.mark.parametrize('with_bias', [True, False])
def test_rnn_bidirectional(with_bias):  # pylint: disable=missing-function-docstring
    model, inputs = _make_rnn_model(5, 3, 4, 6, direction='bidirectional', with_bias=with_bias)
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)


def test_rnn_reverse():  # pylint: disable=missing-function-docstring
    model, inputs = _make_rnn_model(5, 3, 4, 6, direction='reverse')
    check_onnx_model(model, inputs, atol_onnx_torch=1e-5)
