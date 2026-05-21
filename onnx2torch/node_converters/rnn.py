__all__ = ['OnnxLSTM', 'OnnxGRU', 'OnnxRNN']

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult

# Gate reorder: ONNX [I, O, F, C] -> PyTorch [I, F, G, O]
_LSTM_GATE_ORDER = [0, 2, 3, 1]
# Gate reorder: ONNX [Z, R, H] -> PyTorch [R, Z, N]
_GRU_GATE_ORDER = [1, 0, 2]


def _reorder_weight(weight: torch.Tensor, gate_order: List[int]) -> torch.Tensor:
    chunks = weight.chunk(len(gate_order), dim=0)
    return torch.cat([chunks[i] for i in gate_order], dim=0)


def _make_output(output_values: Tuple[str, ...], *tensors) -> Union[torch.Tensor, Tuple]:
    """Return outputs positionally: output_values[i] maps to tensors[i]."""
    if len(output_values) == 1:
        return tensors[0]
    return tuple(tensors[i] if i < len(tensors) else None for i in range(len(output_values)))


def _build_onnx_mapping(node: OnnxNode) -> OnnxMapping:
    """Map only X + inputs after W/R/B (indices 4+), skipping weight inputs."""
    inputs = [node.input_values[0]]
    if len(node.input_values) > 4:
        inputs.extend(node.input_values[4:])
    return OnnxMapping(inputs=tuple(inputs), outputs=node.output_values)


class OnnxLSTM(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, lstm: nn.LSTM, reverse: bool, output_values: Tuple[str, ...]):
        super().__init__()
        self.lstm = lstm
        self._reverse = reverse
        self._output_values = output_values

    def forward(  # pylint: disable=missing-function-docstring
        self,
        X: torch.Tensor,
        sequence_lens: Optional[torch.Tensor] = None,
        initial_h: Optional[torch.Tensor] = None,
        initial_c: Optional[torch.Tensor] = None,
        peephole: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if sequence_lens is not None:
            raise NotImplementedError('LSTM with sequence_lens is not supported')
        if peephole is not None:
            raise NotImplementedError('LSTM with peephole weights is not supported')

        hx = None
        if initial_h is not None or initial_c is not None:
            h = initial_h if initial_h is not None else torch.zeros_like(initial_c)
            c = initial_c if initial_c is not None else torch.zeros_like(initial_h)
            hx = (h, c)

        if self._reverse:
            X = X.flip(0)

        output, (h_n, c_n) = self.lstm(X, hx)

        num_directions = 2 if self.lstm.bidirectional else 1
        seq_len, batch_size, _ = output.shape
        hidden_size = self.lstm.hidden_size
        output = output.reshape(seq_len, batch_size, num_directions, hidden_size).permute(0, 2, 1, 3)

        if self._reverse:
            output = output.flip(0)

        return _make_output(self._output_values, output, h_n, c_n)


class OnnxGRU(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    """GRU converter supporting both ONNX linear_before_reset=0 (default) and =1.

    PyTorch nn.GRU uses linear_before_reset=1, which differs from the ONNX default (0).
    This class implements the correct ONNX formula with a manual timestep loop.
    Weights are stored as buffers in ONNX gate order [Z, R, H].
    """

    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool,
        reverse: bool,
        output_values: Tuple[str, ...],
        has_bias: bool,
        linear_before_reset: int,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._reverse = reverse
        self._output_values = output_values
        self._has_bias = has_bias
        self._linear_before_reset = linear_before_reset

    def _gru_step(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        wih: torch.Tensor,
        whh: torch.Tensor,
        bih: Optional[torch.Tensor],
        bhh: Optional[torch.Tensor],
    ) -> torch.Tensor:
        H = self._hidden_size

        gi = x @ wih.T
        if bih is not None:
            gi = gi + bih

        # Z and R gate hidden projections (shared for both linear_before_reset values)
        gh_zr = h @ whh[: 2 * H].T
        if bhh is not None:
            gh_zr = gh_zr + bhh[: 2 * H]

        z = torch.sigmoid(gi[:, :H] + gh_zr[:, :H])
        r = torch.sigmoid(gi[:, H : 2 * H] + gh_zr[:, H : 2 * H])

        if self._linear_before_reset == 0:
            # ONNX default: ht = tanh(Wh*x + Wbh + (r⊙h)*(Rh^T) + Rbh)
            gh_h = (r * h) @ whh[2 * H :].T
            if bhh is not None:
                gh_h = gh_h + bhh[2 * H :]
        else:
            # linear_before_reset=1: ht = tanh(Wh*x + Wbh + r⊙(Rh*h + Rbh))
            gh_h = h @ whh[2 * H :].T
            if bhh is not None:
                gh_h = gh_h + bhh[2 * H :]
            gh_h = r * gh_h

        h_new = torch.tanh(gi[:, 2 * H :] + gh_h)
        return (1 - z) * h_new + z * h

    def _run_direction(
        self,
        X: torch.Tensor,
        h0: Optional[torch.Tensor],
        wih: torch.Tensor,
        whh: torch.Tensor,
        bih: Optional[torch.Tensor],
        bhh: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = X.shape[1]
        h = h0 if h0 is not None else torch.zeros(batch_size, self._hidden_size, device=X.device, dtype=X.dtype)

        outputs = []
        for x_t in X:
            h = self._gru_step(x_t, h, wih, whh, bih, bhh)
            outputs.append(h)

        return torch.stack(outputs, dim=0), h.unsqueeze(0)  # [seq, batch, H], [1, batch, H]

    def forward(  # pylint: disable=missing-function-docstring
        self,
        X: torch.Tensor,
        sequence_lens: Optional[torch.Tensor] = None,
        initial_h: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if sequence_lens is not None:
            raise NotImplementedError('GRU with sequence_lens is not supported')

        wih_0 = getattr(self, 'weight_ih_0')
        whh_0 = getattr(self, 'weight_hh_0')
        bih_0 = getattr(self, 'bias_ih_0') if self._has_bias else None
        bhh_0 = getattr(self, 'bias_hh_0') if self._has_bias else None
        h0_fwd = initial_h[0] if initial_h is not None else None

        if self._reverse:
            Y, h_n = self._run_direction(X.flip(0), h0_fwd, wih_0, whh_0, bih_0, bhh_0)
            Y = Y.flip(0)
        elif self._bidirectional:
            wih_1 = getattr(self, 'weight_ih_1')
            whh_1 = getattr(self, 'weight_hh_1')
            bih_1 = getattr(self, 'bias_ih_1') if self._has_bias else None
            bhh_1 = getattr(self, 'bias_hh_1') if self._has_bias else None
            h0_bwd = initial_h[1] if initial_h is not None else None

            y_fwd, h_fwd = self._run_direction(X, h0_fwd, wih_0, whh_0, bih_0, bhh_0)
            y_bwd, h_bwd = self._run_direction(X.flip(0), h0_bwd, wih_1, whh_1, bih_1, bhh_1)
            Y = torch.stack([y_fwd, y_bwd.flip(0)], dim=1)  # [seq, 2, batch, H]
            h_n = torch.cat([h_fwd, h_bwd], dim=0)  # [2, batch, H]
            return _make_output(self._output_values, Y, h_n)
        else:
            Y, h_n = self._run_direction(X, h0_fwd, wih_0, whh_0, bih_0, bhh_0)

        Y = Y.unsqueeze(1)  # [seq, 1, batch, H]
        return _make_output(self._output_values, Y, h_n)


class OnnxRNN(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, rnn: nn.RNN, reverse: bool, output_values: Tuple[str, ...]):
        super().__init__()
        self.rnn = rnn
        self._reverse = reverse
        self._output_values = output_values

    def forward(  # pylint: disable=missing-function-docstring
        self,
        X: torch.Tensor,
        sequence_lens: Optional[torch.Tensor] = None,
        initial_h: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if sequence_lens is not None:
            raise NotImplementedError('RNN with sequence_lens is not supported')

        if self._reverse:
            X = X.flip(0)

        output, h_n = self.rnn(X, initial_h)

        num_directions = 2 if self.rnn.bidirectional else 1
        seq_len, batch_size, _ = output.shape
        hidden_size = self.rnn.hidden_size
        output = output.reshape(seq_len, batch_size, num_directions, hidden_size).permute(0, 2, 1, 3)

        if self._reverse:
            output = output.flip(0)

        return _make_output(self._output_values, output, h_n)


def _load_weights(
    module: nn.Module,
    W: torch.Tensor,
    R: torch.Tensor,
    B: Optional[torch.Tensor],
    gate_order: Optional[List[int]],
    num_directions: int,
    num_gates: int,
    hidden_size: int,
) -> None:
    suffixes = [''] if num_directions == 1 else ['', '_reverse']
    with torch.no_grad():
        for i, suffix in enumerate(suffixes):
            w = W[i] if gate_order is None else _reorder_weight(W[i], gate_order)
            r = R[i] if gate_order is None else _reorder_weight(R[i], gate_order)
            getattr(module, f'weight_ih_l0{suffix}').data.copy_(w)
            getattr(module, f'weight_hh_l0{suffix}').data.copy_(r)
            if B is not None:
                split = num_gates * hidden_size
                bih = B[i][:split] if gate_order is None else _reorder_weight(B[i][:split], gate_order)
                bhh = B[i][split:] if gate_order is None else _reorder_weight(B[i][split:], gate_order)
                getattr(module, f'bias_ih_l0{suffix}').data.copy_(bih)
                getattr(module, f'bias_hh_l0{suffix}').data.copy_(bhh)


def _validate_common_attrs(node: OnnxNode, op_type: str) -> Tuple:
    attrs = node.attributes
    hidden_size = attrs['hidden_size']
    direction = attrs.get('direction', 'forward')
    layout = attrs.get('layout', 0)

    if layout != 0:
        raise NotImplementedError(f'{op_type} with layout=1 (batch-major) is not supported')
    if attrs.get('clip') is not None:
        raise NotImplementedError(f'{op_type} with clip is not supported')

    bidirectional = direction == 'bidirectional'
    reverse = direction == 'reverse'
    num_directions = 2 if bidirectional else 1
    return hidden_size, direction, bidirectional, reverse, num_directions


def _get_weights(node: OnnxNode, graph: OnnxGraph, op_type: str):
    w_name = node.input_values[1]
    r_name = node.input_values[2]
    if w_name not in graph.initializers or r_name not in graph.initializers:
        raise NotImplementedError(f'{op_type} with dynamic weights (W, R not initializers) is not supported')

    W = graph.initializers[w_name].to_torch()
    R = graph.initializers[r_name].to_torch()

    b_name = node.input_values[3] if len(node.input_values) > 3 else ''
    if b_name and b_name not in graph.initializers:
        raise NotImplementedError(f'{op_type} with dynamic bias (B not an initializer) is not supported')

    B = graph.initializers[b_name].to_torch() if b_name and b_name in graph.initializers else None
    return W, R, B


@add_converter(operation_type='LSTM', version=14)
@add_converter(operation_type='LSTM', version=7)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    attrs = node.attributes
    if attrs.get('input_forget', 0) != 0:
        raise NotImplementedError('LSTM with input_forget=1 is not supported')

    p_name = node.input_values[7] if len(node.input_values) > 7 else ''
    if p_name:
        raise NotImplementedError('LSTM with peephole weights is not supported')

    hidden_size, _, bidirectional, reverse, num_directions = _validate_common_attrs(node, 'LSTM')
    W, R, B = _get_weights(node, graph, 'LSTM')

    input_size = W.shape[2]
    lstm = nn.LSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=1, bias=B is not None, bidirectional=bidirectional
    )
    _load_weights(lstm, W, R, B, _LSTM_GATE_ORDER, num_directions, 4, hidden_size)

    torch_module = OnnxLSTM(lstm=lstm, reverse=reverse, output_values=node.output_values)
    return OperationConverterResult(torch_module=torch_module, onnx_mapping=_build_onnx_mapping(node))


@add_converter(operation_type='GRU', version=14)
@add_converter(operation_type='GRU', version=7)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    attrs = node.attributes
    linear_before_reset = attrs.get('linear_before_reset', 0)

    hidden_size, _, bidirectional, reverse, num_directions = _validate_common_attrs(node, 'GRU')
    W, R, B = _get_weights(node, graph, 'GRU')

    has_bias = B is not None
    torch_module = OnnxGRU(
        hidden_size=hidden_size,
        bidirectional=bidirectional,
        reverse=reverse,
        output_values=node.output_values,
        has_bias=has_bias,
        linear_before_reset=linear_before_reset,
    )

    # Register weights as buffers in ONNX gate order [Z, R, H]
    for i in range(num_directions):
        torch_module.register_buffer(f'weight_ih_{i}', W[i])
        torch_module.register_buffer(f'weight_hh_{i}', R[i])
        if has_bias:
            split = 3 * hidden_size
            torch_module.register_buffer(f'bias_ih_{i}', B[i][:split])
            torch_module.register_buffer(f'bias_hh_{i}', B[i][split:])

    return OperationConverterResult(torch_module=torch_module, onnx_mapping=_build_onnx_mapping(node))


@add_converter(operation_type='RNN', version=14)
@add_converter(operation_type='RNN', version=7)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    hidden_size, _, bidirectional, reverse, num_directions = _validate_common_attrs(node, 'RNN')
    W, R, B = _get_weights(node, graph, 'RNN')

    activation = 'tanh'
    activations = node.attributes.get('activations')
    if activations is not None:
        name = activations[0].lower()
        if name == 'relu':
            activation = 'relu'

    input_size = W.shape[2]
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=B is not None,
        bidirectional=bidirectional,
        nonlinearity=activation,
    )
    _load_weights(rnn, W, R, B, None, num_directions, 1, hidden_size)

    torch_module = OnnxRNN(rnn=rnn, reverse=reverse, output_values=node.output_values)
    return OperationConverterResult(torch_module=torch_module, onnx_mapping=_build_onnx_mapping(node))
