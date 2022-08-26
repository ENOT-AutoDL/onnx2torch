from typing import Optional
from typing import Tuple
from typing import Union

from torch import nn

from onnx2torch.node_converters.pad import OnnxPadStatic


def is_symmetric_onnx_padding(padding: Tuple[int, ...]) -> bool:  # pylint: disable=missing-function-docstring
    half_len = len(padding) // 2
    return padding[:half_len] == padding[half_len:]


def onnx_auto_pad_to_torch_padding(  # pylint: disable=missing-function-docstring
    auto_pad: str,
    onnx_padding: Tuple[int, ...],
) -> Tuple[Union[int, Tuple[int, ...]], Optional[nn.Module]]:
    if auto_pad == 'NOTSET':
        if onnx_padding is None:
            return 0, None

        if is_symmetric_onnx_padding(onnx_padding):
            half_len = len(onnx_padding) // 2
            return onnx_padding[:half_len], None

        return 0, OnnxPadStatic.create_from_onnx_params(onnx_pads=onnx_padding)

    if auto_pad == 'VALID':
        return 0, None

    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        raise NotImplementedError(f'"{auto_pad}" auto_pad is not implemented')

    raise ValueError(f'Got unexpected auto_pad value "{auto_pad}"')
