from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import numpy as np
import torch


def onnx_dtype_to_torch_dtype(dtype: int) -> Union[torch.dtype, Type[str], Type[bool]]:
    """
    Convert ONNX dtype to PyTorch dtype.

    Parameters
    ----------
    dtype : int
        ONNX data type.

    Returns
    -------
    Union[torch.dtype, Type[str], Type[bool]]
        Corresponding PyTorch dtype.

    """
    # https://github.com/onnx/onnx/blob/main/onnx/onnx-ml.proto#L485
    _dtypes: Dict[int, Union[torch.dtype, Type[str], Type[bool]]] = {
        1: torch.float32,
        2: torch.uint8,
        3: torch.int8,
        # 4: UINT16 is not supported: https://github.com/pytorch/pytorch/issues/58734.
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        8: str,
        9: bool,
        10: torch.float16,
        11: torch.float64,
        # 12: UINT32 is not supported: https://github.com/pytorch/pytorch/issues/58734.
        # 13: UINT64 is not supported: https://github.com/pytorch/pytorch/issues/58734.
        14: torch.complex64,
        15: torch.complex128,
        16: torch.bfloat16,
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f'dtype={dtype} is not supported') from exc


def onnx_dtype_to_numpy_dtype(dtype: int) -> Union[np.dtype, Type[str], Type[bool]]:
    """
    Convert ONNX dtype to Numpy dtype.

    Parameters
    ----------
    dtype : int
        ONNX data type.

    Returns
    -------
    Union[torch.dtype, Type[str], Type[bool]]
        Corresponding Numpy dtype.

    """
    # https://numpy.org/doc/stable/reference/arrays.dtypes.html
    _dtypes: Dict[int, Any] = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        8: str,
        9: bool,
        10: np.float16,
        11: np.float64,
        12: np.uint32,
        13: np.uint64,
        14: np.complex64,
        15: np.complex128,
        # 16: bfloat16 is not supported.
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f'dtype={dtype} is not supported') from exc
