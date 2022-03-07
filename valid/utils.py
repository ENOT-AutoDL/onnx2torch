
from typing import Any, Tuple, List, Dict, Type, Callable

import torch
import numpy as np
from timeit import timeit


def convert_data_onnx2torch(data: Any, device: str = 'cpu') -> Any:
    def convert_function(t):
        return torch.from_numpy(t).to(device=device)

    return _convert_data(data, from_type=np.ndarray, convert_function=convert_function)


def convert_data_torch2onnx(data: Any) -> Any:
    def convert_function(t):
        return t.detach().cpu().numpy()

    return _convert_data(data, from_type=torch.Tensor, convert_function=convert_function)


def _convert_data(data: Any, from_type: Type, convert_function: Callable) -> Any:
    if isinstance(data, Dict):
        return {
            k: _convert_data(v, from_type, convert_function)
            for k, v in data.items()
        }

    if isinstance(data, (Tuple, List)):
        return type(data)(
            _convert_data(v, from_type, convert_function)
            for v in data
        )

    if isinstance(data, from_type):
        return convert_function(data)

    return data
