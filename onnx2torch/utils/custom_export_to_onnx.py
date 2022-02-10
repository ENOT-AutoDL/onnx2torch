__all__ = ['CustomExportToOnnx']

from typing import Any

import torch
from torch import _C as torch_C


class CustomExportToOnnx(torch.autograd.Function):
    _NEXT_OUTPUT = None

    @classmethod
    def set_output_and_apply(cls, output: Any, *args) -> Any:
        CustomExportToOnnx._NEXT_OUTPUT = output
        return cls.apply(*args)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return CustomExportToOnnx._NEXT_OUTPUT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise RuntimeError('backward called while converting to onnx')

    @staticmethod
    def symbolic(graph: torch_C.Graph, *values) -> torch_C.Value:
        raise NotImplementedError
