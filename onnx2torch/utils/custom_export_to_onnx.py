__all__ = [
    'CustomExportToOnnx',
    'OnnxToTorchModuleWithCustomExport',
]

from typing import Any

import torch
from torch import _C as torch_C

from onnx2torch.utils.common import OnnxToTorchModule


class OnnxToTorchModuleWithCustomExport(OnnxToTorchModule):
    """
    Marker class for onnx2torch modules with custom export to onnx.
    """

    pass  # pylint: disable=unnecessary-pass


class CustomExportToOnnx(torch.autograd.Function):  # pylint: disable=missing-class-docstring
    _NEXT_OUTPUT = None

    @classmethod
    def set_output_and_apply(cls, output: Any, *args) -> Any:  # pylint: disable=missing-function-docstring
        CustomExportToOnnx._NEXT_OUTPUT = output
        return cls.apply(*args)

    @staticmethod
    def forward(  # pylint: disable=unused-argument, missing-function-docstring
        ctx: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return CustomExportToOnnx._NEXT_OUTPUT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=unused-argument, missing-function-docstring
        raise RuntimeError('backward called while converting to onnx')

    @staticmethod
    def symbolic(  # pylint: disable=unused-argument, missing-function-docstring
        graph: torch_C.Graph,
        *values,
    ) -> torch_C.Value:
        raise NotImplementedError
