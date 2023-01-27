__all__ = [
    'CustomExportToOnnx',
    'OnnxToTorchModuleWithCustomExport',
]

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import torch
from torch import _C as torch_C

from onnx2torch.utils.common import OnnxToTorchModule


class OnnxToTorchModuleWithCustomExport(OnnxToTorchModule):
    """
    Marker class for onnx2torch modules with custom export to onnx.
    """

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """
        Returns ONNX attributes with their values as a dictionary.

        Parameters
        ----------
        opset_version : int
            ONNX opset version.
            The number of attributes, their names and values depend on opset version;
            function should return correct set of attributes.

        Returns
        -------
        Dict[str, Any]
            ONNX attributes.

        """
        return {}


class CustomExportToOnnx(torch.autograd.Function):  # pylint: disable=missing-class-docstring
    _NEXT_FORWARD_FUNCTION: Optional[Callable] = None

    @classmethod
    def set_forward_and_apply(  # pylint: disable=missing-function-docstring
        cls, forward_function: Callable, *args
    ) -> Any:
        CustomExportToOnnx._NEXT_FORWARD_FUNCTION = forward_function
        return cls.apply(*args)

    @staticmethod
    def forward(  # pylint: disable=unused-argument, missing-function-docstring
        ctx: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if CustomExportToOnnx._NEXT_FORWARD_FUNCTION is None:
            raise RuntimeError('forward function is not set')

        try:
            return CustomExportToOnnx._NEXT_FORWARD_FUNCTION()  # pylint: disable=not-callable
        finally:
            CustomExportToOnnx._NEXT_FORWARD_FUNCTION = None

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=unused-argument, missing-function-docstring
        raise RuntimeError('backward called while converting to onnx')

    @staticmethod
    def symbolic(  # pylint: disable=unused-argument, missing-function-docstring
        graph: torch_C.Graph,
        *values,
    ) -> torch_C.Value:
        raise NotImplementedError
