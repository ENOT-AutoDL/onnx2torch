__all__ = [
    'CustomExportToOnnx',
    'DefaultExportToOnnx',
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


class CustomExportToOnnx(torch.autograd.Function):
    """Customizes ONNX exporting from PyTorch."""

    _NEXT_FORWARD_FUNCTION: Optional[Callable] = None

    @classmethod
    def export(cls, forward_function: Callable, *args) -> Any:
        """
        Substitues custom forward function.
        This function is closely related to forward function, it substitues `forward_function` to real forward.

        Old name: `set_forward_and_apply`.
        """
        CustomExportToOnnx._NEXT_FORWARD_FUNCTION = forward_function
        return cls.apply(*args)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=unused-argument
        """Applies custom forward function."""
        if CustomExportToOnnx._NEXT_FORWARD_FUNCTION is None:
            raise RuntimeError('Forward function is not set')

        try:
            return CustomExportToOnnx._NEXT_FORWARD_FUNCTION()  # pylint: disable=not-callable
        finally:
            CustomExportToOnnx._NEXT_FORWARD_FUNCTION = None

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=unused-argument, missing-function-docstring
        raise RuntimeError('Backward called while converting to ONNX')

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:  # pylint: disable=unused-argument
        """Export implementation. Return ONNX operation from this function using graph."""
        raise NotImplementedError


class DefaultExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    """
    CustomExportToOnnx with default symbolic method implementation.

    Please follow our convention, args consists of:
        - op_type
        - operation inputs
        - operation attributes

    DO NOT REORDER!

    Note: the number of operation outputs can be added later.

    This class should be used in most cases:
    >>> return DefaultExportToOnnx.export(_forward, op_type, *inputs, onnx_attrs)
    """

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        op_type, *inputs, onnx_attrs = args
        return graph.op(op_type, *inputs, **onnx_attrs, outputs=1)
