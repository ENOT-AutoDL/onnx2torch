import torch
from torch import nn

from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxBaseElementWise(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-docstring
    def __init__(self, onnx_export_class: CustomExportToOnnx):
        super().__init__()
        self._onnx_export_class = onnx_export_class

    @staticmethod
    def _broadcast_shape(*tensors: torch.Tensor):
        shapes = [t.shape for t in tensors]
        broadcast_shape = torch.broadcast_shapes(*shapes)
        return broadcast_shape

    def apply_reduction(self, *tensors: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        raise NotImplementedError

    def forward(self, *input_tensors: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if len(input_tensors) == 1:
            # If there is a single element, return it (no op).
            # Also, no need for manually building the ONNX node.
            return input_tensors[0]

        output = self.apply_reduction(*input_tensors)

        if torch.onnx.is_in_onnx_export():
            return self._onnx_export_class.set_output_and_apply(output, *input_tensors)

        return output
