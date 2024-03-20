import numpy as np
import torch
from onnx import numpy_helper
from onnx.onnx_ml_pb2 import TensorProto


class OnnxTensor:  # pylint: disable=missing-class-docstring
    def __init__(self, onnx_tensor_proto: TensorProto):
        self._proto = onnx_tensor_proto

    @classmethod
    def from_numpy(cls, array: np.ndarray, name: str = None):  # pylint: disable=missing-function-docstring
        onnx_tensor_proto = numpy_helper.from_array(array, name=name)
        return cls(onnx_tensor_proto)

    @classmethod
    def from_torch(cls, tensor: torch.Tensor, name: str = None):  # pylint: disable=missing-function-docstring
        array = tensor.detach().cpu().numpy()
        return cls.from_numpy(array, name=name)

    @property
    def proto(self) -> TensorProto:  # pylint: disable=missing-function-docstring
        return self._proto

    @property
    def name(self) -> str:  # pylint: disable=missing-function-docstring
        return self._proto.name

    def to_numpy(self) -> np.ndarray:  # pylint: disable=missing-function-docstring
        return numpy_helper.to_array(self._proto).copy()

    def to_torch(self) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.from_numpy(self.to_numpy())
