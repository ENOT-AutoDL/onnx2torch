import numpy as np
import torch
from onnx import numpy_helper
from onnx.onnx_ml_pb2 import TensorProto


class OnnxTensor:
    def __init__(self, onnx_tensor_proto: TensorProto):
        self._proto = onnx_tensor_proto

    @classmethod
    def from_numpy(cls, array: np.ndarray, name: str = None):
        onnx_tensor_proto = numpy_helper.from_array(array, name=name)
        return cls(onnx_tensor_proto)

    @classmethod
    def from_torch(cls, tensor: torch.Tensor, name: str = None):
        array = tensor.detach().cpu().numpy()
        return cls.from_numpy(array, name=name)

    @property
    def proto(self) -> TensorProto:
        return self._proto

    @property
    def name(self) -> str:
        return self._proto.name

    def to_numpy(self) -> np.ndarray:
        return numpy_helper.to_array(self._proto).copy()

    def to_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy())
