import logging
from typing import Callable
import requests

import numpy as np
import torch
from  torchvision import datasets
import onnx
import onnxruntime

from onnx2torch import convert

_LOGGER = logging.getLogger(__name__)


def get_imagenet():
    datasets.imagenet


def get_validation(
    torch_output: torch.Tensor,
    onnx_output: np.ndarray,
    atol_onnx_torch: float = 0.0,
    ) -> None:

    if len(onnx_output) == 1:
        torch_output = [torch_output]
    for a, b in zip(onnx_output, torch_output):
        assert np.all(np.isclose(a, b, atol_onnx_torch)), 'ort and torch outputs have significant difference'

from flask import Flask

app = Flask(__name__)
