from typing import Tuple

import numpy as np
import onnx
import pytest
from PIL import Image
from onnx import version_converter

from tests.utils.common import check_model
from tests.utils.resources import get_minimal_dataset_path
from tests.utils.resources import get_model_path

_COCO_MEAN = np.array([0.406, 0.485, 0.456], dtype=np.float32)
_COCO_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def create_test_batch(n: int = 32, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    minimal_dataset_path = get_minimal_dataset_path()

    batch = []
    for i, image_path in enumerate(minimal_dataset_path.glob('*.jpg')):
        if i >= n:
            break

        image = Image.open(image_path).convert('RGB')
        image = image.resize(size=target_size)
        image = (np.array(image, dtype=np.float32) / 255.0 - _COCO_MEAN) / _COCO_STD
        image = image.transpose([2, 0, 1])

        batch.append(image)
    else:
        raise ValueError('Batch size ({n}) is too large.')

    return np.array(batch)


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_resnet50():
    model_path = get_model_path('resnet50')
    model = onnx.load_model(str(model_path.resolve()))
    model = version_converter.convert_version(model, 11)

    input_name = model.graph.input[0].name
    test_inputs = {
        input_name: np.random.randn(1, 3, 224, 224).astype(dtype=np.float32)
    }

    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -5,
        atol_torch_cpu_cuda=10 ** -5,
        atol_onnx_torch2onnx=10 ** -5,
    )


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
def test_ssdlite() -> None:
    model_path = get_model_path('ssdlite')
    model = onnx.load_model(str(model_path.resolve()))

    input_name = model.graph.input[0].name
    test_inputs = {
        input_name: create_test_batch(n=32),
    }

    check_model(
        model,
        test_inputs,
        atol_onnx_torch=10 ** -4,
        atol_torch_cpu_cuda=10 ** -4,
        atol_onnx_torch2onnx=10 ** -4,
    )
