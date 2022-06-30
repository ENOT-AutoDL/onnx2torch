from typing import Tuple

import numpy as np
import pytest
import torchvision
from onnx import version_converter
from PIL import Image

from tests.utils.common import check_onnx_model
from tests.utils.common import check_torch_model
from tests.utils.resources import get_minimal_dataset_path
from tests.utils.resources import get_model

_COCO_MEAN = np.array([0.406, 0.485, 0.456], dtype=np.float32)
_COCO_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def create_test_batch(  # pylint: disable=missing-function-docstring
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    minimal_dataset_path = get_minimal_dataset_path()

    batch = []
    for index, image_path in enumerate(minimal_dataset_path.glob('*.jpg')):
        if index >= batch_size:
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
def test_resnet50():  # pylint: disable=missing-function-docstring
    model = get_model('resnet50')
    model = version_converter.convert_version(model, 11)

    input_name = model.graph.input[0].name
    test_inputs = {input_name: np.random.randn(1, 3, 224, 224).astype(dtype=np.float32)}

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-5,
        atol_torch_cpu_cuda=10**-5,
    )


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'model_name,resolution',
    (
        ('retinanet', (604, 604)),
        ('ssd300_vgg', (604, 604)),
        ('ssdlite', (224, 224)),
        ('yolov3_d53', (604, 604)),
        ('yolov5_ultralitics', (672, 256)),
        ('deeplabv3_mnv3_large', (320, 320)),
        ('deeplabv3_plus_resnet101', (486, 500)),
        ('hrnet', (321, 321)),
        ('unet', (320, 320)),
    ),
)
def test_onnx_models(  # pylint: disable=missing-function-docstring
    model_name: str, resolution: Tuple[int, int]
) -> None:
    model = get_model(model_name)
    input_name = model.graph.input[0].name
    test_inputs = {
        input_name: create_test_batch(batch_size=1, target_size=resolution),
    }

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-3,
        atol_torch_cpu_cuda=10**-3,
        atol_onnx_torch2onnx=10**-3,
    )


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'model_name',
    (
        'resnet18',
        'resnet50',
        'mobilenet_v2',
        'mobilenet_v3_large',
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
        'efficientnet_b3',
        'wide_resnet50_2',
        'resnext50_32x4d',
        'vgg16',
        'googlenet',
        'mnasnet1_0',
        'regnet_y_400mf',
        'regnet_y_16gf',
    ),
)
def test_torchvision_classification(model_name: str) -> None:  # pylint: disable=missing-function-docstring
    torch_model = getattr(torchvision.models, model_name)(pretrained=True)
    test_inputs = {
        'inputs': create_test_batch(batch_size=32),
    }

    check_torch_model(
        torch_model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
        atol_onnx_torch2onnx=10**-4,
    )


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'model_name',
    (
        'fcn_resnet50',
        'deeplabv3_resnet50',
        'lraspp_mobilenet_v3_large',
    ),
)
def test_torchvision_segmentation(model_name: str) -> None:  # pylint: disable=missing-function-docstring
    torch_model = getattr(torchvision.models.segmentation, model_name)(pretrained=True)
    test_inputs = {
        'inputs': create_test_batch(batch_size=8),
    }

    check_torch_model(
        torch_model,
        test_inputs,
        atol_onnx_torch=10**-3,
        atol_torch_cpu_cuda=10**-3,
        atol_onnx_torch2onnx=10**-3,
    )


@pytest.mark.filterwarnings('ignore::torch.jit._trace.TracerWarning')
@pytest.mark.parametrize(
    'model_name',
    (
        'vit',
        'swin',
    ),
)
def test_transformer_models(model_name: str) -> None:  # pylint: disable=missing-function-docstring
    model = get_model(model_name)
    input_name = model.graph.input[0].name
    test_inputs = {
        input_name: create_test_batch(batch_size=8, target_size=(224, 224)),
    }

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
        atol_onnx_torch2onnx=10**-4,
    )


def test_3d_gan() -> None:  # pylint: disable=missing-function-docstring
    model = get_model('3d_gan')
    input_name = model.graph.input[0].name
    test_inputs = {input_name: np.random.randn(32, 200).astype(dtype=np.float32)}

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
    )


def test_shelfnet() -> None:  # pylint: disable=missing-function-docstring
    model = get_model('shelfnet')
    input_name = model.graph.input[0].name
    test_inputs = {input_name: np.random.randn(8, 3, 384, 288).astype(dtype=np.float32)}

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
    )


def test_model_with_pad_node() -> None:  # pylint: disable=missing-function-docstring
    model = get_model('point_arch')
    input_name = model.graph.input[0].name
    test_inputs = {input_name: np.random.randn(1, 49, 40, 1).astype(dtype=np.float32)}

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-4,
        atol_torch_cpu_cuda=10**-4,
    )


def test_gptj() -> None:  # pylint: disable=missing-function-docstring
    model = get_model('gptj_2_random_blocks')
    input_name = model.graph.input[0].name
    test_inputs = {
        input_name: np.random.randint(low=1, high=1024, size=[4, 256], dtype=np.int64),
    }

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=10**-5,
        atol_torch_cpu_cuda=10**-5,
        atol_onnx_torch2onnx=10**-7,
    )
