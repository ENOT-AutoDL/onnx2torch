import tarfile
import urllib.request
from pathlib import Path

import onnx
from google_drive_downloader import GoogleDriveDownloader
from onnx import ModelProto  # pylint: disable=no-name-in-module

from tests import DATASETS_DIR
from tests import MODELS_DIR

_BASE_URL = 'https://gitlab.expasoft.com/p.ivanov/onnx2torch_data/-/raw/main/models_for_tests'

_CHKP_DETECTION_URL = f'{_BASE_URL}/detection'
_CHKP_SEGMENTATION_URL = f'{_BASE_URL}/segmentation'
_CHKP_TRANSFORMERS_URL = f'{_BASE_URL}/transformers'
_CHKP_KEYPOINTS_URL = f'{_BASE_URL}/keypoints'
_CHKP_OTHER_URL = f'{_BASE_URL}/other'

_ONNX_MODELS_IDS = {
    'deeplabv3_mnv3_large': f'{_CHKP_SEGMENTATION_URL}/deeplabv3_mobilenet_v3_large.onnx',
    'deeplabv3_plus_resnet101': f'{_CHKP_SEGMENTATION_URL}/deeplabv3_resnet101_dimans.onnx',
    'hrnet': f'{_CHKP_SEGMENTATION_URL}/hrnet.onnx',
    'unet': f'{_CHKP_SEGMENTATION_URL}/unet_resnet34.onnx',
    'retinanet': f'{_CHKP_DETECTION_URL}/retinanet_r50_fpn.onnx',
    'ssd300_vgg': f'{_CHKP_DETECTION_URL}/ssd300.onnx',
    'ssdlite': f'{_CHKP_DETECTION_URL}/ssdlite.onnx',
    'yolov3_d53': f'{_CHKP_DETECTION_URL}/yolov3_d53_tuned_shape.onnx',
    'yolov5_ultralitics': f'{_CHKP_DETECTION_URL}/yolov5_ultralitics.onnx',
    'swin': f'{_CHKP_TRANSFORMERS_URL}/swin.onnx',
    'vit': f'{_CHKP_TRANSFORMERS_URL}/vit.onnx',
    'gptj_2_random_blocks': f'{_CHKP_TRANSFORMERS_URL}/gptj_2_random_blocks.onnx',
    'resnet50': 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx',
    '3d_gan': f'{_CHKP_OTHER_URL}/3d_gan.onnx',
    'shelfnet': f'{_CHKP_KEYPOINTS_URL}/shelfnet.onnx',
    'point_arch': f'{_CHKP_OTHER_URL}/point_arch_nq.onnx',
}

_MINIMAL_DATASETS_ID = '1Vd7qfQotrRADPLFxViA2tRpz7tBymR31'


def get_model_path(name: str) -> Path:  # pylint: disable=missing-function-docstring
    model_path = MODELS_DIR / f'{name}.onnx'
    if not model_path.exists():
        if name in _ONNX_MODELS_IDS:
            url = _ONNX_MODELS_IDS[name]
            urllib.request.urlretrieve(url=url, filename=model_path)
        else:
            raise RuntimeError('Cannot find model path.')

    return model_path


def get_model(name: str) -> ModelProto:  # pylint: disable=missing-function-docstring
    model_path = get_model_path(name)
    return onnx.load_model(str(model_path))


def get_minimal_dataset_path():  # pylint: disable=missing-function-docstring
    dataset_path = DATASETS_DIR / 'minimal_dataset'
    if not dataset_path.exists():
        arch_path = dataset_path.with_suffix('.tar.gz')
        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=_MINIMAL_DATASETS_ID,
            dest_path=arch_path,
            overwrite=True,
        )
        with tarfile.open(arch_path, 'r:gz') as arch_file:
            arch_file.extractall(path=dataset_path)

    return dataset_path
