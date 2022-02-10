import tarfile
from pathlib import Path

import requests
from google_drive_downloader import GoogleDriveDownloader

from tests import DATASETS_DIR
from tests import MODELS_DIR

_ONNX_MODELS_URLS = {
    'resnet50': 'https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx',
}

_ONNX_MODELS_IDS = {
    'deeplabv3_mnv3_large': '1XIH4kBrGYNYtrW6EtSsPrfmjxGKT8u93',
    'deeplabv3_plus_resnet101': '1ZTP4FCs8ileMm7ZZMwb1X1hP2KmXcU1X',
    'hrnet': '1Bp1bSaWq9Yrad7IejeRquw6zAVWqEWsl',
    'mask_rcnn': '1Tw7I07D3ZYo-x7GaJ_nFavG0qyKNyvee',
    'retinanet': '1VeFvfiMwIaw1VIipzMzfBm-nzUGq-Xbv',
    'ssd300_vgg': '1740qkxD5dSBuPnN5s7HfyvQy20x68lYJ',
    'ssdlite': '1E-sqOn-DVrQ-mLtBr9elHTTh8BvMIyFK',
    'unet': '1CEp3h3_4viSgcMOtjwhT4h7PPsbjqzZs',
    'yolov3_d53': '1XLjil1iBDsIVwpi9Yp7aEbuJMnlmCsyc',
    'yolov5_ultralitics': '1JCXndU8Y5wv_Yx_3ahzIzooO-m4iwiP4',
}

_MINIMAL_DATASETS_ID = '1Vd7qfQotrRADPLFxViA2tRpz7tBymR31'


def get_model_path(name: str) -> Path:
    model_path = MODELS_DIR / f'{name}.onnx'
    if not model_path.exists():
        if name in _ONNX_MODELS_URLS:
            url = _ONNX_MODELS_URLS[name]
            with model_path.open(mode='wb') as model_file:
                response = requests.get(url, stream=True)
                for chunk in response.iter_content(chunk_size=4*1024):
                    model_file.write(chunk)
        elif name in _ONNX_MODELS_IDS:
            GoogleDriveDownloader.download_file_from_google_drive(
                file_id=_ONNX_MODELS_IDS[name],
                dest_path=model_path,
                overwrite=True,
            )
        else:
            raise RuntimeError('Cannot find model path.')

    return model_path


def get_minimal_dataset_path():
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
