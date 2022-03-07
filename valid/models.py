from pathlib import Path

from loguru import logger
import torch
import onnx
from torchvision import models

from valid import CifarValid, ImageNetValid, CoCoValid
from onnx2torch import convert


logger_formate = "{time} - {level} - {file} - {line} - {message}"
logger.add(Path("logpath.log"), formate = logger_formate)

transformer = ''

coco = CoCoValid(root = '', logger = logger, num_workers = 8, transformer = transformer)
cifar = CifarValid(root = '', logger = logger, num_workers = 8, transformer = transformer)
imagenet = ImageNetValid(root = '', logger = logger, num_workers = 8, transformer = transformer)


def cifar_valid_resnet50():
    onnx_model = onnx.load('path to imagnet resnet50')
    torch_model = convert(onnx_model)
    cifar.valid(onnx_model, torch_model, device = 'cpu',  model_des = 'resnet50')

def imagenet_valid_resnet50():
    onnx_model = onnx.load('path to cifar10 resnet50')
    torch_model = convert(onnx_model)
    imagenet.valid(onnx_model, torch_model, device = 'cpu',  model_des = 'resnet50')

def coco_valid_resnet50():
    onnx_model = onnx.load('path to coco2014 resnet50')
    torch_model = convert(onnx_model)
    coco.valid(onnx_model, torch_model, device = 'cpu',  model_des = 'resnet50')






