import os
from typing import Callable, Any, Optional
import requests
from pathlib import Path
import zipfile
from abc import ABCMeta, abstractclassmethod

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import DatasetFolder
from  torchvision import datasets
import onnxruntime as ort
from torchmetrics import Accuracy
from torchmetrics.detection.map import MeanAveragePrecision

from valid import convert_data_onnx2torch, convert_data_torch2onnx


class Valid(metaclass=ABCMeta):
    def __init__(self, root: str, 
        logger: Any, 
        num_workers: int,
        transformer: Any[Callable]
    ) -> None:
        self._root = root
        self._logger = logger
        self._num_workers = num_workers
        self._transformer = transformer

    @abstractclassmethod
    def valid(self, ):
        pass
    
    @abstractclassmethod
    def _get_data(self):
        pass


class CifarValid(Valid):
    def __init__(self, root: str, 
        logger: Any, 
        num_workers: int = 8,    
        transformer: Any[Callable] = None,
    ) -> None:
        super().__init__(root = root, logger = logger, transformer = transformer, num_workers = num_workers)
    
    def valid(self, onnx_model, torch_model, device: str = 'cpu', model_des: str = '') -> Callable:
        loader = self._get_data()
        torch_metric = Accuracy()
        onnx_metric = Accuracy()
        torch_model = torch_model.to(device).eval()

        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
        onnx_model = onnx_model.SerializeToString()
        if device is "cpu":
            ort_session = ort.InferenceSession(onnx_model.SerializeToString(),providers=['CPUExecutionProvider'],)
        else:
            ort_session = ort.InferenceSession(onnx_model.SerializeToString(),providers=['CUDAExecutionProvider'],)
        input_names = ort_session.get_inputs()[0].name

        for _, (sample, target) in enumerate(loader):

            input = convert_data_torch2onnx(sample)
            onnx_out = ort_session.run(output_names=None,input_feed={input_names: input},)
            onnx_out = convert_data_onnx2torch(onnx_out, device)

            sample, target = sample.to(device), target.to(device)
            torch_out = torch_model(sample).to('cpu')

            batch_torch = torch_metric(torch_out, target)
            batch_onnx = onnx_metric(onnx_out, target)

        torch_acc = torch_metric.computer()
        onnx_acc = onnx_metric.computer()
        self._logger.info('cifar valid - {:6} - torch_acc: {:.6f} - onnx_acc: {:.6f}'.format(model_des, torch_acc, onnx_acc) )
        
    def _get_data(self):

        assert self._transformer is not None
        set = datasets.CIFAR10(root = self._root,
            train = False,
            transform = self._transformer, 
            download = True,
            )
        loader = DataLoader(
            dataset = set,
            num_workers = self._num_workers,
            drop_last = False
            )
        return loader


class CoCoValid(Valid):
    
    # train_data_url = 'http://images.cocodataset.org/zips/train2014.zip'
    valid_data_url = 'http://images.cocodataset.org/zips/val2014.zip'
    annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'

    def __init__(self, root: str, 
        logger: Any, 
        num_workers: int = 8,    
        transformer: Any[Callable] = None,
    ) -> None:
        super().__init__(root = root, logger = logger, transformer = transformer, num_workers = num_workers)

    def valid(self, onnx_model, torch_model, device: str = 'cpu', model_des: str = '') -> Callable:
        loader = self._get_data()
        torch_metric = MeanAveragePrecision()
        onnx_metric = MeanAveragePrecision()
        torch_model = torch_model.to(device)

        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
        onnx_model = onnx_model.SerializeToString()
        if device is "cpu":
            ort_session = ort.InferenceSession(onnx_model.SerializeToString(),providers=['CPUExecutionProvider'],)
        else:
            ort_session = ort.InferenceSession(onnx_model.SerializeToString(),providers=['CUDAExecutionProvider'],)
        input_names = ort_session.get_inputs()[0].name

        for _, (sample, target) in enumerate(loader): 
            input = convert_data_torch2onnx(sample)
            onnx_out = ort_session.run(output_names=None,input_feed=input,)
            onnx_out = convert_data_onnx2torch(onnx_out, device)

            sample, target = sample.to(device), target.to(device)
            torch_out = torch_model(sample).to('cpu')

            batch_torch = torch_metric(torch_out, target)
            batch_onnx = onnx_metric(onnx_out, target)

        torch_acc = torch_metric.computer()
        onnx_acc = onnx_metric.computer()
        self._logger.info('cifar valid - {:6} - torch_acc: {:.6f} - onnx_acc: {:.6f}'.format(model_des, torch_acc, onnx_acc) )       

    def _get_data(self, ):
        valid_path = Path(self._root).joinpath('/coco2014/valid')
        anno_path = Path(self._root).joinpath('/coco2014/anno/')
        set = datasets.CocoDetection(
            root = valid_path, 
            annFile = anno_path, 
            transform = self._transformer
            )
        loader = DataLoader(
            dataset = set,
            num_workers = self._num_workers,
            drop_last = False
            )
        return loader
        

class ImageNetValid(Valid):
    def __init__(self, root: str, 
        logger: Any, 
        num_workers: int = 8,    
        transformer: Any[Callable] = None,
    ) -> None:
        super().__init__(root = root, logger = logger, transformer = transformer, num_workers = num_workers)
    
    def valid(self, onnx_model, torch_model, device: str = 'cpu', model_des: str = '') -> Callable:
        loader = self._get_data()
        torch_metric = Accuracy()
        onnx_metric = Accuracy()
        torch_model = torch_model.to(device).eval()

        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
        onnx_model = onnx_model.SerializeToString()
        if device is "cpu":
            ort_session = ort.InferenceSession(onnx_model.SerializeToString(),providers=['CPUExecutionProvider'],)
        else:
            ort_session = ort.InferenceSession(onnx_model.SerializeToString(),providers=['CUDAExecutionProvider'],)
        input_names = ort_session.get_inputs()[0].name

        for _, (sample, target) in enumerate(loader):
            input = convert_data_torch2onnx(sample)
            onnx_out = ort_session.run(output_names=None,input_feed={input_names: input},)
            onnx_out = convert_data_onnx2torch(onnx_out, device)

            sample, target = sample.to(device), target.to(device)
            torch_out = torch_model(sample).to('cpu')

            batch_torch = torch_metric(torch_out, target)
            batch_onnx = onnx_metric(onnx_out, target)

        torch_acc = torch_metric.computer()
        onnx_acc = onnx_metric.computer()
        self._logger.info('cifar valid - {:6} - torch_acc: {:.6f} - onnx_acc: {:.6f}'.format(model_des, torch_acc, onnx_acc) )
        
    def _get_data(self):

        assert self._transformer is not None
        os.system('./shell/build_imagenet.sh'+ self._root)
        set = DatasetFolder(
            root=Path(self._root).joinpath(Path('val')), 
            transform=self._transformer
            )
        loader = DataLoader(
            dataset = set,
            num_workers = self._num_workers,
            drop_last = False
            )
        return loader

    