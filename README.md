<div align="center">
  <img src="assets/logo/onnx2torch_light.png#gh-light-mode-only">
  <img src="assets/logo/onnx2torch_dark.png#gh-dark-mode-only">
</div>

<p align="center">
    <a href="https://pypi.org/project/onnx2torch">
        <img src="https://badgen.net/pypi/v/onnx2torch" />
    </a>
    <a href="https://anaconda.org/conda-forge/onnx2torch">
        <img src="https://img.shields.io/conda/vn/conda-forge/onnx2torch" />
    </a>
    <a href="https://pypi.org/project/onnx2torch">
        <img src="https://badgen.net/pypi/license/onnx2torch" />
    </a>
    <a href="https://pypi.org/project/onnx2torch">
        <img src="https://img.shields.io/pypi/format/onnx2torch" />
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-black" />
    </a>
    <a href="https://github.com/ENOT-AutoDL/onnx2torch/stargazers">
        <img src="https://img.shields.io/github/stars/ENOT-AutoDL/onnx2torch.svg?style=social&label=Star&maxAge=2592000" />
    </a>
</p>

onnx2torch is an ONNX to PyTorch converter.
Our converter:
* Is easy to use – Convert the ONNX model with the function call ``convert``;
* Is easy to extend – Write your own custom layer in PyTorch and register it with ``@add_converter``;
* Convert back to ONNX – You can convert the model back to ONNX using the ``torch.onnx.export`` function.

If you find an issue, please [let us know](https://github.com/ENOT-AutoDL/onnx2torch/issues)! And feel free to create merge requests.

Please note that this converter covers only a limited number of PyTorch / ONNX models and operations.
Let us know which models you use or want to convert from onnx to torch [here](https://github.com/ENOT-AutoDL/onnx2torch/discussions).

## Installation

```bash
pip install onnx2torch
```
or
```bash
conda install -c conda-forge onnx2torch
```

## Usage

Below you can find some examples of use.

### Convert

```python
import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/some/path/mobile_net_v2.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
```

### Execute

We can execute the returned ``PyTorch model`` in the same way as the original torch model.

```python
import onnxruntime as ort
# Create example data
x = torch.ones((1, 2, 224, 224)).cuda()

out_torch = torch_model_1(x)

ort_sess = ort.InferenceSession(onnx_model_path)
outputs_ort = ort_sess.run(None, {'input': x.numpy()})

# Check the Onnx output against PyTorch
print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))
```

## Models

We have tested the following models:

Segmentation models:
- [x] DeepLabv3plus
- [x] DeepLabv3 resnet50 (torchvision)
- [x] HRNet
- [x] UNet (torchvision)
- [x] FCN resnet50 (torchvision)
- [x] lraspp mobilenetv3 (torchvision)

Detection  from MMdetection:
- [x] [SSDLite with MobileNetV2 backbone](https://github.com/open-mmlab/mmdetection)
- [x] [RetinaNet R50](https://github.com/open-mmlab/mmdetection)
- [x] [SSD300 with VGG backbone](https://github.com/open-mmlab/mmdetection)
- [x] [Yolov3_d53](https://github.com/open-mmlab/mmdetection)
- [x] [Yolov5](https://github.com/ultralytics/yolov5)

Classification from __torchvision__:
- [x] Resnet18
- [x] Resnet50
- [x] MobileNet v2
- [x] MobileNet v3 large
- [x] EfficientNet_b{0, 1, 2, 3}
- [x] WideResNet50
- [x] ResNext50
- [x] VGG16
- [x] GoogleleNet
- [x] MnasNet
- [x] RegNet

Transformers:
- [x] Vit
- [x] Swin
- [x] GPT-J

#### :page_facing_up: List of currently supported operations can be founded [here](operators.md).

## How to add new operations to converter

Here we show how to add the module:
1. Supported by both PyTorch and ONNX and has the same behaviour.

An example of such a module is [Relu](./onnx2torch/node_converters/activations.py)

```python
@add_converter(operation_type='Relu', version=6)
@add_converter(operation_type='Relu', version=13)
@add_converter(operation_type='Relu', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=nn.ReLU(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
```

Here we have registered an operation named ``Relu`` for opset versions 6, 13, 14.
Note that the ``torch_module`` argument in ``OperationConverterResult`` must be a torch.nn.Module, not just a callable object!
If Operation's behaviour differs from one opset version to another, you should implement it separately.

2. Operations supported by PyTorch and ONNX BUT have different behaviour

```python
class OnnxExpand(nn.Module, OnnxToTorchModuleWithCustomExport):

    def forward(self, input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        output = input_tensor * torch.ones(torch.Size(shape), dtype=input_tensor.dtype, device=input_tensor.device)
        if torch.onnx.is_in_onnx_export():
            return _ExpandExportToOnnx.set_output_and_apply(output, input_tensor, shape)

        return output


class _ExpandExportToOnnx(CustomExportToOnnx):

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('Expand', *args, outputs=1)


@add_converter(operation_type='Expand', version=8)
@add_converter(operation_type='Expand', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxExpand(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
```

Here we have used a trick to convert the model from torch back to ONNX by defining the custom ``_ExpandExportToOnnx``.

## Acknowledgments

Thanks to Dmitry Chudakov [@cakeofwar42](https://github.com/cakeofwar42) for his contributions.\
Special thanks to Andrey Denisov [@denisovap2013](https://github.com/denisovap2013) for the logo design.
