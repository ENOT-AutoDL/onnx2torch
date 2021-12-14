# onnx2torch

onnx2torch is an ONNX to PyTorch converter. 
Our converter:
* Is easy to use – Convert the ONNX model with the function call ``convert``;
* Is easy to extend – Write your own custom layer in PyTorch and register it with ``@add_converter``;
* Convert back to ONNX – You can convert the model back to ONNX using the ``torch.onnx.export`` function.

If you find an issue, please [let us know](https://github.com/ENOT-AutoDL/onnx2torch/issues)! And feel free to create merge requests.

Please note that this converter covers only a limited number of PyTorch / ONNX models and operations.  
Let us know which models you use or want to convert from onnx to torch [here](https://github.com/ENOT-AutoDL/onnx2torch/discussions).

## Installation

### From PyPi

```bash
pip install onnx2torch
```

## Usage

Below you can find some examples of use.

### Convert
```python
import torch
from onnx2torch.converter import convert

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
- [x] ResNet50
- [x] SSDLite with MobileNetV2 backbone

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
class OnnxExpand(nn.Module):

    @staticmethod
    def _do_forward(input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return input_tensor * torch.ones(torch.Size(shape), dtype=input_tensor.dtype, device=input_tensor.device)

    def forward(self, *args) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            with SkipTorchTracing():
                output = self._do_forward(*args)
                return _ExpandExportToOnnx.set_output_and_apply(output, *args)

        return self._do_forward(*args)


class _ExpandExportToOnnx(CustomExportToOnnx):

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args, **kwargs) -> torch_C.Value:
        return graph.op('Expand', *args, **kwargs, outputs=1)


@add_converter(operation_type='Expand', version=8)
@add_converter(operation_type='Expand', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxExpand(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
```

Here we have used a trick to convert the model from torch back to ONNX by defining the custom ``_ExpandExportToOnnx``.
