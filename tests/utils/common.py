import io
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import torch

import numpy as np
import onnx
import onnxruntime as ort

from onnx import defs
from onnx import numpy_helper
from onnx.helper import make_graph
from onnx.helper import make_model
from onnx.helper import make_operatorsetid
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.onnx_ml_pb2 import ModelProto
from onnx.onnx_ml_pb2 import NodeProto
from onnx.onnx_ml_pb2 import ValueInfoProto
from onnx.shape_inference import infer_shapes

from onnx2torch.converter import convert


def make_model_from_nodes(
        nodes: Union[NodeProto, Sequence[NodeProto]],
        initializers: Dict[str, np.ndarray],
        inputs_example: Optional[Dict[str, np.ndarray]] = None,
        inputs_info: Optional[Sequence[ValueInfoProto]] = None,
        outputs_info: Optional[Sequence[ValueInfoProto]] = None,
        opset_version: Optional[int] = 11,
) -> ModelProto:
    if inputs_info is None and inputs_example is None:
        raise ValueError('inputs_example or inputs_info must be set')

    if inputs_info is None:
        inputs_info = []
        for name, data in inputs_example.items():
            elem_type = NP_TYPE_TO_TENSOR_TYPE[data.dtype]
            inputs_info.append(make_tensor_value_info(name=name, elem_type=elem_type, shape=data.shape))

    if outputs_info is None:
        outputs_info = []
        elem_type = inputs_info[0].type.tensor_type.elem_type
        for name in tuple(nodes.output):
            output_proto = make_tensor_value_info(name=name, elem_type=elem_type, shape=None)
            outputs_info.append(output_proto)

    graph_proto = make_graph(
        nodes=(nodes,),
        name='test_graph',
        inputs=inputs_info,
        outputs=outputs_info,
        initializer=[
            numpy_helper.from_array(data, name=name)
            for name, data in initializers.items()
        ],
    )

    opset_imports = None
    if opset_version is not None:
        opset_imports = [
            make_operatorsetid(
                domain=defs.ONNX_DOMAIN,
                version=opset_version,
            ),
        ]

    model = make_model(graph_proto, opset_imports=opset_imports)
    model = infer_shapes(model, check_type=False)
    onnx.checker.check_model(model, False)

    return model


def _convert_data(data: Any, from_type: Type, convert_function: Callable) -> Any:
    if isinstance(data, Dict):
        return {
            k: _convert_data(v, from_type, convert_function)
            for k, v in data.items()
        }

    if isinstance(data, (Tuple, List)):
        return type(data)(
            _convert_data(v, from_type, convert_function)
            for v in data
        )

    if isinstance(data, from_type):
        return convert_function(data)

    return data


def convert_data_onnx2torch(data: Any, device: str = 'cpu') -> Any:
    def convert_function(t):
        return torch.from_numpy(t).to(device=device)

    return _convert_data(data, from_type=np.ndarray, convert_function=convert_function)


def convert_data_torch2onnx(data: Any) -> Any:
    def convert_function(t):
        return t.detach().cpu().numpy()

    return _convert_data(data, from_type=torch.Tensor, convert_function=convert_function)


def convert_onnx_inputs_to_torch_inputs(
        onnx_model: ModelProto,
        onnx_inputs: Dict[str, Any],
        device: str = 'cpu',
) -> List[Any]:
    return [
        convert_data_onnx2torch(onnx_inputs[graph_input.name], device=device)
        for graph_input in onnx_model.graph.input
        if graph_input.name in onnx_inputs
    ]


def calc_ort_outputs(model: ModelProto, inputs: Dict[str, Any], skip_unused_inputs: bool = False) -> List[Any]:
    ort_session = ort.InferenceSession(
        model.SerializeToString(),
        providers=['CPUExecutionProvider'],
    )

    if skip_unused_inputs:
        graph_inputs = [i.name for i in model.graph.input]
        inputs = {
            k: v for k, v in inputs.items()
            if k in graph_inputs
        }

    outputs = ort_session.run(
        output_names=None,
        input_feed=inputs,
    )

    return outputs


def calc_torch_outputs(model: ModelProto, inputs: Dict[str, Any], device: str = 'cpu') -> Any:
    inputs = convert_onnx_inputs_to_torch_inputs(onnx_model=model, onnx_inputs=inputs, device=device)
    model = convert(model).to(device=device)
    outputs = model(*inputs)

    return convert_data_torch2onnx(outputs)


def calc_torch_and_ort_outputs(
        model: ModelProto,
        test_inputs: Dict[str, np.ndarray],
):
    torch_outputs = calc_torch_outputs(model=model, inputs=test_inputs)
    ort_outputs = calc_ort_outputs(model=model, inputs=test_inputs)

    return torch_outputs, ort_outputs


def convert_onnx2torch2onnx(
        model: ModelProto,
        inputs: Dict[str, np.ndarray],
        opset_version: int = 13,
        **export_kwargs,
) -> ModelProto:
    torch_model = convert(model)
    input_names = list(inputs.keys())
    args = list(inputs.values())
    args = tuple(torch.tensor(arg) for arg in args)

    with io.BytesIO() as tmp_file:
        torch.onnx.export(
            model=torch_model,
            args=args,
            f=tmp_file,
            input_names=input_names,
            opset_version=opset_version,
            **export_kwargs,
        )

        return onnx.load_from_string(tmp_file.getvalue())


def _check_onnx_model(
        onnx_model: ModelProto,
        onnx_inputs: Dict[str, Any],
        onnx_torch_check_function: Callable,
        torch_cpu_cuda_check_function: Optional[Callable] = None,
        onnx_torch2onnx_check_function: Optional[Callable] = None,
        opset_version: int = 13,
) -> None:
    ort_outputs = calc_ort_outputs(onnx_model, onnx_inputs)
    torch_outputs = calc_torch_outputs(onnx_model, onnx_inputs, device='cpu')

    onnx_torch_check_function(ort_outputs, torch_outputs)

    if torch_cpu_cuda_check_function is not None:
        torch_cuda_outputs = calc_torch_outputs(onnx_model, onnx_inputs, device='cuda')
        torch_cpu_cuda_check_function(torch_outputs, torch_cuda_outputs)

    if onnx_torch2onnx_check_function is not None:
        torch2onnx_model = convert_onnx2torch2onnx(onnx_model, inputs=onnx_inputs, opset_version=opset_version)
        ort_torch2onnx_outputs = calc_ort_outputs(torch2onnx_model, onnx_inputs, skip_unused_inputs=True)
        onnx_torch2onnx_check_function(ort_outputs, ort_torch2onnx_outputs)


def check_onnx_model(
        onnx_model: ModelProto,
        onnx_inputs: Dict[str, Any],
        atol_onnx_torch: float = 0.0,
        atol_torch_cpu_cuda: float = 0.0,
        atol_onnx_torch2onnx: float = 0.0,
        opset_version: int = 13,
) -> None:
    def onnx_torch_check_function(onnx_output, torch_output):
        if len(onnx_output) == 1:
            torch_output = [torch_output]

        for a, b in zip(onnx_output, torch_output):
            assert np.all(np.isclose(a, b, atol=atol_onnx_torch)), 'ort and torch outputs have significant difference'

    def torch_cpu_cuda_check_function(torch_cpu_output, torch_cuda_output):
        if not isinstance(torch_cpu_output, (List, Tuple)):
            torch_cpu_output = [torch_cpu_output]
            torch_cuda_output = [torch_cuda_output]

        for a, b in zip(torch_cpu_output, torch_cuda_output):
            assert np.all(np.isclose(a, b, atol=atol_torch_cpu_cuda)), \
                'torch cpu and torch cuda outputs have significant difference'

        return True

    def onnx_torch2onnx_check_function(onnx_output, torch2onnx_output):
        for a, b in zip(onnx_output, torch2onnx_output):
            assert np.all(np.isclose(a, b, atol=atol_onnx_torch2onnx)), \
                'ort and ort+torch2onnx outputs have significant difference'

        return True

    _check_onnx_model(
        onnx_model=onnx_model,
        onnx_inputs=onnx_inputs,
        onnx_torch_check_function=onnx_torch_check_function,
        torch_cpu_cuda_check_function=torch_cpu_cuda_check_function,
        onnx_torch2onnx_check_function=onnx_torch2onnx_check_function,
        opset_version=opset_version,
    )


def check_torch_model(
        torch_model: torch.nn.Module,
        onnx_inputs: Dict[str, Any],
        atol_onnx_torch: float = 0.0,
        atol_torch_cpu_cuda: float = 0.0,
        atol_onnx_torch2onnx: float = 0.0,
        opset_version: int = 13,
) -> None:
    arguments = locals()
    input_names = list(onnx_inputs.keys())
    args = tuple(torch.tensor(arg) for arg in onnx_inputs.values())

    with io.BytesIO() as tmp_file:
        torch.onnx.export(
            model=torch_model,
            args=args,
            f=tmp_file,
            input_names=input_names,
            opset_version=opset_version,
        )

        arguments.pop('torch_model')
        arguments['onnx_model'] = onnx.load_from_string(tmp_file.getvalue())
        check_onnx_model(**arguments)
