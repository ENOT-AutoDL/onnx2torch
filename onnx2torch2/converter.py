import inspect
from collections import OrderedDict
from operator import getitem
from pathlib import Path
from typing import Union

import torch
from onnx.onnx_ml_pb2 import ModelProto
from torch import fx
from torch import nn

from onnx2torch.node_converters import get_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_graph import ValueType
from onnx2torch.utils.safe_shape_inference import safe_shape_inference


def _remove_initializers_from_input(model: ModelProto) -> ModelProto:
    graph_inputs = model.graph.input
    graph_inputs_mapping = {one_input.name: one_input for one_input in graph_inputs}

    for initializer in model.graph.initializer:
        if initializer.name in graph_inputs_mapping:
            graph_inputs.remove(graph_inputs_mapping[initializer.name])

    return model


class InitializersContainer(nn.Module):
    """Module for storing initializers in torch fx graph."""

    def add_initializer(self, name: str, initializer: torch.Tensor) -> None:  # pylint: disable=missing-docstring
        self.register_buffer(name, initializer)

    def forward(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
        raise RuntimeError('Got unexpected "forward" on constant container')


def convert(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    onnx_model_or_path: Union[str, Path, ModelProto],
    save_input_names: bool = False,
    attach_onnx_mapping: bool = False,
) -> fx.GraphModule:
    """Convert model from onnx to PyTorch.

    This function build torch.fx GraphModule from onnx ModelProto using operations from the converter registry.
    The registered operation can be found in onnx2torch/node_converters.

    Usage example:

        from onnx2torch import convert
        torch_module = convert('path/to/onnx_model.onnx')


    Parameters
    ----------
    onnx_model_or_path : Union[str, Path, ModelProto]
        Onnx ModelProto or model path to convert.
    save_input_names : bool
        Whether to use original onnx inputs names as fx graph placeholders names or to use generated names (input_n).
        False by default.
    attach_onnx_mapping : bool
        Whether to attach info about mapping to original onnx tensors names.

    Returns
    -------
    fx.GraphModule
        PyTorch GraphModule

    """

    onnx_model = safe_shape_inference(onnx_model_or_path)

    if onnx_model.ir_version < 3:
        raise NotImplementedError('Onnx IR is too old (minimal supported version is 3).')

    onnx_model = _remove_initializers_from_input(onnx_model)
    opset_import = {opsetid_proto.domain: opsetid_proto.version for opsetid_proto in onnx_model.opset_import}

    onnx_graph = OnnxGraph(onnx_model.graph)  # pylint: disable=no-member
    torch_graph = fx.Graph()

    torch_initializers = InitializersContainer()
    torch_modules = nn.Module()
    torch_modules.add_module('initializers', torch_initializers)
    torch_nodes = {}

    # create input nodes
    for input_value, name in enumerate(onnx_graph.input_values, 1):
        if save_input_names:
            if not name.isidentifier():
                raise ValueError(f'Input name "{name}" cannot be used as name of placeholder in fx.GraphModule.')

            placeholder_name = name
        else:
            placeholder_name = f'input_{input_value}'

        torch_nodes[name] = torch_graph.placeholder(name=placeholder_name)

    # create intermediate nodes
    # IMPORTANT: nodes already topologically sorted
    for name, onnx_node in onnx_graph.nodes.items():
        version = opset_import[onnx_node.domain]
        converter = get_converter(
            domain=onnx_node.domain,
            operation_type=onnx_node.operation_type,
            version=version,
        )

        torch_module, onnx_mapping = converter(onnx_node, onnx_graph)
        if attach_onnx_mapping:
            setattr(torch_module, 'onnx_mapping', onnx_mapping)

        torch_modules.add_module(name, torch_module)

        args = []
        for value_name in onnx_mapping.inputs:
            value_type = onnx_graph.value_type(value_name)
            if value_type == ValueType.GRAPH_INPUT:
                args.append(torch_nodes[value_name])

            elif value_type == ValueType.NODE_OUTPUT:
                onnx_input_node, _ = onnx_graph.value_as_node_output(value_name)
                torch_input_node = torch_nodes[onnx_input_node.unique_name]

                # Get only one needed output of torch_input_node by index
                if len(onnx_input_node.output_values) > 1:
                    index = onnx_input_node.output_values.index(value_name)
                    torch_input_node = torch_graph.call_function(getitem, args=(torch_input_node, index))
                    torch_nodes[name + '_split_output'] = torch_input_node
                args.append(torch_input_node)

            elif value_type == ValueType.GRAPH_INITIALIZER:
                # The name of pytorch buffer must not contain '.'(dot)
                len_torch_initializers = sum(1 for _ in torch_initializers.buffers())
                torch_buffer_name = f'onnx_initializer_{len_torch_initializers}'
                if value_name not in torch_nodes:
                    torch_initializers.add_initializer(
                        torch_buffer_name,
                        onnx_graph.initializers[value_name].to_torch(),
                    )
                    torch_nodes[torch_buffer_name] = torch_graph.get_attr(f'initializers.{torch_buffer_name}')
                args.append(torch_nodes[torch_buffer_name])

            elif value_type == ValueType.EMPTY:
                args.append(None)

            else:
                raise RuntimeError(f'Got unexpected input value type ({value_type})')

        # Collect kwargs if there are some skipped args
        kwargs = {}
        if None in args:
            first_skipped_arg = args.index(None)
            forward_args = tuple(inspect.signature(torch_module.forward).parameters.keys())
            forward_args = forward_args[first_skipped_arg : len(args)]
            args, kwargs_values = args[:first_skipped_arg], args[first_skipped_arg:]
            kwargs.update({name: value for name, value in zip(forward_args, kwargs_values) if value is not None})

        torch_nodes[name] = torch_graph.call_module(module_name=name, args=tuple(args), kwargs=kwargs)

    # Create output nodes
    onnx_output_nodes = [onnx_graph.value_as_node_output(value_name)[0] for value_name in onnx_graph.output_values]
    # Delete duplicates and save order
    onnx_output_nodes = list(OrderedDict.fromkeys(onnx_output_nodes))

    torch_output_nodes = [torch_nodes[onnx_node.unique_name] for onnx_node in onnx_output_nodes]
    if len(torch_output_nodes) == 1:
        torch_output_nodes = torch_output_nodes[0]
    torch_graph.output(torch_output_nodes)

    torch_graph.lint()
    torch_model = fx.GraphModule(root=torch_modules, graph=torch_graph)

    return torch_model
