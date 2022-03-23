__all__ = ['OnnxShape']

from typing import Optional

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxShape(nn.Module, OnnxToTorchModuleWithCustomExport):

    def __init__(self, start: Optional[int] = None, end: Optional[int] = None):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = torch.tensor(
            input_tensor.shape[self.start:self.end],
            device=input_tensor.device,
        )
        if torch.onnx.is_in_onnx_export():
            args = [
                input_tensor,
            ]
            if self.start is not None:
                args.append(self.start)
                if self.end is not None:
                    args.append(self.end)
            elif self.end is not None:
                args += [0, self.end]

            return _ShapeExportToOnnx.set_output_and_apply(output, *args)

        return output


class _ShapeExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        if len(args) == 2:
            return graph.op('Shape', args[0], start_i=args[1], outputs=1)

        if len(args) == 3:
            return graph.op('Shape', args[0], start_i=args[1], end_i=args[2], outputs=1)

        return graph.op('Shape', *args, outputs=1)


@add_converter(operation_type='Shape', version=1)
@add_converter(operation_type='Shape', version=13)
@add_converter(operation_type='Shape', version=15)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxShape(
            start=node.attributes.get('start', None),
            end=node.attributes.get('end', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
