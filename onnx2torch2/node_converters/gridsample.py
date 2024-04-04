# Copyright 2023 Untether AI Inc. ALL RIGHTS RESERVED.
#
# This source code (Information) is proprietary to Untether AI Inc.
# (Untether) and MAY NOT be copied by any method, incorporated into or
# linked to another program, reverse engineered or otherwise used in
# any manner whatsoever without the express prior written consent of
# Untether. This Information and any portion thereof is and shall
# remain the property of Untether. Untether makes no representation
# or warranty regarding the accuracy of the Information nor its fitness nor
# merchantability for any purpose.  Untether assumes no responsibility
# or liability for its use in any way and conveys to the user no license,
# right or title under any patent, copyright or otherwise, and makes
# no representation or warranty that this Information does not infringe
# any third party patent, copyright or other proprietary right.

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import onnx_mapping_from_node


class UGridSampler(nn.Module):  # pylint: disable=missing-class-docstring
    def __init__(
        self,
        align_corners: int = 0,
        mode: int = 0,
        padding_mode: int = 0,
    ):
        super().__init__()
        self.align_corners = bool(align_corners)
        self.interpolation_mode = mode if mode != "linear" else "bilinear"
        self.padding_mode = padding_mode

    def forward(  # pylint: disable=missing-function-docstring
        self,
        x,
        grid,
    ) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            x, grid, mode=self.interpolation_mode, padding_mode=self.padding_mode, align_corners=self.align_corners
        )


@add_converter(operation_type='GridSample', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    align_corners = node_attributes.get('align_corners', 0)
    padding_mode = node_attributes.get('padding_mode', 0)
    mode = node_attributes.get('mode', 0)
    print(align_corners, padding_mode, mode)

    torch_module = UGridSampler(align_corners, mode, padding_mode)
    return OperationConverterResult(torch_module=torch_module, onnx_mapping=onnx_mapping_from_node(node))
