__all__ = ['OnnxNonMaxSuppression']

from typing import Optional

import torch
import torch._C as torch_C
import torchvision
from torch import nn

from onnx2torch.common import OperationConverterResult
from onnx2torch.common import onnx_mapping_from_node
from onnx2torch.common import skip_torch_tracing
from onnx2torch.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode


class OnnxNonMaxSuppression(nn.Module):

    @staticmethod
    def _do_forward(
            boxes: torch.Tensor,
            scores: torch.Tensor,
            max_output_boxes_per_class: Optional[torch.Tensor] = None,
            iou_threshold: Optional[torch.Tensor] = None,
            score_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_output_boxes_per_class is None:
            return torch.empty([0, 3], dtype=torch.int64, device=boxes.device)

        max_output_boxes_per_class = max_output_boxes_per_class.item()
        iou_threshold = 0.0 if iou_threshold is None else iou_threshold.item()
        score_threshold = 0.0 if score_threshold is None else score_threshold.item()

        out = []
        # boxes - [bs, num_boxes, 4], scores - [bs, n_classes, num_boxes]
        for batch_index, (batch_boxes, batch_scores) in enumerate(zip(boxes, scores)):
            # bbox - [num_boxes, 4], score - [n_classes, num_boxes]
            for class_index, class_scores in enumerate(batch_scores):
                confidence_mask = class_scores > score_threshold
                confidence_indexes = confidence_mask.nonzero(as_tuple=False).squeeze(1)
                nms_indexes = torchvision.ops.nms(
                    batch_boxes[confidence_indexes],
                    class_scores[confidence_indexes],
                    iou_threshold,
                )
                num_boxes = min(max_output_boxes_per_class, nms_indexes.size(0))
                nms_indexes = nms_indexes[:num_boxes]
                indexes = confidence_indexes[nms_indexes]

                out.extend(
                    [batch_index, class_index, box_index]
                    for box_index in indexes
                )

        return torch.tensor(out, dtype=torch.int64, device=boxes.device)

    def forward(self, *args) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            with skip_torch_tracing():
                output = self._do_forward(*args)
                return _NmsExportToOnnx.set_output_and_apply(output, *args)

        return self._do_forward(*args)


class _NmsExportToOnnx(CustomExportToOnnx):

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args, **kwargs) -> torch_C.Value:
        return graph.op('NonMaxSuppression', *args, **kwargs, outputs=1)


@add_converter(operation_type='NonMaxSuppression', version=10)
@add_converter(operation_type='NonMaxSuppression', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxNonMaxSuppression(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
