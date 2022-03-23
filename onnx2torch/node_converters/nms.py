__all__ = ['OnnxNonMaxSuppression']

from typing import Optional

import torch
import torch._C as torch_C
import torchvision
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxNonMaxSuppression(nn.Module, OnnxToTorchModuleWithCustomExport):

    def __init__(self, center_point_box: bool = False):
        super().__init__()
        self.center_point_box = center_point_box

    def _do_forward(
            self,
            boxes: torch.Tensor,
            scores: torch.Tensor,
            max_output_boxes_per_class: Optional[torch.Tensor],
            iou_threshold: Optional[torch.Tensor],
            score_threshold: Optional[torch.Tensor],
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

                filtered_batch_boxes = batch_boxes[confidence_indexes]
                if self.center_point_box:
                    filtered_batch_boxes = torchvision.ops.box_convert(
                        filtered_batch_boxes,
                        in_fmt='cxcywh',
                        out_fmt='xyxy',
                    )

                nms_indexes = torchvision.ops.nms(
                    boxes=filtered_batch_boxes,
                    scores=class_scores[confidence_indexes],
                    iou_threshold=iou_threshold,
                )
                num_boxes = min(max_output_boxes_per_class, nms_indexes.size(0))
                nms_indexes = nms_indexes[:num_boxes]
                indexes = confidence_indexes[nms_indexes]

                out.extend(
                    [batch_index, class_index, box_index]
                    for box_index in indexes
                )
        if len(out) == 0:
            return torch.empty([0, 3], dtype=torch.int64, device=boxes.device)

        return torch.tensor(out, dtype=torch.int64, device=boxes.device)

    def forward(
            self,
            boxes: torch.Tensor,
            scores: torch.Tensor,
            max_output_boxes_per_class: Optional[torch.Tensor] = None,
            iou_threshold: Optional[torch.Tensor] = None,
            score_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self._do_forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
        if torch.onnx.is_in_onnx_export():
            if max_output_boxes_per_class is None:
                max_output_boxes_per_class = torch.tensor([0], dtype=torch.int64)
            if iou_threshold is None:
                iou_threshold = torch.tensor([0.0], dtype=torch.float32)
            if score_threshold is None:
                score_threshold = torch.tensor([0.0], dtype=torch.float32)

            return _NmsExportToOnnx.set_output_and_apply(
                output,
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                int(self.center_point_box),
            )

        return output


class _NmsExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method

    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box = args
        return graph.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            center_point_box_i=center_point_box,
            outputs=1,
        )


@add_converter(operation_type='NonMaxSuppression', version=10)
@add_converter(operation_type='NonMaxSuppression', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxNonMaxSuppression(center_point_box=node.attributes.get('center_point_box', 0) == 1),
        onnx_mapping=onnx_mapping_from_node(node),
    )
