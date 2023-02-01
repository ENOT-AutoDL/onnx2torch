__all__ = [
    'OnnxNonMaxSuppression',
]

from typing import Any
from typing import Dict
from typing import Optional

import torch
import torchvision
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxNonMaxSuppression(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, center_point_box: int = 0):
        super().__init__()
        self._center_point_box = center_point_box

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        return {'center_point_box_i': self._center_point_box}

    def forward(  # pylint: disable=missing-function-docstring
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        max_output_boxes_per_class: Optional[torch.Tensor] = None,
        iou_threshold: Optional[torch.Tensor] = None,
        score_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        forward_lambda = lambda: self._nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

        if torch.onnx.is_in_onnx_export():
            if max_output_boxes_per_class is None:
                max_output_boxes_per_class = torch.tensor([0], dtype=torch.int64)
            if iou_threshold is None:
                iou_threshold = torch.tensor([0.0], dtype=torch.float32)
            if score_threshold is None:
                score_threshold = torch.tensor([0.0], dtype=torch.float32)

            onnx_attrs = self._onnx_attrs(opset_version=get_onnx_version())
            return DefaultExportToOnnx.export(
                forward_lambda,
                'NonMaxSuppression',
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                onnx_attrs,
            )

        return forward_lambda()

    def _nms(
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
                if self._center_point_box:
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

                out.extend([batch_index, class_index, box_index] for box_index in indexes)
        if len(out) == 0:
            return torch.empty([0, 3], dtype=torch.int64, device=boxes.device)

        return torch.tensor(out, dtype=torch.int64, device=boxes.device)


@add_converter(operation_type='NonMaxSuppression', version=10)
@add_converter(operation_type='NonMaxSuppression', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    center_point_box = node.attributes.get('center_point_box', 0)
    return OperationConverterResult(
        torch_module=OnnxNonMaxSuppression(center_point_box=center_point_box),
        onnx_mapping=onnx_mapping_from_node(node),
    )
