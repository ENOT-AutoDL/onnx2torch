from typing import Optional

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from tests.utils.common import check_model
from tests.utils.common import make_model_from_nodes


def _test_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        max_output_boxes_per_class: Optional[int] = None,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
) -> None:
    test_inputs = {
        'boxes': boxes,
        'scores': scores,
    }
    if max_output_boxes_per_class is not None:
        test_inputs['max_output_boxes_per_class'] = np.array(max_output_boxes_per_class, dtype=np.int64)
    if iou_threshold is not None:
        test_inputs['iou_threshold'] = np.array(iou_threshold, dtype=np.float32)
    if score_threshold is not None:
        test_inputs['score_threshold'] = np.array(score_threshold, dtype=np.float32)

    node = onnx.helper.make_node(
        op_type='NonMaxSuppression',
        inputs=list(test_inputs),
        outputs=['y'],
    )

    outputs_info = [
        make_tensor_value_info(
            name='y',
            elem_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
            shape=None,
        )
    ]

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        outputs_info=outputs_info,
    )
    check_model(model, test_inputs)


def test_nms() -> None:
    # 1 batch 1 class
    boxes = np.array([
        [[0.0, 0.0, 1.0, 1.0],
         [0.0, 0.1, 1.0, 0.0],
         [0.0, 0.9, 1.0, 0.95],
         [0.0, 0.8, 0.8, 0.9],
         [0.4, 0.1, 0.8, 0.5],
         [0.3, 0.4, 0.7, 0.6]]
    ], dtype=np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)

    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=3,
        iou_threshold=0.5,
        score_threshold=0.0,
    )
    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=3,
        iou_threshold=0.5,
        score_threshold=0.4,
    )

    # 1 batch 2 classes
    scores = np.array([
        [
            [0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
            [0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
        ]
    ], dtype=np.float32)

    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=2,
        iou_threshold=0.5,
        score_threshold=0.0,
    )

    # 2 batches 2 classes
    boxes = np.array([
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 0.0],
            [0.0, 0.9, 1.0, 0.95],
            [0.0, 0.8, 0.8, 0.9],
            [0.4, 0.1, 0.8, 0.5],
            [0.3, 0.4, 0.7, 0.6]
        ],
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 0.0],
            [0.0, 0.9, 1.0, 0.95],
            [0.0, 0.8, 0.8, 0.9],
            [0.4, 0.1, 0.8, 0.5],
            [0.3, 0.4, 0.7, 0.6]
        ]
    ], dtype=np.float32)
    scores = np.array([
        [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
        [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
    ], dtype=np.float32)

    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=2,
        iou_threshold=0.5,
        score_threshold=0.0,
    )
    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=2,
        iou_threshold=0.5,
        score_threshold=None,
    )
    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=2,
        iou_threshold=None,
        score_threshold=None,
    )
    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=None,
        iou_threshold=None,
        score_threshold=None,
    )
