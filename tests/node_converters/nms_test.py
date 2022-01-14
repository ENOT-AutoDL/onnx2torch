from typing import Optional

import numpy as np
import onnx
import pytest
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
        center_point_box: Optional[bool] = None,
) -> None:
    test_inputs = {
        'boxes': boxes,
        'scores': scores,
    }
    inputs = list(test_inputs)

    if max_output_boxes_per_class is not None:
        test_inputs['max_output_boxes_per_class'] = np.array(max_output_boxes_per_class, dtype=np.int64)
        inputs.append('max_output_boxes_per_class')
    else:
        inputs.append('')

    if iou_threshold is not None:
        test_inputs['iou_threshold'] = np.array(iou_threshold, dtype=np.float32)
        inputs.append('iou_threshold')
    else:
        inputs.append('')

    if score_threshold is not None:
        test_inputs['score_threshold'] = np.array(score_threshold, dtype=np.float32)
        inputs.append('score_threshold')
    else:
        inputs.append('')

    node = onnx.helper.make_node(
        op_type='NonMaxSuppression',
        inputs=inputs,
        outputs=['y'],
        center_point_box=center_point_box,
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


_BOXES = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, -0.1, 1.0, 0.9],
    [0.0, 10.0, 1.0, 11.0],
    [0.0, 10.1, 1.0, 11.1],
    [0.0, 100.0, 1.0, 101.0]
]], dtype=np.float32)
_SCORES = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)

_BOXES_CXCYWH_FORMAT_TEST = np.array([
    [[1.0, 1.0, 1.1, 1.1],
     [1.5, 1.5, 1.6, 1.6]]
], dtype=np.float32)
_SCORES_CXCYWH_FORMAT_TEST = np.array([[[0.9, 0.75]]], dtype=np.float32)

_BOXES_FLIPPED_COORDINATES_TEST = np.array([[
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, 0.9, 1.0, -0.1],
    [0.0, 10.0, 1.0, 11.0],
    [1.0, 10.1, 0.0, 11.1],
    [1.0, 101.0, 0.0, 100.0],
]], dtype=np.float32)
_SCORES_FLIPPED_COORDINATES_TEST = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)

_BOXES_IDENTICAL_BOXES_TEST = np.array([[[0.0, 0.0, 1.0, 1.0]] * 10], dtype=np.float32)
_SCORES_IDENTICAL_BOXES_TEST = np.array([[[0.9] * 9 + [0.91]]], dtype=np.float32)

_BOXES_LIMIT_OUT_TEST = _BOXES
_SCORES_LIMIT_OUT_TEST = _SCORES

_BOXES_1_BOX_TEST = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
_SCORES_1_BOX_TEST = np.array([[[0.9]]], dtype=np.float32)

_BOXES_SCORE_TEST = _BOXES
_SCORES_SCORE_TEST = _SCORES

_BOXES_IOU_SCORE_TEST = _BOXES
_SCORES_IOU_SCORE_TEST = _SCORES

_BOXES_2_BATCHES_TEST = np.asarray([_BOXES[0], _BOXES[0]])
_SCORES_2_BATCHES_TEST = np.asarray([_SCORES[0], _SCORES[0]])

_BOXES_2_BATCHES_2_CLASSES_TEST = np.asarray([_BOXES[0], _BOXES[0]])
_SCORES_2_BATCHES_2_CLASSES_TEST = np.asarray([
    [_SCORES[0, 0], _SCORES[0, 0][::-1]],  # 1 batch
    [_SCORES[0, 0][::-1], _SCORES[0, 0]],  # 2 batch
])

_BOXES_2_CLASSES_TEST = _BOXES
_SCORES_2_CLASSES_TEST = np.asarray([[_SCORES[0, 0], _SCORES[0, 0]]])

_BOXES_NONE_TEST = _BOXES
_SCORES_NONE_TEST = _SCORES


@pytest.mark.parametrize(
    'boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold,center_point_box',
    (
            (_BOXES_CXCYWH_FORMAT_TEST, _SCORES_CXCYWH_FORMAT_TEST, 3, 0.1, 0.0, True),  # center point box format
            # FIXME
            # flipped coordinates
            # (_BOXES_FLIPPED_COORDINATES_TEST, _SCORES_FLIPPED_COORDINATES_TEST, 3, 0.5, 0.0, None),
            (_BOXES_IDENTICAL_BOXES_TEST, _SCORES_IDENTICAL_BOXES_TEST, 3, 0.5, 0.0, None),  # identical boxes
            (_BOXES, _SCORES, 2, 0.5, 0.0, None),  # limit output size
            (_BOXES_1_BOX_TEST, _SCORES_1_BOX_TEST, 3, 0.5, 0.0, None),  # single box
            (_BOXES_SCORE_TEST, _SCORES_SCORE_TEST, 3, 0.5, 0.0, None),  # suppress by IOU
            (_BOXES_IOU_SCORE_TEST, _SCORES_IOU_SCORE_TEST, 3, 0.5, 0.4, None),  # suppress by IOU and score
            (_BOXES_2_BATCHES_TEST, _SCORES_2_BATCHES_TEST, 2, 0.5, 0.0, None),  # two batches
            (_BOXES_2_CLASSES_TEST, _SCORES_2_CLASSES_TEST, 2, 0.5, 0.0, None),  # two classes
            (
                _BOXES_2_BATCHES_2_CLASSES_TEST, _SCORES_2_BATCHES_2_CLASSES_TEST, 2, 0.5, 0.8, None,
            ),  # two batches two classes
            (_BOXES_NONE_TEST, _SCORES_NONE_TEST, 3, None, 0.4, None),  # test None params
            (_BOXES_NONE_TEST, _SCORES_NONE_TEST, 3, 0.5, None, None),  # test None params
            (_BOXES_NONE_TEST, _SCORES_NONE_TEST, None, 0.5, 0.4, None),  # test None params
            (_BOXES_NONE_TEST, _SCORES_NONE_TEST, 3, None, None, None),  # test None params
    ),
)
def test_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        max_output_boxes_per_class: Optional[int],
        iou_threshold: Optional[float],
        score_threshold: Optional[float],
        center_point_box: Optional[bool],
) -> None:
    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        center_point_box=center_point_box,
    )
