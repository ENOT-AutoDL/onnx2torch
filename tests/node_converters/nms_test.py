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


_BOXES_0 = np.array([
    [[0.0, 0.0, 1.0, 1.0],
     [0.0, 0.1, 1.0, 0.0],
     [0.0, 0.9, 1.0, 0.95],
     [0.0, 0.8, 0.8, 0.9],
     [0.4, 0.1, 0.8, 0.5],
     [0.3, 0.4, 0.7, 0.6]]
], dtype=np.float32)
_SCORES_0 = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)

_BOXES_1 = np.array([[
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, 0.9, 1.0, -0.1],
    [0.0, 10.0, 1.0, 11.0],
    [1.0, 10.1, 0.0, 11.1],
    [1.0, 101.0, 0.0, 100.0],
]], dtype=np.float32)
_SCORES_1 = _SCORES_0

_BOXES_2 = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],

    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
]], dtype=np.float32)
_SCORES_2 = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.95, 0.9, 0.9, 0.9, 0.9]]], dtype=np.float32)

_BOXES_3 = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, -0.1, 1.0, 0.9],
    [0.0, 10.0, 1.0, 11.0],
    [0.0, 10.1, 1.0, 11.1],
    [0.0, 100.0, 1.0, 101.0]
]], dtype=np.float32)
_SCORES_3 = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)

_BOXES_4 = np.array([[
    [0.0, 0.0, 1.0, 1.0]
]], dtype=np.float32)
_SCORES_4 = np.array([[[0.9]]], dtype=np.float32)

_BOXES_5 = _BOXES_3
_SCORES_5 = _SCORES_3

_BOXES_6 = _BOXES_3
_SCORES_6 = _SCORES_3

_BOXES_7 = np.asarray([_BOXES_3[0], _BOXES_3[0]])
_SCORES_7 = np.asarray([_SCORES_3[0], _SCORES_3[0]])

_BOXES_8 = _BOXES_3
_SCORES_8 = np.asarray([[_SCORES_3[0, 0], _SCORES_3[0, 0]]])


@pytest.mark.parametrize(
    'boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold',
    (
            (_BOXES_0, _SCORES_0, 3, 0.5, 0.0),
            (_BOXES_1, _SCORES_1, 3, 0.5, 0.0),  # flipped coordinates
            (_BOXES_2, _SCORES_2, 3, 0.5, 0.0),  # identical boxes
            (_BOXES_3, _SCORES_3, 2, 0.5, 0.0),  # limit output size
            (_BOXES_4, _SCORES_4, 3, 0.5, 0.0),  # single box
            (_BOXES_5, _SCORES_5, 3, 0.5, 0.0),  # suppress by IOU
            (_BOXES_6, _SCORES_6, 3, 0.5, 0.4),  # suppress by IOU and score
            (_BOXES_7, _SCORES_7, 2, 0.5, 0.0),  # two batches
            (_BOXES_8, _SCORES_8, 2, 0.5, 0.0),  # two classes

            (_BOXES_6, _SCORES_6, 3, None, 0.4),  # test None params
            (_BOXES_6, _SCORES_6, 3, 0.5, None),  # test None params
            (_BOXES_6, _SCORES_6, None, 0.5, 0.4),  # test None params
            (_BOXES_6, _SCORES_6, 3, None, None),  # test None params
    ),
)
def test_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        max_output_boxes_per_class: Optional[int],
        iou_threshold: Optional[float],
        score_threshold: Optional[float],
) -> None:
    _test_nms(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )