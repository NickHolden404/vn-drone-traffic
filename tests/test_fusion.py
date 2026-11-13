import numpy as np
import pytest

from cvt.detect.base import Detections
from cvt.detect.fusion import box_iou, nms, wbf


def D(boxes, scores, labels):
    return Detections(np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32),
                      np.array(labels, np.int64))


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 1 / 3, 0.0])


def test_nms_is_classwise_and_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 10]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5)
    assert keep.tolist() == [0, 2]


def test_wbf_agreement_averages_and_single_model_box_is_halved():
    a = D([[0, 0, 10, 10], [50, 50, 60, 60]], [0.8, 0.6], [0, 1])
    b = D([[2, 0, 12, 10]], [0.4], [0])
    f = wbf([a, b], iou_thr=0.55)
    order = np.argsort(f.labels)
    fb, fs, fl = f.boxes[order], f.scores[order], f.labels[order]
    # cluster of class 0: score-weighted coordinates, mean score * min(2, 2) / 2
    np.testing.assert_allclose(fb[0], [(0 * 0.8 + 2 * 0.4) / 1.2, 0, (10 * 0.8 + 12 * 0.4) / 1.2, 10], rtol=1e-6)
    assert fs[0] == pytest.approx(0.6)
    # class 1 box seen by one of two models: 0.6 * 1 / 2
    assert fs[1] == pytest.approx(0.3) and fl.tolist() == [0, 1]


def test_wbf_does_not_fuse_across_classes_and_handles_empty():
    a = D([[0, 0, 10, 10]], [0.9], [0])
    b = D([[0, 0, 10, 10]], [0.9], [1])
    assert len(wbf([a, b]).boxes) == 2
    e = wbf([Detections.empty(), Detections.empty()])
    assert e.boxes.shape == (0, 4)


def test_nms_returns_descending_score_order():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]], float)
    keep = nms(boxes, np.array([0.2, 0.9, 0.5]), np.array([0, 0, 0]), 0.5)
    assert keep.tolist() == [1, 2, 0]


def test_wbf_iou_equal_to_threshold_does_not_fuse():
    a = D([[0, 0, 10, 10]], [0.8], [0])
    b = D([[5, 0, 15, 10]], [0.6], [0])  # IoU exactly 1/3
    assert len(wbf([a, b], iou_thr=1 / 3).boxes) == 2
    assert len(wbf([a, b], iou_thr=0.3).boxes) == 1


def test_wbf_skip_thr_drops_low_boxes():
    a = D([[0, 0, 10, 10], [50, 50, 60, 60]], [0.8, 0.05], [0, 0])
    f = wbf([a], skip_thr=0.1)
    assert len(f.boxes) == 1 and f.scores[0] == pytest.approx(0.8)


def test_wbf_counts_distinct_models_not_boxes():
    a = D([[0, 0, 10, 10], [1, 0, 11, 10]], [0.8, 0.6], [0, 0])
    f = wbf([a, Detections.empty()], iou_thr=0.55)
    assert len(f.boxes) == 1
    assert f.scores[0] == pytest.approx((0.8 + 0.6) / 2 / 2)


def test_cross_class_suppress_keeps_highest_score_across_classes_only():
    from cvt.detect.fusion import cross_class_suppress

    b = np.array([[0, 0, 10, 10], [0, 0, 10, 9], [0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    s = np.array([0.5, 0.9, 0.7, 0.3], np.float32)
    c = np.array([5, 0, 0, 4])
    # box 1 (0.9, class 0) suppresses box 0 (other class); box 2 is same class -> kept (left to NMS)
    assert cross_class_suppress(b, s, c, 0.5).tolist() == [False, True, True, True]
    assert cross_class_suppress(b[:0], s[:0], c[:0], 0.5).tolist() == []
