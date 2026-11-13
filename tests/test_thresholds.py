import math

import numpy as np

from cvt.eval.ap import MatchTable, per_class_thresholds
from tests.test_ap import _tables


def test_per_class_thresholds_maximise_f1_and_fall_back():
    t = MatchTable(
        np.zeros(4, int),
        np.array([[1, 1, 0, 1]], bool),
        np.array([[0, 0, 1, 0]], bool),
        np.array([3.0]),
        np.array([0.9, 0.8, 0.7, 0.6]),
    )
    empty = MatchTable(np.zeros(0, int), np.zeros((1, 0), bool), np.zeros((1, 0), bool),
                       np.array([0.0]), np.zeros(0))
    # F1 by cut: 0.5, 0.8, 0.667, 0.857 -> best at the 4th detection (score 0.6)
    assert per_class_thresholds({(0, "all"): t, (1, "all"): empty}, [0, 1]) == {0: 0.6, 1: 0.5}


def test_build_tables_keeps_descending_scores():
    _, _, tables = _tables()
    for t in tables.values():
        assert t.scores is not None and len(t.scores) == t.tp.shape[1]
        assert np.all(np.diff(t.scores) <= 0)


def test_per_class_thresholds_without_true_positives_is_inf():
    t = MatchTable(np.zeros(3, int), np.zeros((1, 3), bool), np.ones((1, 3), bool),
                   np.array([2.0]), np.array([0.9, 0.8, 0.7]))
    assert per_class_thresholds({(0, "all"): t}, [0]) == {0: math.inf}


def test_per_class_thresholds_uses_iou_index():
    t = MatchTable(
        np.zeros(2, int),
        np.array([[1, 0], [0, 1]], bool),
        np.array([[0, 1], [1, 0]], bool),
        np.array([1.0]),
        np.array([0.9, 0.4]),
    )
    assert per_class_thresholds({(0, "all"): t}, [0], iou_index=0) == {0: 0.9}
    assert per_class_thresholds({(0, "all"): t}, [0], iou_index=1) == {0: 0.4}
