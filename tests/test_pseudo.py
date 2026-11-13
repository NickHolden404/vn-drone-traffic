import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt.detect.base import Detections
from cvt.pseudo.disagreement import disagreement
from cvt.pseudo.label import load_pseudo, pseudo_label_frames, save_pseudo


def D(boxes, labels, scores=None):
    b = np.array(boxes, np.float32).reshape(-1, 4)
    return Detections(b, np.ones(len(b), np.float32) if scores is None else np.array(scores, np.float32),
                      np.array(labels, np.int64))


def test_disagreement_cases():
    a = D([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])
    assert disagreement(a, a) == 0.0
    assert disagreement(Detections.empty(), Detections.empty()) == 0.0
    assert disagreement(a, Detections.empty()) == 1.0
    assert disagreement(a, D([[100, 100, 110, 110]], [0])) == 1.0  # nothing matches
    # one exact match and one unmatched box: (1 + 0) / 2
    b = D([[0, 0, 10, 10]], [0])
    c = D([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 0])
    assert disagreement(b, c) == pytest.approx(0.5)
    shifted = D([[0, 0, 12.5, 10]], [0])  # IoU 0.8 -> (0 + 0.2) / 1
    assert disagreement(b, shifted) == pytest.approx(0.2)


def _df(rows):
    return pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score", "label"])


def test_pseudo_label_frames_thresholds_fusion_and_disagreement(tmp_path: Path):
    images = pd.DataFrame({"image_id": ["a", "b", "c"]})
    t1 = _df([("a", 0, 0, 10, 10, 0.9, 0), ("a", 50, 50, 60, 60, 0.9, 5), ("b", 0, 0, 10, 10, 0.2, 0)])
    t2 = _df([("a", 0, 0, 10, 10, 0.7, 0)])
    th = {c: 0.3 for c in range(6)}
    labels, frames = pseudo_label_frames([t1, t2], images, th, iou_thr=0.55)
    a = labels[labels.image_id == "a"].sort_values("label")
    # motorbike seen by both (0.8 kept), pedestrian by one teacher (0.45 kept)
    assert a.label.tolist() == [0, 5] and a.score.tolist() == pytest.approx([0.8, 0.45])
    assert "b" not in set(labels.image_id)  # below threshold everywhere
    f = frames.set_index("image_id")
    assert f.loc["a", "n_boxes"] == 2 and f.loc["a", "disagreement"] == pytest.approx(0.5)
    assert f.loc["c", "n_boxes"] == 0 and f.loc["c", "disagreement"] == 0.0
    save_pseudo(labels, frames, {"thresholds": th}, tmp_path / "run")
    l2, f2 = load_pseudo(tmp_path / "run")
    pd.testing.assert_frame_equal(l2.reset_index(drop=True), labels.reset_index(drop=True))
    assert len(f2) == 3


def test_pseudo_label_frames_caps_fused_boxes_per_frame():
    images = pd.DataFrame({"image_id": ["a"]})
    t = _df([("a", 0, 0, 10, 10, 0.9, 0), ("a", 50, 50, 60, 60, 0.6, 0)])
    th = {c: 0.0 for c in range(6)}
    labels, frames = pseudo_label_frames([t], images, th, max_det=1)
    assert len(labels) == 1 and labels.score.tolist() == pytest.approx([0.9])
    assert frames.n_boxes.tolist() == [1]


def test_disagreement_class_flip_and_greedy_highest_iou_first():
    a = D([[0, 0, 10, 10]], [0])
    assert disagreement(a, D([[0, 0, 10, 10]], [1])) == pytest.approx(1.0)  # same box, flipped
    # b0 overlaps a0 at IoU 0.6, b1 at IoU 0.9: greedy takes b1, b0 stays unmatched
    b = D([[0, 0, 10, 100 / 6], [0, 0, 10, 100 / 9]], [0, 0])
    assert disagreement(a, b) == pytest.approx((1 + (1 - 0.9)) / 2)


def test_pseudo_label_frames_per_class_and_infinite_thresholds():
    images = pd.DataFrame({"image_id": ["a"]})
    t = _df([("a", 0, 0, 10, 10, 0.5, 0), ("a", 20, 0, 30, 10, 0.5, 1),
             ("a", 40, 0, 50, 10, 0.99, 2)])
    th = {0: 0.4, 1: 0.6, 2: float("inf"), 3: 0.5, 4: 0.5, 5: 0.5}
    labels, frames = pseudo_label_frames([t], images, th)
    assert labels.label.tolist() == [0]
    assert frames.n_boxes.tolist() == [1] and frames.n_boxes_m0.tolist() == [1]


def test_pseudo_label_frames_caps_after_thresholds():
    images = pd.DataFrame({"image_id": ["a"]})
    # the 0.99 box is of a class with tau = inf: it must not use up the max_det slot
    t = _df([("a", 0, 0, 10, 10, 0.99, 2), ("a", 50, 0, 60, 10, 0.5, 0)])
    th = {c: 0.0 for c in range(6)} | {2: float("inf")}
    labels, _ = pseudo_label_frames([t], images, th, max_det=1)
    assert labels.label.tolist() == [0]


def test_pseudo_label_frames_member_thresholds_and_counts():
    images = pd.DataFrame({"image_id": ["a"]})
    t1 = _df([("a", 0, 0, 10, 10, 0.5, 0)])
    t2 = _df([("a", 0, 0, 10, 10, 0.9, 0)])
    th = {c: 0.3 for c in range(6)}
    labels, frames = pseudo_label_frames([t1, t2], images, th)
    assert list(frames.columns) == ["image_id", "n_boxes", "disagreement", "n_boxes_m0", "n_boxes_m1"]
    assert frames.disagreement.tolist() == [0.0]
    assert frames.n_boxes_m0.tolist() == [1] and frames.n_boxes_m1.tolist() == [1]
    # member 0's own threshold drops its only box: the members now disagree completely
    member_th = [{c: 0.6 for c in range(6)}, {c: 0.3 for c in range(6)}]
    l2, f2 = pseudo_label_frames([t1, t2], images, th, member_thresholds=member_th)
    assert f2.disagreement.tolist() == [1.0]
    assert f2.n_boxes_m0.tolist() == [0] and f2.n_boxes_m1.tolist() == [1]
    pd.testing.assert_frame_equal(l2, labels)  # fused labels use the fused thresholds
    with pytest.raises(ValueError, match="one per member"):
        pseudo_label_frames([t1, t2], images, th, member_thresholds=member_th[:1])


def test_save_pseudo_writes_meta_last_and_atomically(tmp_path: Path):
    images = pd.DataFrame({"image_id": ["a"]})
    labels, frames = pseudo_label_frames([_df([])], images, {c: 0.5 for c in range(6)})
    out = tmp_path / "run"
    out.mkdir()
    (out / "meta.json").write_text("{\"stale\": true}", encoding="utf-8")
    save_pseudo(labels, frames, {"manifest_hash": "abc"}, out)
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == {"manifest_hash": "abc"}
    assert sorted(p.name for p in out.iterdir()) == ["frames.parquet", "labels.parquet", "meta.json"]


def test_pseudo_label_frames_cross_class_suppression():
    images = pd.DataFrame({"image_id": ["a"]})
    # one scooter predicted as motorbike (0.8) and bicycle (0.4); a separate pedestrian far away
    t = _df([("a", 0, 0, 10, 20, 0.8, 0), ("a", 1, 1, 10, 20, 0.4, 4), ("a", 50, 50, 55, 55, 0.5, 5)])
    th = {c: 0.3 for c in range(6)}
    off, _ = pseudo_label_frames([t], images, th)
    on, frames = pseudo_label_frames([t], images, th, cross_class_iou=0.5)
    assert sorted(off.label) == [0, 4, 5]
    assert sorted(on.label) == [0, 5] and frames.n_boxes.tolist() == [2]
