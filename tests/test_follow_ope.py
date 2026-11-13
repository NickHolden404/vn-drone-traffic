import numpy as np
import pandas as pd
import pytest

from cvt.data.schema import CLASS_ID
from cvt.follow.ope import aggregate, lock_targets, ope_scores


def _track(seq, tid, n, label="car", start=0, x0=100.0):
    f = np.arange(start, start + n)
    return pd.DataFrame({
        "seq_id": seq, "frame": f, "image_id": [f"{seq}_{i}" for i in f], "track_id": tid,
        "x1": x0 + 0.0 * f, "y1": 100.0, "x2": x0 + 40.0, "y2": 140.0, "label": CLASS_ID[label],
    })


def test_lock_targets_min_seconds():
    gt = pd.concat([
        _track("s", 1, 149), _track("s", 2, 150), _track("s", 3, 400, label="pedestrian"),
    ], ignore_index=True)
    out = lock_targets(gt, fps=30.0)
    assert out["track_id"].tolist() == [2]
    assert out.iloc[0]["first"] == 0 and out.iloc[0]["last"] == 149
    assert set(out.columns) >= {"seq_id", "track_id", "label", "first", "last"}


def _pred(gt):
    return gt[["frame", "x1", "y1", "x2", "y2"]].copy()


def test_ope_perfect_and_lost():
    gt = _track("s", 1, 50)
    others = gt.iloc[0:0]
    r = ope_scores(_pred(gt), gt, others)
    assert abs(r["success_auc"] - 20 / 21) < 1e-9
    assert r["precision_20"] == 1.0 and r["lock_kept"] == 1.0
    assert r["n_frames"] == 50
    assert r["norm_precision_auc"] == 1.0 and r["wrong_frames"] == 0
    lost = _pred(gt)
    lost[["x1", "y1", "x2", "y2"]] = np.nan
    z = ope_scores(lost, gt, others)
    assert z["success_auc"] == 0 and z["precision_20"] == 0 and z["norm_precision_auc"] == 0
    assert z["lock_kept"] == 0 and z["wrong_frames"] == 0 and z["wrong_switches"] == 0


def test_ope_skips_frames_without_gt():
    gt = _track("s", 1, 50)
    gt = gt[~gt["frame"].between(10, 19)]
    r = ope_scores(_pred(_track("s", 1, 50)), gt, gt.iloc[0:0])
    assert r["n_frames"] == 40


def test_wrong_switch_counting():
    gt = _track("s", 1, 60)
    other = _track("s", 2, 60, x0=300.0)
    pred = _pred(gt)
    m = pred["frame"].between(30, 39)
    pred.loc[m, ["x1", "x2"]] = [300.0, 340.0]
    r = ope_scores(pred, gt, other)
    assert r["wrong_switches"] == 1 and r["wrong_frames"] == 10
    assert abs(r["lock_kept"] - 50 / 60) < 1e-9


def _row(seq, v, n=100):
    return {"seq_id": seq, "track_id": 1, "n_frames": n, "success_auc": v, "precision_20": v,
            "norm_precision_auc": v, "lock_kept": v, "wrong_frames": 0, "wrong_switches": 0}


def test_aggregate_bootstrap_by_sequence():
    same = pd.DataFrame([_row("a", 0.5), _row("a", 0.5), _row("b", 0.5), _row("c", 0.5)])
    assert aggregate(same, n_boot=50).keys() >= {"wrong_frame_rate", "wrong_switches_per_target"}
    assert "wrong_frames" not in aggregate(same, n_boot=50)
    m, lo, hi = aggregate(same, n_boot=50)["success_auc"]
    assert m == 0.5 and lo == hi == 0.5
    diff = pd.DataFrame([_row("a", 0.0), _row("b", 1.0), _row("c", 0.5), _row("d", 0.5)])
    m, lo, hi = aggregate(diff, n_boot=200)["success_auc"]
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi
    # many targets in one sequence still resample as one cluster
    one = pd.DataFrame([_row("a", 0.2)] * 5 + [_row("b", 0.8)])
    _, lo, hi = aggregate(one, n_boot=200)["success_auc"]
    assert lo <= 0.2 and hi >= 0.8


def test_aggregate_wrong_target_rates():
    a = {**_row("a", 0.5, n=100), "wrong_frames": 10, "wrong_switches": 2}
    b = {**_row("b", 0.5, n=300), "wrong_frames": 30, "wrong_switches": 0}
    r = aggregate(pd.DataFrame([a, b]), n_boot=50)
    assert abs(r["wrong_frame_rate"][0] - 40 / 400) < 1e-12
    assert abs(r["wrong_switches_per_target"][0] - 1.0) < 1e-12
    for k in ("wrong_frame_rate", "wrong_switches_per_target"):
        assert r[k][1] <= r[k][0] <= r[k][2] or np.isnan(r[k][1])


def test_lock_targets_label_change_is_one_target_with_majority_label():
    car = _track("s", 5, 100)  # frames 0..99 car
    truck = _track("s", 5, 60, label="truck", start=100)  # frames 100..159 truck
    walk = _track("s", 5, 300, label="pedestrian", start=160)  # non-vehicle rows are ignored
    out = lock_targets(pd.concat([car, truck, walk], ignore_index=True), fps=30.0)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["track_id"] == 5 and r["label"] == CLASS_ID["car"]
    assert r["first"] == 0 and r["last"] == 159  # the span covers both labels' vehicle rows
    # neither label alone lasts 5 s at 30 fps, together they do
    short = pd.concat([_track("s", 6, 80), _track("s", 6, 80, label="truck", start=80)])
    out = lock_targets(short, fps=30.0)
    assert out["track_id"].tolist() == [6] and out.iloc[0]["last"] == 159
    # a tie goes to the smaller label id
    lo, hi = sorted([CLASS_ID["car"], CLASS_ID["truck"]])
    assert out.iloc[0]["label"] == lo and lo != hi


def test_aggregate_raises_on_nan_metric_naming_the_target():
    bad = {**_row("b", 0.5), "track_id": 7, "lock_kept": float("nan")}
    with pytest.raises(ValueError, match=r"lock_kept.*b.*7"):
        aggregate(pd.DataFrame([_row("a", 0.5), bad]), n_boot=10)
    bad = {**_row("b", 0.5), "track_id": 7, "wrong_frames": float("nan")}
    with pytest.raises(ValueError, match=r"wrong_frames.*b.*7"):
        aggregate(pd.DataFrame([_row("a", 0.5), bad]), n_boot=10)
