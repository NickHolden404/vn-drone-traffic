from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt.analytics.pipeline import ClipResult, anomaly_rows, clip_summary, load_cfg, od_eval
from cvt.eval.results import RESULT_COLS
from tests.test_analytics_od import _scene

CFG_A = load_cfg(Path("configs/analytics.yaml"))
BASE = {c: "x" for c in RESULT_COLS}
N_LABELS = 40


def test_od_eval_counts_against_gt_arms():
    gt = _scene()  # 6 W->E cars, 5 N->E bikes
    pred = gt[~gt.track_id.isin([0, 10])].copy()  # two vehicles missed entirely
    half = pred.track_id == 1  # one track broken in two mid-frame
    pred.loc[half & (pred.frame > pred[half].frame.median()), "track_id"] = 900
    sizes = pd.DataFrame({"frame": range(0, 201), "width": 1000, "height": 500})
    r = od_eval(gt, pred, sizes, CFG_A)
    assert r["arms"] == 3 and r["complete_gt"] == 11 and r["complete_pred"] == 8
    assert r["cells"]["gt"].sum() == 11 and r["cells"]["pred"].sum() == 8


def _result(clip: str, scores: np.ndarray) -> ClipResult:
    kin = pd.DataFrame({"track_id": [1, 1, 2, 4, 5], "frame": [0, 1, 0, 0, 0],
                        "label": [1, 1, 1, 5, 0],         # car, car, car, pedestrian, motorbike
                        "speed_kmh": [10.0, 30.0, 0.5, 6.0, 40.0]})
    ep = pd.DataFrame({"track_id": [1, 2, 3], "label": [1, 1, 0]})
    ev = pd.DataFrame({"kind": ["stopped", "wrong_way", "wrong_way"]})
    return ClipResult(
        clip, {"n_frames": 200, "fps": 20.0},
        {"camera": "FC7303", "height_med": 50.0, "mpp_telemetry": 0.05,
         "mpp_car": 0.06, "source": "telemetry"},
        kin, scores, scores, {"stopped": scores, "wrong": scores, "both": scores}, ev, ep,
        np.zeros(0), pd.DataFrame(), {"tracks": 3, "complete": 2, "complete_frac": 2 / 3,
                                      "assigned": 2, "arms": 3},
        {}, {"n_parked": 1, "median_kmh": 0.2, "p95_kmh": 0.9}, 0)


@pytest.fixture
def labels2():
    a = (np.arange(N_LABELS) % 4 == 0).astype(np.int8)
    return {"a": a, "b": a.copy()}


@pytest.fixture
def results2(labels2):
    s = np.repeat(labels2["a"], 5).astype(float)
    return [_result("a", s), _result("b", 1.0 - s)]


def test_anomaly_rows_slices_and_notes(results2, labels2):
    rows = anomaly_rows(results2, {**CFG_A, "held_out": ["a"]}, labels2, BASE)
    df = pd.DataFrame(rows)
    assert set(df.columns) >= set(RESULT_COLS) and (df.metric == "anomaly/auc").all()
    both = df[df.model.str.endswith("+events-both")].set_index("slice")
    assert both.loc["held-out", "value"] == 1.0 and both.loc["trained", "value"] == 0.0
    assert both.loc["all", "n_scenes"] == 2 and "not comparable" in both.loc["all", "notes"]
    assert both.loc["all", "n_images"] == 2 * N_LABELS
    assert set(df.model.str.rsplit("-", n=1).str[-1]) == {"stopped", "wrong", "both"}


def test_one_class_clip_has_nan_auc_but_pools(results2, labels2):
    labels2 = {**labels2, "a": np.ones_like(labels2["a"])}
    s = clip_summary(results2[0], labels2["a"], {**CFG_A, "held_out": ["a"]})
    assert np.isnan(s["auc_both"]) and s["abnormal_frac"] == 1.0 and s["group"] == "held-out"
    rows = pd.DataFrame(anomaly_rows(results2, {**CFG_A, "held_out": ["a"]}, labels2, BASE))
    assert rows[(rows.slice == "all")].value.notna().all()


def test_clip_summary_columns(results2, labels2):
    s = clip_summary(results2[1], labels2["b"], {**CFG_A, "held_out": ["a"]})
    assert s["group"] == "trained" and s["duration_s"] == 10.0 and s["tracks"] == 3
    assert s["vehicles_car"] == 2 and s["vehicles_motorbike"] == 1 and s["vehicles_bus"] == 0
    assert s["stopped_events"] == 1 and s["wrong_way_events"] == 2
    assert s["speed_median_kmh"] == 30.0 and s["auc_both"] == 0.0   # the pedestrian is left out
    assert s["speed_median_kmh_car"] == 20.0 and s["speed_p85_kmh_motorbike"] == 40.0
    assert np.isnan(s["speed_median_kmh_truck"]) and np.isnan(s["speed_p85_kmh_bus"])
    assert s["noise_p95_kmh"] == 0.9 and s["scale_source"] == "telemetry"


# --- od_rows over a fake T-MOT set (frames numbered from 1, as in the manifest) ----------------

SEQS = ("seqA", "seqB", "seqC")


def _od_set(monkeypatch, scenes):
    """Three identical sequences of `_scene()` shifted to frames 1..201; the predictions equal the
    GT except that seqB misses car track 0 and seqA has an extra 20-frame (< 1 s) W->E stub."""
    from cvt.analytics import pipeline as pl
    from tests.test_analytics_od import _line

    imgs, gts, preds = [], [], []
    for q in SEQS:
        ids = [f"T-MOT/{q}/{f:07d}" for f in range(1, 202)]
        imgs += [{"image_id": i, "path": "x", "width": 1000, "height": 500, "source": "s",
                  "scene_id": f"visdrone/{q}", "split": "test", "set_name": "T-MOT"} for i in ids]
        g = _scene().assign(seq_id=q)
        g["frame"] += 1
        g = g[g.frame <= 201]
        gts.append(g.assign(image_id=[f"T-MOT/{q}/{f:07d}" for f in g.frame]))
        p = g[g.track_id != 0] if q == "seqB" else g
        if q == "seqA":
            p = pd.concat([p, _line(77, (10, 250), (990, 250), 100, n=20).assign(seq_id=q)])
        preds.append(p)
    images = pd.DataFrame(imgs)
    gt = pd.concat(gts, ignore_index=True)
    tracks = pd.concat(preds, ignore_index=True).assign(score=0.9)
    boxes = pd.DataFrame({"image_id": ["T-MOT/seqA/0000006"], "x1": [0.0], "y1": [0.0],
                          "x2": [5.0], "y2": [5.0], "label": [1], "is_ignore": [True]})
    cols = ["seq_id", "frame", "image_id", "track_id", "x1", "y1", "x2", "y2", "label"]
    monkeypatch.setattr(pl, "load_checked_manifest", lambda *a: (images, boxes, "mh"))
    monkeypatch.setattr(pl, "load_mot", lambda d: (gt[cols], None))
    monkeypatch.setattr(pl, "load_tracks", lambda *a: (tracks, {"detections_cache_tag": "tag"}))
    monkeypatch.setattr(pl, "load_scenes",
                        lambda d, sets: [{"seq_id": q, "set_name": "T-MOT"} for q in scenes])
    return pl


def _od_rows(pl):
    return pl.od_rows({**CFG_A, "n_boot": 200}, ["T-MOT"], models_cfg={}, manifest_dir=Path("."),
                      hash_cache=Path("."), out_root=Path("."), base=BASE, log=lambda m: None)


def test_od_rows_hand_computed(monkeypatch):
    pl = _od_set(monkeypatch, ["seqA", "seqB"])               # seqC has no scene: left out
    df = pd.DataFrame(_od_rows(pl))
    assert set(df.columns) >= set(RESULT_COLS)
    df = df.set_index("metric")
    # cells (W->E car, N->E motorbike): seqA gt 6, 5 / pred 6, 5 (the stub is < 1 s);
    # seqB gt 6, 5 / pred 5, 5 -> |diff| sum 1 over 22 GT, mean over 4 cells 0.25
    assert df.loc["od/count_pct_err", "value"] == pytest.approx(1 / 22)
    assert df.loc["od/count_MAE", "value"] == pytest.approx(0.25)
    assert df.loc["od/complete_ratio", "value"] == pytest.approx(21 / 22)
    # bootstrap over 2 sequences: {A, A} has no error, {B, B} twice seqB's
    assert df.loc["od/count_pct_err", ["ci_lo", "ci_hi"]].tolist() == pytest.approx([0, 2 / 22])
    assert df.loc["od/count_MAE", ["ci_lo", "ci_hi"]].tolist() == pytest.approx([0, 0.5])
    assert np.isnan(df.loc["od/complete_ratio", "ci_lo"])
    assert (df.notes == "2 sequences, 6 arms, 22 complete GT tracks").all()
    assert (df.n_scenes == 2).all() and (df.n_images == 2 * 201).all()


def test_od_rows_rejects_a_scene_missing_from_its_set(monkeypatch):
    pl = _od_set(monkeypatch, ["seqA", "seqZ"])
    with pytest.raises(ValueError, match="seqZ"):
        _od_rows(pl)


def test_stacked_errors_keep_sequences_apart_and_boot_needs_two():
    from cvt.analytics.pipeline import _boot_od, _stacked_errors

    idx = pd.MultiIndex.from_tuples([(1, 0, 1)], names=["label", "origin", "dest"])
    a = pd.DataFrame({"gt": [1], "pred": [0]}, index=idx)
    b = pd.DataFrame({"gt": [0], "pred": [1]}, index=idx)
    e = _stacked_errors([a, b])                               # same cell, different sequences
    assert e["count_pct_err"] == 2.0 and e["count_MAE"] == 1.0
    assert all(np.isnan(v).all() for v in _boot_od([a], 50, 0).values())
