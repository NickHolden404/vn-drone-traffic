"""Tracking pipeline on a synthetic MOT manifest with fake cached detections (no GPU, no model)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import cvt.detect.ultralytics_det as ud
import cvt.track.pipeline as pl
from cvt.cli import app
from cvt.data.manifest import load_manifest, save_manifest
from cvt.eval.predictions import PRED_COLS
from cvt.eval.report import tracking_table
from cvt.eval.results import RESULT_COLS, append_results
from cvt.eval.run import set_hash_of
from cvt.track.count import Line
from cvt.track.pipeline import (
    compare_tracking,
    count_scene,
    evaluate_tracking,
    run_tracking,
    tracker_grid,
    tune_tracker,
)
from cvt.track.run import TRACK_COLS
from tests.helpers import write_noise_jpg

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "trackers"
W, H = 128, 96
FRAMES = [1, 2, 3, 5, 6]  # frame 4 missing on disk: numbering comes from the file, not position
TAG = "abcdef012345"
SEQS = {
    "MOT-tune": [("visdrone_mot", "val", "uav0000001_00000_v")],
    "T-MOT": [("visdrone_mot", "test", "uav0000002_00000_v"),
              ("visdrone_mot", "test", "uav0000003_00000_v")],
    "T-UAVDT": [("uavdt", "test", "M0101"), ("uavdt", "test", "M0202")],
}
OBJECTS = {  # static objects: Kalman output equals the detection, so perfect dets give HOTA 1
    "visdrone_mot": [(1, 1, (10, 10, 40, 30)), (2, 5, (70, 40, 80, 60))],
    "uavdt": [(1, 1, (10, 10, 40, 30)), (2, 3, (50, 40, 90, 70))],
}
IGNORE = (100.0, 70.0, 127.0, 95.0)


def _image_id(source: str, split: str, seq: str, f: int) -> str:
    return f"{source}/{split}/{seq}/{f:07d}" if source == "visdrone_mot" else f"{source}/{seq}/{f:06d}"


def _manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """MOT-tune (1 seq), T-MOT (2 seqs, car+pedestrian), T-UAVDT (2 seqs, car+truck), each
    frame with one ignore region; a fake `det` model whose detections are cached = GT."""

    def no_model(*a, **k):
        raise AssertionError("an Ultralytics model must never be constructed here")

    monkeypatch.setattr(ud.UltralyticsDetector, "__init__", no_model)
    imgs, boxes, gt = [], [], []
    seed = 0
    for set_name, seqs in SEQS.items():
        for source, split, seq in seqs:
            for f in FRAMES:
                iid = _image_id(source, split, seq, f)
                path = tmp_path / "img" / seq / f"{f:07d}.jpg"
                write_noise_jpg(path, seed, (W, H))
                seed += 1
                imgs.append((iid, str(path), W, H, source, f"{source}/{seq}", split, set_name))
                for tid, lab, b in OBJECTS[source]:
                    boxes.append((iid, *map(float, b), lab, False))
                    gt.append((seq, f, iid, tid, *map(float, b), lab))
                boxes.append((iid, *IGNORE, -1, True))
    images = pd.DataFrame(imgs, columns=["image_id", "path", "width", "height", "source",
                                         "scene_id", "split", "set_name"])
    boxes = pd.DataFrame(boxes, columns=["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"])
    mot_gt = pd.DataFrame(gt, columns=["seq_id", "frame", "image_id", "track_id", "x1", "y1",
                                       "x2", "y2", "label"])
    attrs = pd.DataFrame(
        [("M0101", True, False, False, True, False, False, True, False, False),
         ("M0202", False, True, False, True, False, False, False, True, False)],
        columns=["seq_id", "daylight", "night", "fog", "low_alt", "medium_alt", "high_alt",
                 "front_view", "side_view", "bird_view"])
    manifest = tmp_path / "manifest"
    save_manifest(images, boxes, manifest, mot_gt, attrs)
    p = {"manifest": manifest, "hash": tmp_path / "ph.parquet", "preds": tmp_path / "preds",
         "tracks": tmp_path / "tracks", "gmc": tmp_path / "gmc", "results": tmp_path / "res.csv",
         "models": tmp_path / "models.yaml", "reports": tmp_path / "reports"}
    p["models"].write_text(yaml.safe_dump(
        {"det": {"arch": "yolo", "weights": str(tmp_path / "absent.pt"), "names_map": "cvt"}}))
    for set_name in SEQS:
        write_det_cache(p, set_name)
    return p


def write_det_cache(p: dict, set_name: str, tag: str = TAG, model: str = "det") -> Path:
    images, boxes, _ = load_manifest(p["manifest"])
    im = images[images.set_name == set_name].reset_index(drop=True)
    obj = boxes[~boxes.is_ignore & boxes.image_id.isin(im.image_id)]
    dt = obj[["image_id", "x1", "y1", "x2", "y2"]].assign(score=0.9, label=obj.label)[PRED_COLS]
    path = Path(p["preds"]) / f"{model}_{tag}_{set_name}_{set_hash_of(im)}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dt.to_parquet(path, index=False)
    return path


def _kw(p: dict) -> dict:
    return dict(models_cfg=yaml.safe_load(p["models"].read_text()), manifest_dir=p["manifest"],
                hash_cache=p["hash"], pred_cache=p["preds"], out_root=p["tracks"],
                warp_cache=p["gmc"])


def _common(p: dict, *skip: str) -> list[str]:
    opts = {"--models": p["models"], "--manifest": p["manifest"], "--hash-cache": p["hash"],
            "--pred-cache": p["preds"], "--out-root": p["tracks"]}
    return [x for k, v in opts.items() if k not in skip for x in (k, str(v))]


def _cli(*args: str):
    r = CliRunner().invoke(app, list(args))
    assert r.exit_code == 0, r.output
    return r


# --- end to end ----------------------------------------------------------------------------

def test_run_and_eval_end_to_end_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = _manifest(tmp_path, monkeypatch)
    out = run_tracking("det", CFG_DIR / "bytetrack.yaml", "T-MOT", **_kw(p))
    assert out == p["tracks"] / "det+bytetrack" / "T-MOT.parquet"
    tracks = pd.read_parquet(out)
    assert list(tracks.columns) == TRACK_COLS and len(tracks) == 2 * 2 * len(FRAMES)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["detections_cache_tag"] == TAG
    assert {"tracker_cfg_hash", "manifest_hash"} <= set(meta)

    tracker = str(CFG_DIR / "bytetrack.yaml")
    _cli("track", "run", "--model", "det", "--tracker", tracker, "--sets", "T-MOT,T-UAVDT",
         "--warp-cache", str(p["gmc"]), *_common(p))
    _cli("track", "eval", "--model", "det", "--tracker", tracker, "--sets", "T-MOT,T-UAVDT",
         "--n-boot", "50", "--results", str(p["results"]), *_common(p))
    df = pd.read_csv(p["results"])
    assert list(df.columns) == RESULT_COLS
    assert (df.model == "det+bytetrack").all() and (df.imgsz == 1280).all()
    head = df[(df["slice"] == "all") & (df.metric == "HOTA")].set_index("eval_set")
    assert head.loc["T-MOT", "value"] == pytest.approx(1.0)
    assert head.loc["T-UAVDT", "value"] == pytest.approx(1.0)
    assert head.ci_lo.notna().all() and head.ci_hi.notna().all()
    assert head.loc["T-MOT", "n_images"] == 2 * len(FRAMES) and head.loc["T-MOT", "n_scenes"] == 2
    allm = df[df["slice"] == "all"]
    assert {"HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDsw"} <= set(allm.metric)
    mot = set(allm[allm.eval_set == "T-MOT"].metric)
    uav = set(allm[allm.eval_set == "T-UAVDT"].metric)
    assert {"HOTA/car", "HOTA/pedestrian"} <= mot and "HOTA/bus" not in mot
    assert {"HOTA/car", "HOTA/truck"} <= uav and not {"HOTA/bus", "HOTA/pedestrian"} & uav
    slices = df[df["slice"] != "all"]
    assert set(slices.eval_set) == {"T-UAVDT"}
    assert set(slices["slice"]) == {"daylight", "night", "low_alt"}  # empty slices give no rows
    night = slices[(slices["slice"] == "night") & (slices.metric == "HOTA")].iloc[0]
    assert night.value == pytest.approx(1.0) and math.isnan(night.ci_lo)  # one sequence


def test_missing_or_ambiguous_detection_cache_raises(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    for f in Path(p["preds"]).glob("*_T-MOT_*.parquet"):
        f.unlink()
    with pytest.raises(RuntimeError, match=r"scripts\\track\.ps1 -Step detect"):
        run_tracking("det", CFG_DIR / "bytetrack.yaml", "T-MOT", **_kw(p))
    write_det_cache(p, "T-MOT")
    write_det_cache(p, "T-MOT", tag="0123456789ab")
    with pytest.raises(RuntimeError, match="several cached"):
        run_tracking("det", CFG_DIR / "bytetrack.yaml", "T-MOT", **_kw(p))


def test_botsort_warps_cached_per_sequence(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    calls = []

    def fake_warps(paths, method, downscale=2):
        calls.append((len(paths), method))
        return np.tile(np.eye(2, 3), (len(paths), 1, 1))

    monkeypatch.setattr(pl, "gmc_warps", fake_warps)
    for _ in range(2):
        out = run_tracking("det", CFG_DIR / "botsort.yaml", "T-MOT", **_kw(p))
    assert calls == [(len(FRAMES), "sparseOptFlow")] * 2  # once per sequence, then cached
    assert len(list(Path(p["gmc"]).glob("*.npy"))) == 2
    assert out.parent.name == "det+botsort"


def test_uavdt_drops_other_classes_before_tracking(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    path = write_det_cache(p, "T-UAVDT")
    dt = pd.read_parquet(path)
    ped = dt.drop_duplicates("image_id").assign(x1=95.0, y1=5.0, x2=105.0, y2=25.0, label=5)
    pd.concat([dt, ped], ignore_index=True).to_parquet(path, index=False)
    tracks = pd.read_parquet(run_tracking("det", CFG_DIR / "bytetrack.yaml", "T-UAVDT", **_kw(p)))
    assert set(tracks.label) == {1, 3}  # rule 3: car/truck/bus only


# --- metrics -------------------------------------------------------------------------------

def _tracks(rows, seq="s") -> pd.DataFrame:
    """rows: (frame, track_id, x1, y1, x2, y2, label)."""
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "label"])
    return df.assign(seq_id=seq)


def _gt(rows, seq="s") -> pd.DataFrame:
    return _tracks(rows, seq)


def _ignore(rows, seq="s") -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2"]).assign(seq_id=seq)


def _value(rows, metric, slice_="all"):
    hit = [r for r in rows if r["metric"] == metric and r["slice"] == slice_]
    assert len(hit) == 1, (metric, slice_)
    return hit[0]


def test_tracker_boxes_in_ignore_dropped():
    gt = _gt([(f, 1, 10, 10, 30, 30, 1) for f in (1, 2, 3)])
    fp = [(f, 2, 100, 100, 120, 120, 1) for f in (1, 2, 3)]  # area 400
    tracks = _tracks([(f, 1, 10, 10, 30, 30, 1) for f in (1, 2, 3)] + fp)
    half = _ignore([(f, 110, 100, 140, 140) for f in (1, 2, 3)])  # exactly 50% inside -> dropped
    rows = evaluate_tracking(tracks, gt, half, [1], None, n_boot=20)
    assert _value(rows, "MOTA")["value"] == pytest.approx(1.0)
    assert _value(rows, "HOTA")["value"] == pytest.approx(1.0)
    less = _ignore([(f, 112, 100, 140, 140) for f in (1, 2, 3)])  # 40% inside -> kept
    rows = evaluate_tracking(tracks, gt, less, [1], None, n_boot=20)
    assert _value(rows, "MOTA")["value"] == pytest.approx(0.0)
    other_frame = _ignore([(9, 100, 100, 140, 140)])  # region of another frame never applies
    rows = evaluate_tracking(tracks, gt, other_frame, [1], None, n_boot=20)
    assert _value(rows, "MOTA")["value"] == pytest.approx(0.0)


def _two_seq_perfect():
    rows = [(f, 1, 10, 10, 30, 30, 1) for f in (1, 2, 3)]
    gt = pd.concat([_gt(rows, "a"), _gt(rows, "b")], ignore_index=True)
    return gt, gt.copy()


def test_slice_with_few_sequences_has_no_ci():
    gt, tracks = _two_seq_perfect()
    attrs = pd.DataFrame({"seq_id": ["a", "b"], "daylight": [True, False], "night": [False, True],
                          "fog": [False, False], "low_alt": [True, True],
                          "medium_alt": [False, False], "high_alt": [False, False]})
    rows = evaluate_tracking(tracks, gt, _ignore([]), [1], attrs, n_boot=50)
    night = _value(rows, "HOTA", "night")
    assert night["value"] == pytest.approx(1.0)
    assert math.isnan(night["ci_lo"]) and math.isnan(night["ci_hi"])
    low = _value(rows, "HOTA", "low_alt")
    assert low["ci_lo"] == pytest.approx(1.0) and low["ci_hi"] == pytest.approx(1.0)
    assert not [r for r in rows if r["slice"] == "fog"]
    assert _value(rows, "HOTA")["n_scenes"] == 2 and night["n_scenes"] == 1


def test_bootstrap_resamples_whole_sequences():
    # sequence a tracked perfectly, b missed: resamples range from all-a (1) to all-b (0)
    gt, _ = _two_seq_perfect()
    tracks = gt[gt.seq_id == "a"]
    rows = evaluate_tracking(tracks, gt, _ignore([]), [1], None, n_boot=400)
    expect = {"HOTA": math.sqrt(0.5), "MOTA": 0.5, "IDF1": 2 / 3}  # DetA 0.5, AssA 1
    for m, v in expect.items():
        r = _value(rows, m)
        assert r["value"] == pytest.approx(v)
        assert r["ci_lo"] == pytest.approx(0.0) and r["ci_hi"] == pytest.approx(1.0)


def test_bootstrap_matches_explicit_resampling():
    """CI from scaled SeqStats == CI from evaluating the resampled sequences as real copies."""
    from cvt.eval.bootstrap import _interval, _resamples

    box = (10, 10, 30, 30)
    gt = pd.concat([_gt([(f, 1, *box, 1) for f in (1, 2, 3, 4)], s) for s in "abc"])
    tracks = pd.concat([
        _tracks([(f, 1, *box, 1) for f in (1, 2, 3, 4)], "a"),  # perfect
        _tracks([(f, 1, *box, 1) for f in (1, 3)], "b"),  # misses frames 2 and 4
        _tracks([(f, 1 if f < 3 else 2, *box, 1) for f in (1, 2, 3, 4)], "c"),  # id switch
    ])
    rows = evaluate_tracking(tracks, gt, _ignore([]), [1], None, n_boot=60, seed=3)
    for metric in ("HOTA", "MOTA", "IDF1"):
        vals = []
        for w in _resamples(np.array(["a", "b", "c"]), 60, 3):
            gts, trs, names = [], [], []
            for seq, k in zip("abc", w, strict=True):
                for j in range(int(k)):
                    names.append(f"{seq}{j}")
                    gts.append(gt[gt.seq_id == seq].assign(seq_id=names[-1]))
                    trs.append(tracks[tracks.seq_id == seq].assign(seq_id=names[-1]))
            r = evaluate_tracking(pd.concat(trs), pd.concat(gts), _ignore([]), [1], None,
                                  n_boot=0, sequences=names)
            vals.append(_value(r, metric)["value"])
        lo, hi = _interval(vals, 0.05, 0.95)
        got = _value(rows, metric)
        assert (got["ci_lo"], got["ci_hi"]) == pytest.approx((lo, hi)), metric
        assert lo < hi


def test_class_combined_sums_per_class_stats():
    # car perfect, bus missed entirely: combined is detection-weighted, per-class rows separate
    gt = _gt([(f, 1, 10, 10, 30, 30, 1) for f in (1, 2, 3)]
             + [(f, 2, 50, 50, 70, 70, 2) for f in (1, 2, 3)])
    tracks = _tracks([(f, 1, 10, 10, 30, 30, 1) for f in (1, 2, 3)])
    rows = evaluate_tracking(tracks, gt, _ignore([]), [1, 2, 3], None, n_boot=10)
    assert _value(rows, "HOTA/car")["value"] == pytest.approx(1.0)
    assert _value(rows, "HOTA/bus")["value"] == pytest.approx(0.0)
    assert not [r for r in rows if r["metric"] == "HOTA/truck"]  # no truck GT
    assert _value(rows, "MOTA")["value"] == pytest.approx(0.5)


# --- tuning --------------------------------------------------------------------------------

def test_default_grid_is_rule6():
    for kind in ("bytetrack", "botsort"):
        grid = tracker_grid(kind)
        assert len(grid) == 54
        assert grid[0] == {"track_high_thresh": 0.2, "new_track_thresh": 0.3,
                           "match_thresh": 0.7, "track_buffer": 30}
        assert grid[1]["track_buffer"] == 60 and grid[-1]["track_high_thresh"] == 0.4
    assert {g["match_thresh"] for g in grid} == {0.7, 0.8, 0.9}


def test_grid_3b_is_the_predeclared_192_points():
    for kind in ("bytetrack", "botsort"):
        grid = tracker_grid(kind, "3b")
        assert len(grid) == 192
        assert grid[0] == {"track_high_thresh": 0.2, "new_track_thresh": 0.4,
                           "match_thresh": 0.7, "track_buffer": 30}
        assert [g["track_buffer"] for g in grid[:4]] == [30, 60, 90, 30]
        assert grid[3]["match_thresh"] == 0.8
        assert grid[-1] == {"track_high_thresh": 0.5, "new_track_thresh": 0.7,
                            "match_thresh": 0.95, "track_buffer": 90}
        assert tracker_grid(kind, "plan3") == tracker_grid(kind)
    assert {g["new_track_thresh"] for g in grid} == {0.4, 0.5, 0.6, 0.7}
    with pytest.raises(ValueError, match="grid"):
        tracker_grid("bytetrack", "plan4")


def test_tuned_names_and_grid_report_names():
    from cvt.track.run import load_tracker_cfg

    base = load_tracker_cfg(CFG_DIR / "botsort.yaml")
    point = tracker_grid("botsort", "3b")[7]
    assert pl.tuned_cfg(base, point)["name"] == "botsort-tuned"
    cfg = pl.tuned_cfg(base, point, suffix="tuned3b")
    assert cfg["name"] == "botsort-tuned3b" and cfg["tracker_type"] == "botsort"
    assert cfg["gmc_method"] == "sparseOptFlow" and cfg["track_low_thresh"] == 0.1
    assert {k: cfg[k] for k in point} == point
    assert pl.grid_report_name("bytetrack") == "tracker_grid_bytetrack.csv"
    assert pl.grid_report_name("bytetrack", "plan3", "m") == "tracker_grid_bytetrack.csv"
    assert (pl.grid_report_name("botsort", "3b", "m@1536-fp16")
            == "tracker_grid_botsort_3b_m@1536-fp16.csv")


def test_tune_selects_best_by_combined_hota(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    base = {"track_high_thresh": 0.3, "match_thresh": 0.8}
    grid = [{**base, "new_track_thresh": 0.95, "track_buffer": 30},  # no track ever starts
            {**base, "new_track_thresh": 0.3, "track_buffer": 30},
            {**base, "new_track_thresh": 0.3, "track_buffer": 60}]  # tie -> the earlier wins
    kw = {k: v for k, v in _kw(p).items() if k != "out_root"}
    best, table = tune_tracker("det", "bytetrack", "MOT-tune", grid=grid,
                               reports_dir=p["reports"], **kw)
    assert best["new_track_thresh"] == 0.3 and best["track_buffer"] == 30
    assert best["name"] == "bytetrack-tuned" and best["tracker_type"] == "bytetrack"
    assert best["track_low_thresh"] == 0.1 and best["fuse_score"] is True
    assert best["frame_rate"] == 30
    assert table["HOTA"].tolist() == pytest.approx([0.0, 1.0, 1.0])
    saved = pd.read_csv(p["reports"] / "tracker_grid_bytetrack.csv")
    assert len(saved) == 3 and "HOTA" in saved
    with pytest.raises(ValueError, match="test set"):
        tune_tracker("det", "bytetrack", "T-MOT", grid=grid, reports_dir=p["reports"], **kw)
    # grid 3b: tuned3b name, its own csv; the plan3 csv is not touched
    before = (p["reports"] / "tracker_grid_bytetrack.csv").read_bytes()
    best, table = tune_tracker("det", "bytetrack", "MOT-tune", grid=grid, grid_name="3b",
                               reports_dir=p["reports"], **kw)
    assert best["name"] == "bytetrack-tuned3b" and best["new_track_thresh"] == 0.3
    assert len(pd.read_csv(p["reports"] / "tracker_grid_bytetrack_3b_det.csv")) == 3
    assert (p["reports"] / "tracker_grid_bytetrack.csv").read_bytes() == before


@pytest.mark.parametrize("set_name", ["T-MOT", "VD-train", "UD-pool", "P-train", "nope"])
def test_tune_refuses_non_tune_sets_before_the_leakage_preflight(set_name, monkeypatch):
    def no_preflight(*a, **k):
        raise AssertionError("the role check must come before the leakage preflight")

    monkeypatch.setattr(pl, "load_checked_manifest", no_preflight)
    with pytest.raises(ValueError, match=rf"{set_name}.*only on a tune set"):
        tune_tracker("det", "bytetrack", set_name, models_cfg={}, manifest_dir=Path("m"),
                     hash_cache=Path("h"), pred_cache=Path("p"))


def test_tune_cli_writes_loadable_yaml(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(pl, "tracker_grid", lambda kind, grid="plan3": [
        {"track_high_thresh": 0.3, "new_track_thresh": 0.3, "match_thresh": 0.8,
         "track_buffer": 30}])
    out = tmp_path / "bytetrack.tuned.yaml"
    _cli("track", "tune", "--model", "det", "--tracker-type", "bytetrack", "--out", str(out),
         "--reports-dir", str(p["reports"]), "--warp-cache", str(p["gmc"]),
         *_common(p, "--out-root"))
    from cvt.track.run import load_tracker_cfg

    cfg = load_tracker_cfg(out)
    assert cfg["name"] == "bytetrack-tuned" and cfg["new_track_thresh"] == 0.3
    head = out.read_text(encoding="utf-8").splitlines()[0]
    assert head == (f"# bytetrack-tuned: rule-6 grid on MOT-tune, detector det, best of 1 by "
                    f"class-combined HOTA ({1.0:.4f}); written by `cvt track tune`, grid in "
                    f"{p['reports']}/tracker_grid_bytetrack.csv")
    # --grid 3b: default out configs/trackers/<type>.tuned3b.<detector>.yaml (relative)
    monkeypatch.chdir(tmp_path)
    _cli("track", "tune", "--model", "det", "--tracker-type", "bytetrack", "--grid", "3b",
         "--base", str(CFG_DIR / "bytetrack.yaml"),
         "--reports-dir", str(p["reports"]), "--warp-cache", str(p["gmc"]),
         *_common(p, "--out-root"))
    out3b = tmp_path / "configs" / "trackers" / "bytetrack.tuned3b.det.yaml"
    assert load_tracker_cfg(out3b)["name"] == "bytetrack-tuned3b"
    head = out3b.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("# bytetrack-tuned3b: grid 3b on MOT-tune, detector det, best of 1 ")
    assert head.endswith(f"{p['reports']}/tracker_grid_bytetrack_3b_det.csv")
    r = CliRunner().invoke(app, ["track", "tune", "--model", "det", "--tracker-type",
                                 "bytetrack", "--grid", "plan4", *_common(p, "--out-root")])
    assert r.exit_code == 1 and "plan4" in r.output


# --- comparison ----------------------------------------------------------------------------

def test_compare_rows_paired(tmp_path, monkeypatch):
    gt, tracks = _two_seq_perfect()
    res = compare_tracking(tracks, tracks.iloc[0:0], gt, _ignore([]), [1], n_boot=30)
    assert set(res) == {"delta/HOTA", "delta/MOTA", "delta/IDF1"}
    d, lo, hi = res["delta/HOTA"]
    assert d == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)

    p = _manifest(tmp_path, monkeypatch)
    strict = yaml.safe_load((CFG_DIR / "bytetrack.yaml").read_text())
    strict.update(name="bytetrack-strict", new_track_thresh=0.95)
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text(yaml.safe_dump(strict))
    for cfg in (CFG_DIR / "bytetrack.yaml", strict_path):
        _cli("track", "run", "--model", "det", "--tracker", str(cfg), "--sets", "T-MOT",
             *_common(p))
    _cli("track", "compare", "--a", "det+bytetrack", "--b", "det+bytetrack-strict", "--sets",
         "T-MOT", "--n-boot", "30", "--results", str(p["results"]),
         *_common(p, "--models", "--pred-cache"))
    df = pd.read_csv(p["results"])
    assert sorted(df.metric) == ["delta/HOTA", "delta/IDF1", "delta/MOTA"]
    assert (df.model == "det+bytetrack").all() and (df.notes.str.startswith("vs det+bytetrack-strict")).all()
    assert df.loc[df.metric == "delta/HOTA", "value"].iloc[0] == pytest.approx(1.0)
    assert (df.n_scenes == 2).all() and (df.imgsz == 1280).all()


def test_compare_refuses_runs_on_different_detections(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    strict = yaml.safe_load((CFG_DIR / "bytetrack.yaml").read_text())
    strict.update(name="bytetrack-strict", new_track_thresh=0.95)
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text(yaml.safe_dump(strict))
    for cfg in (CFG_DIR / "bytetrack.yaml", strict_path):
        run_tracking("det", cfg, "T-MOT", **_kw(p))
    meta_b = p["tracks"] / "det+bytetrack-strict" / "T-MOT.json"
    meta = json.loads(meta_b.read_text())
    meta["detections_cache_tag"] = "0123456789ab"
    meta_b.write_text(json.dumps(meta))
    for cross in (False, True):  # same detector, other detections: never paired
        with pytest.raises(ValueError, match=rf"{TAG}.*0123456789ab"):
            list(pl.compare_tracking_sets("det+bytetrack", "det+bytetrack-strict", ["T-MOT"],
                                          manifest_dir=p["manifest"], hash_cache=p["hash"],
                                          out_root=p["tracks"], n_boot=5, cross_detector=cross))
    r = CliRunner().invoke(app, ["track", "compare", "--a", "det+bytetrack", "--b",
                                 "det+bytetrack-strict", "--sets", "T-MOT", "--results",
                                 str(p["results"]), *_common(p, "--models", "--pred-cache")])
    assert r.exit_code == 1 and "0123456789ab" in r.output and not p["results"].exists()


def _with_det1536(p: dict) -> None:
    """A second detector `det1536` (registry imgsz 1536) with its own cached detections."""
    cfg = yaml.safe_load(p["models"].read_text())
    cfg["det1536"] = {**cfg["det"], "imgsz": 1536}
    p["models"].write_text(yaml.safe_dump(cfg))
    for set_name in SEQS:
        write_det_cache(p, set_name, tag="0123456789ab", model="det1536")


def test_rows_take_imgsz_from_the_registry(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    _with_det1536(p)
    tracker = str(CFG_DIR / "bytetrack.yaml")
    for model in ("det", "det1536"):
        _cli("track", "run", "--model", model, "--tracker", tracker, "--sets", "T-MOT",
             *_common(p))
        _cli("track", "eval", "--model", model, "--tracker", tracker, "--sets", "T-MOT",
             "--n-boot", "0", "--results", str(p["results"]), *_common(p))
    df = pd.read_csv(p["results"])
    assert df.groupby("model").imgsz.unique().map(list).to_dict() == {
        "det+bytetrack": [1280], "det1536+bytetrack": [1536]}
    meta = json.loads((p["tracks"] / "det1536+bytetrack" / "T-MOT.json").read_text())
    assert meta["imgsz"] == 1536 and meta["detections_cache_tag"] == "0123456789ab"


def test_compare_across_detectors_needs_the_flag(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    _with_det1536(p)
    tracker = CFG_DIR / "bytetrack.yaml"
    for model in ("det", "det1536"):
        run_tracking(model, tracker, "T-MOT", **_kw(p))
    kw = dict(manifest_dir=p["manifest"], hash_cache=p["hash"], out_root=p["tracks"], n_boot=5)
    with pytest.raises(ValueError, match="--cross-detector"):
        list(pl.compare_tracking_sets("det1536+bytetrack", "det+bytetrack", ["T-MOT"], **kw))
    args = ["track", "compare", "--a", "det1536+bytetrack", "--b", "det+bytetrack", "--sets",
            "T-MOT", "--n-boot", "5", "--results", str(p["results"]),
            *_common(p, "--models", "--pred-cache")]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1 and not p["results"].exists()
    _cli(*args, "--cross-detector")
    df = pd.read_csv(p["results"], dtype={"model_sha": str})
    assert sorted(df.metric) == ["delta/HOTA", "delta/IDF1", "delta/MOTA"]
    assert (df.imgsz == 1536).all() and (df.model_sha == f"0123456789ab|{TAG}").all()
    assert df.loc[df.metric == "delta/HOTA", "value"].iloc[0] == pytest.approx(0.0)
    assert (df.notes.str.startswith("vs det+bytetrack;")).all()


# --- counting ------------------------------------------------------------------------------

def test_count_scene_identity_homography():
    frames = [1, 2, 3, 4]
    cross = [(f, 1, x - 5, 45, x + 5, 55, 1) for f, x in zip(frames, (10, 20, 40, 50), strict=True)]
    gt = _gt(cross, "s")
    # track 2 is on the far side only in frame 3, whose homography is broken -> no crossing
    pred = _tracks(cross + [(f, 2, x - 5, 75, x + 5, 85, 1)
                            for f, x in zip(frames, (10, 20, 40, 20), strict=True)], "s")
    Hs = np.tile(np.eye(3), (4, 1, 1))
    Hs[2] = np.nan
    scene = {"seq_id": "s", "set_name": "T-MOT", "ref_frame": 0,
             "lines": [Line("l", (30.0, 0.0), (30.0, 100.0))]}
    out = count_scene(pred, gt, Hs, np.ones(4, bool), scene, frames=frames)
    assert out["errors"]["count_MAE"] == pytest.approx(0.0)
    assert out["errors"]["gt_total"] == 1 and out["errors"]["pred_total"] == 1
    cells = out["cells"]
    assert list(cells.index.names) == ["line", "label", "direction"]
    # d = (0, 100): cross(d, v) = -100 * (x - 30), so moving to x > 30 is "neg"
    assert cells.loc[("l", 1, "neg"), "gt"] == 1 and cells.loc[("l", 1, "neg"), "pred"] == 1


def _rule2_scene():
    frames = [1, 2, 3, 4]
    scene = {"seq_id": "s", "set_name": "T-MOT", "ref_frame": 0,
             "lines": [Line("l", (30.0, 0.0), (30.0, 200.0))]}
    sizes = np.tile([200.0, 200.0], (4, 1))  # border margin m = max(2, 0.01 * 200) = 2 px
    return frames, scene, np.tile(np.eye(3), (4, 1, 1)), np.ones(4, bool), sizes


def test_count_rule2_counts_a_track_crossing_back_and_forth_once():
    frames, scene, Hs, valid, sizes = _rule2_scene()
    # track 1 crosses x = 30, comes back and stays: 2 events under rule 1, 1 under rule 2
    jitter = [(f, 1, x - 5, 45, x + 5, 55, 1) for f, x in zip(frames, (10, 40, 20, 15), strict=True)]
    clean = [(f, 2, x - 5, 95, x + 5, 105, 1) for f, x in zip(frames, (10, 20, 40, 50), strict=True)]
    gt, pred = _gt(jitter), _tracks(jitter + clean)
    one = count_scene(pred, gt, Hs, valid, scene, frames=frames)
    same = count_scene(pred, gt, Hs, valid, scene, frames=frames, rule=1, sizes=sizes)
    assert one["errors"] == same["errors"] and one["cells"].equals(same["cells"])
    assert (one["errors"]["gt_total"], one["errors"]["pred_total"]) == (2, 3)
    two = count_scene(pred, gt, Hs, valid, scene, frames=frames, rule=2, sizes=sizes)
    assert (two["errors"]["gt_total"], two["errors"]["pred_total"]) == (1, 2)  # same rule on GT
    assert two["cells"].to_dict("index") == {("l", 1, "neg"): {"gt": 1, "pred": 2}}


def test_count_rule2_ignores_boxes_at_the_image_border():
    frames, scene, Hs, valid, sizes = _rule2_scene()
    xs = (10, 20, 40, 50)
    inner = [(f, 3, x - 5, 95, x + 5, 105, 1) for f, x in zip(frames, xs, strict=True)]
    top = [(f, 1, x - 5, 1.5, x + 5, 12, 1) for f, x in zip(frames, xs, strict=True)]  # y1 <= m
    bottom = [(f, 4, x - 5, 190, x + 5, 198.5, 1) for f, x in zip(frames, xs, strict=True)]
    gt, pred = _gt(inner + top), _tracks(inner + top + bottom)
    one = count_scene(pred, gt, Hs, valid, scene, frames=frames)
    assert (one["errors"]["gt_total"], one["errors"]["pred_total"]) == (2, 3)
    two = count_scene(pred, gt, Hs, valid, scene, frames=frames, rule=2, sizes=sizes)
    assert (two["errors"]["gt_total"], two["errors"]["pred_total"]) == (1, 1)
    assert two["errors"]["count_MAE"] == pytest.approx(0.0)
    # per-frame sizes: on a larger image the bottom box is away from the border
    tall = np.tile([400.0, 400.0], (4, 1))
    assert count_scene(pred, gt, Hs, valid, scene, frames=frames, rule=2,
                       sizes=tall)["errors"]["pred_total"] == 2
    with pytest.raises(ValueError, match="sizes"):
        count_scene(pred, gt, Hs, valid, scene, frames=frames, rule=2)
    with pytest.raises(ValueError, match="rule"):
        count_scene(pred, gt, Hs, valid, scene, frames=frames, rule=3, sizes=sizes)


def _crossing_in_ignore():
    """A predicted track crossing x = 30 inside an ignore region (GT never sees it), plus a GT
    and predicted track crossing outside it."""
    frames = [1, 2, 3, 4]
    xs = (10, 20, 40, 50)
    inside = [(f, 7, x - 5, 45, x + 5, 55, 1) for f, x in zip(frames, xs, strict=True)]
    outside = [(f, 1, x - 5, 145, x + 5, 155, 1) for f, x in zip(frames, xs, strict=True)]
    ignore = _ignore([(f, 0, 30, 60, 70) for f in frames])  # covers y 30..70 only
    scene = {"seq_id": "s", "set_name": "T-MOT", "ref_frame": 0,
             "lines": [Line("l", (30.0, 0.0), (30.0, 200.0))]}
    return frames, _gt(outside), _tracks(outside + inside), ignore, scene


def test_count_scene_drops_tracker_rows_in_ignore_regions():
    frames, gt, pred, ignore, scene = _crossing_in_ignore()
    Hs, valid = np.tile(np.eye(3), (4, 1, 1)), np.ones(4, bool)
    out = count_scene(pred, gt, Hs, valid, scene, frames=frames, ignore=ignore)
    assert out["errors"]["gt_total"] == 1 and out["errors"]["pred_total"] == 1
    assert out["errors"]["count_MAE"] == pytest.approx(0.0)
    unmasked = count_scene(pred, gt, Hs, valid, scene, frames=frames)
    assert unmasked["errors"]["pred_total"] == 2  # without the rule it would add a crossing


def test_count_rows_apply_the_ignore_rule(tmp_path, monkeypatch):
    frames, gt, pred, ignore, scene = _crossing_in_ignore()
    monkeypatch.setattr(pl, "reference_homographies",
                        lambda paths, boxes=None, ref=0: (np.tile(np.eye(3), (len(paths), 1, 1)),
                                                          np.ones(len(paths), bool)))
    fr = pd.DataFrame({"image_id": [f"s/{f}" for f in frames], "path": "unused",
                       "width": 200, "height": 200, "seq_id": "s", "frame": frames})
    s = pl.SetData("T-MOT", fr, fr, gt, ignore, (1,))
    dets = pred.assign(score=0.9)
    rows = pl._count_rows(s, pred, [scene], dets, TAG, tmp_path / "stab", pl._quiet)
    mae = [r for r in rows if r["metric"] == "count_MAE" and r["slice"] == "all"]
    assert len(mae) == 1 and mae[0]["value"] == pytest.approx(0.0)
    # rule-1 rows first, then the same rows for rule 2
    assert [(r["slice"], r["metric"]) for r in rows] == [
        ("s", "count_MAE"), ("s", "count_pct_err"), ("all", "count_MAE"), ("all", "count_pct_err"),
        ("s", "count2_MAE"), ("s", "count2_pct_err"), ("all", "count2_MAE"),
        ("all", "count2_pct_err")]
    mae2 = [r for r in rows if r["metric"] == "count2_MAE" and r["slice"] == "all"]
    assert mae2[0]["value"] == pytest.approx(0.0) and mae2[0]["n_scenes"] == 1


def test_eval_with_scenes_writes_count_rows(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(pl, "reference_homographies",
                        lambda paths, boxes=None, ref=0: (np.tile(np.eye(3), (len(paths), 1, 1)),
                                                          np.ones(len(paths), bool)))
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "M0101.yaml").write_text(yaml.safe_dump(
        {"seq_id": "M0101", "set_name": "T-UAVDT", "ref_frame": 0,
         "lines": [{"name": "road", "p1": [0, 35], "p2": [127, 35]}]}))
    (scenes / "README.md").write_text("not a scene")
    tracker = str(CFG_DIR / "bytetrack.yaml")
    _cli("track", "run", "--model", "det", "--tracker", tracker, "--sets", "T-UAVDT",
         *_common(p))
    _cli("track", "eval", "--model", "det", "--tracker", tracker, "--sets", "T-UAVDT",
         "--n-boot", "10", "--scenes", str(scenes), "--stab-cache", str(tmp_path / "stab"), "--results", str(p["results"]),
         *_common(p))
    df = pd.read_csv(p["results"])
    cnt = df[df.metric.isin(["count_MAE", "count_pct_err"])]
    assert set(cnt["slice"]) == {"M0101", "all"}
    cnt2 = df[df.metric.isin(["count2_MAE", "count2_pct_err"])]
    assert set(cnt2["slice"]) == {"M0101", "all"} and len(cnt2) == len(cnt)
    mae = cnt[(cnt.metric == "count_MAE") & (cnt["slice"] == "all")].iloc[0]
    assert mae.n_scenes == 1 and mae.eval_set == "T-UAVDT"


def _scenes_dir(tmp_path: Path) -> Path:
    """One T-MOT and one T-UAVDT scene; the T-UAVDT line crosses nothing, the T-MOT line is
    crossed by nobody either (static objects), so the moving track added below is the only count."""
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "M0101.yaml").write_text(yaml.safe_dump(
        {"seq_id": "M0101", "set_name": "T-UAVDT", "ref_frame": 0,
         "lines": [{"name": "road", "p1": [0, 35], "p2": [127, 35]}]}))
    (scenes / "uav0000002_00000_v.yaml").write_text(yaml.safe_dump(
        {"seq_id": "uav0000002_00000_v", "set_name": "T-MOT", "ref_frame": 0,
         "lines": [{"name": "cut", "p1": [60, 0], "p2": [60, 95]}]}))
    (scenes / "README.md").write_text("not a scene")
    return scenes


def test_track_counts_writes_cells_and_never_touches_results(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(pl, "reference_homographies",
                        lambda paths, boxes=None, ref=0: (np.tile(np.eye(3), (len(paths), 1, 1)),
                                                          np.ones(len(paths), bool)))
    scenes = _scenes_dir(tmp_path)
    tracker = str(CFG_DIR / "bytetrack.yaml")
    _cli("track", "run", "--model", "det", "--tracker", tracker, "--sets", "T-MOT,T-UAVDT",
         *_common(p))
    # add a predicted car crossing the T-MOT line from left to right (GT has no such crossing)
    path = p["tracks"] / "det+bytetrack" / "T-MOT.parquet"
    tr = pd.read_parquet(path)
    extra = tr[(tr.seq_id == "uav0000002_00000_v") & (tr.label == 1)].copy()
    extra["track_id"] = 999
    extra["x1"], extra["x2"] = [30.0, 40.0, 70.0, 80.0, 90.0], [40.0, 50.0, 80.0, 90.0, 100.0]
    pd.concat([tr, extra], ignore_index=True).to_parquet(path, index=False)

    _cli("track", "eval", "--model", "det", "--tracker", tracker, "--sets", "T-MOT,T-UAVDT",
         "--n-boot", "0", "--scenes", str(scenes), "--stab-cache", str(tmp_path / "stab"), "--results", str(p["results"]), *_common(p))
    before = p["results"].read_bytes()
    monkeypatch.chdir(tmp_path)  # the default --out is relative: reports/counts_<name>.csv
    r = _cli("track", "counts", "--model", "det", "--tracker", tracker, "--sets", "T-MOT,T-UAVDT",
             "--scenes", str(scenes), "--stab-cache", str(tmp_path / "stab"), *_common(p))
    out = tmp_path / "reports" / "counts_det+bytetrack.csv"
    assert str(Path("reports") / "counts_det+bytetrack.csv") in r.output
    assert p["results"].read_bytes() == before  # never appends result rows

    cells = pd.read_csv(out)
    assert list(cells.columns) == ["eval_set", "scene", "line", "label", "class", "direction",
                                   "gt", "pred"]
    # rule 2 next to it (same columns); the added car crosses once, away from the border
    out2 = tmp_path / "reports" / "counts2_det+bytetrack.csv"
    assert str(Path("reports") / "counts2_det+bytetrack.csv") in r.output
    pd.testing.assert_frame_equal(pd.read_csv(out2), cells)
    assert len(cells) == 1  # the only crossing anywhere: the added car on T-MOT
    c = cells.iloc[0]
    assert (c.eval_set, c.scene, c.line, c.label, c["class"]) == (
        "T-MOT", "uav0000002_00000_v", "cut", 1, "car")
    assert (c["gt"], c["pred"]) == (0, 1)
    # the same cells give the count rows `cvt track eval` wrote
    res = pd.read_csv(p["results"])
    mae = res[(res.metric == "count_MAE") & (res.eval_set == "T-MOT")].set_index("slice")
    assert mae.loc["uav0000002_00000_v", "value"] == pytest.approx(
        (cells["pred"] - cells["gt"]).abs().mean())


def test_track_counts_needs_scene_files(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    empty = tmp_path / "noscenes"
    empty.mkdir()
    tracker = str(CFG_DIR / "bytetrack.yaml")
    _cli("track", "run", "--model", "det", "--tracker", tracker, "--sets", "T-MOT", *_common(p))
    r = CliRunner().invoke(app, ["track", "counts", "--model", "det", "--tracker", tracker,
                                 "--sets", "T-MOT", "--scenes", str(empty), "--out",
                                 str(tmp_path / "c.csv"), *_common(p)])
    assert r.exit_code == 1 and "no scene files" in r.output and not (tmp_path / "c.csv").exists()


def test_track_counts_rule2_cells(tmp_path, monkeypatch):
    """A predicted car crossing back and forth: two crossings under rule 1, one under rule 2."""
    p = _manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(pl, "reference_homographies",
                        lambda paths, boxes=None, ref=0: (np.tile(np.eye(3), (len(paths), 1, 1)),
                                                          np.ones(len(paths), bool)))
    scenes = _scenes_dir(tmp_path)
    tracker = str(CFG_DIR / "bytetrack.yaml")
    _cli("track", "run", "--model", "det", "--tracker", tracker, "--sets", "T-MOT", *_common(p))
    path = p["tracks"] / "det+bytetrack" / "T-MOT.parquet"
    tr = pd.read_parquet(path)
    extra = tr[(tr.seq_id == "uav0000002_00000_v") & (tr.label == 1)].copy()
    extra["track_id"] = 999
    extra["x1"], extra["x2"] = [30.0, 70.0, 30.0, 30.0, 30.0], [40.0, 80.0, 40.0, 40.0, 40.0]
    pd.concat([tr, extra], ignore_index=True).to_parquet(path, index=False)
    o1, o2 = tmp_path / "c1.csv", tmp_path / "c2.csv"
    _cli("track", "counts", "--model", "det", "--tracker", tracker, "--sets", "T-MOT",
         "--scenes", str(scenes), "--stab-cache", str(tmp_path / "stab"), "--out", str(o1), "--out2", str(o2), *_common(p))
    c1, c2 = pd.read_csv(o1), pd.read_csv(o2)
    assert c1[["direction", "gt", "pred"]].values.tolist() == [["neg", 0, 1], ["pos", 0, 1]]
    assert c2[["direction", "gt", "pred"]].values.tolist() == [["neg", 0, 1]]


def test_eval_refuses_stale_tracks(tmp_path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    tracker = str(CFG_DIR / "bytetrack.yaml")
    r = CliRunner().invoke(app, ["track", "eval", "--model", "det", "--tracker", tracker,
                                 "--sets", "T-MOT", "--results", str(p["results"]), *_common(p)])
    assert r.exit_code == 1 and "cvt track run" in r.output
    _cli("track", "run", "--model", "det", "--tracker", tracker, "--sets", "T-MOT", *_common(p))
    edited = yaml.safe_load(Path(tracker).read_text())
    edited["match_thresh"] = 0.9
    other = tmp_path / "bytetrack.yaml"
    other.write_text(yaml.safe_dump(edited))
    r = CliRunner().invoke(app, ["track", "eval", "--model", "det", "--tracker", str(other),
                                 "--sets", "T-MOT", "--results", str(p["results"]), *_common(p)])
    assert r.exit_code == 1 and "different tracker config" in r.output


# --- report --------------------------------------------------------------------------------

def _row(**kw):
    base = {c: "" for c in RESULT_COLS}
    base.update(timestamp="2026-10-22T00:00:00", model="det+botsort-tuned", imgsz=1280,
                eval_set="T-MOT", slice="all", metric="HOTA", value=0.61, ci_lo=0.55,
                ci_hi=0.66, notes="classes: motorbike,car")
    base.update(kw)
    return base


def test_tracking_table_renders(tmp_path: Path):
    assert tracking_table(pd.DataFrame(columns=RESULT_COLS)) == "(no tracking results yet)"
    res = tmp_path / "r.csv"
    append_results(res, [
        _row(),
        _row(metric="MOTA", value=0.5, ci_lo=0.4, ci_hi=0.6),
        _row(eval_set="T-UAVDT", metric="HOTA", value=0.7, ci_lo=0.6, ci_hi=0.8),
        _row(eval_set="T-UAVDT", slice="night", metric="HOTA", value=0.4, ci_lo=float("nan"),
             ci_hi=float("nan")),
        _row(eval_set="T-UAVDT", metric="count_MAE", value=1.5, ci_lo=float("nan"),
             ci_hi=float("nan")),
        _row(eval_set="T-UAVDT", metric="count_pct_err", value=0.1, ci_lo=float("nan"),
             ci_hi=float("nan")),
        _row(eval_set="T-UAVDT", slice="M0101", metric="count_MAE", value=9.0),
        _row(metric="delta/HOTA", value=0.03, ci_lo=0.01, ci_hi=0.05,
             notes="vs det+bytetrack-tuned"),
        _row(model="dronefreak-yolo11s", metric="mAP50"),  # own small table, after the note
        _row(model="dronefreak-yolo11s", eval_set="T-VD", metric="mAP50"),  # main README table
    ])
    df = pd.read_csv(res, dtype={"slice": str, "notes": str})
    md = tracking_table(df)
    track_md, det_md = md.split("Detection mAP on tracking sets")
    assert "| det+botsort-tuned | 61.0 [55.0, 66.0] | 50.0 [40.0, 60.0] | – | 70.0 [60.0, 80.0]" in md
    assert "T-UAVDT HOTA night" in md and "| 40.0 |" in md
    assert "1.5" in md and "10.0%" in md and "9.0" not in md
    assert "| det+botsort-tuned | vs det+bytetrack-tuned | +3.0 [+1.0, +5.0] |" in md
    assert "count2" not in md and "rule 2" not in md  # no rule-2 rows: table unchanged
    assert "dronefreak" not in track_md
    assert "| dronefreak-yolo11s | 1280 | 61.0 [55.0, 66.0] |" in det_md and "T-VD" not in det_md

    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- tracking:start -->\n(no tracking results yet)\n<!-- tracking:end -->\n")
    _cli("report", "--results", str(res), "--readme", str(readme))
    assert "| det+botsort-tuned | 61.0 [55.0, 66.0]" in readme.read_text()


def test_tracking_table_count2_columns_after_the_rule1_ones():
    nan = float("nan")
    rows = [_row(),
            _row(eval_set="T-MOT", metric="count_pct_err", value=0.2, ci_lo=nan, ci_hi=nan),
            _row(eval_set="T-MOT", metric="count2_pct_err", value=0.05, ci_lo=nan, ci_hi=nan),
            _row(eval_set="T-MOT", metric="count2_MAE", value=3.0, ci_lo=nan, ci_hi=nan),
            _row(model="det+bytetrack-tuned")]  # a run without rule-2 rows
    md = tracking_table(pd.DataFrame(rows))
    head = md.splitlines()[0]
    assert head.endswith("| T-MOT count MAE | T-MOT count % err | T-UAVDT count MAE | "
                         "T-UAVDT count % err | T-MOT count2 % err | T-UAVDT count2 % err |")
    assert "| – | 20.0% | – | – | 5.0% | – |" in md
    assert "| – | – | – | – | – | – |" in md.splitlines()[3]  # det+bytetrack-tuned
    assert "rule 2" in md


def test_tracking_table_skips_tiled_detectors_and_keeps_each_comparison():
    rows = [_row(),
            _row(model="m@1280+tiles640-fp16", eval_set="T-VN", metric="mAP50"),  # a detector
            _row(model="det+botsort-tuned3b", metric="delta/HOTA", value=0.016, ci_lo=0.005,
                 ci_hi=0.026, notes="vs det+botsort-tuned; classes: car"),
            _row(timestamp="2026-10-23T00:00:00", model="det+botsort-tuned3b",
                 metric="delta/HOTA", value=0.043, ci_lo=0.022, ci_hi=0.071,
                 notes="vs det+bytetrack-tuned3b; classes: car")]
    md = tracking_table(pd.DataFrame(rows))
    assert "tiles640" not in md
    assert "| det+botsort-tuned3b | vs det+botsort-tuned | +1.6 [+0.5, +2.6] |" in md
    assert "| det+botsort-tuned3b | vs det+bytetrack-tuned3b | +4.3 [+2.2, +7.1] |" in md


def test_tracking_table_slice_headers_name_the_sequence_count():
    rows = [_row(eval_set="T-UAVDT", slice="high_alt", metric="HOTA", value=0.28, ci_lo=0.27,
                 ci_hi=0.29, n_scenes=2),
            _row(eval_set="T-UAVDT", slice="daylight", metric="HOTA", value=0.55, ci_lo=0.5,
                 ci_hi=0.6, n_scenes=12),
            _row(eval_set="T-UAVDT", slice="all", metric="HOTA", value=0.5, n_scenes=20)]
    md = tracking_table(pd.DataFrame(rows))
    head = md.splitlines()[0]
    assert "| T-UAVDT HOTA high_alt (2 seq) |" in head
    assert "| T-UAVDT HOTA daylight (12 seq) |" in head
    assert "| T-UAVDT HOTA night |" in head  # no rows for the slice: no count to show
    assert "| T-UAVDT HOTA |" in head  # the whole-set column is unchanged
    assert "CIs from 3 or fewer sequences are not meaningful" in md
