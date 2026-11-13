"""Lock tuning, evaluation, README block and script on the synthetic MOT manifest of
tests/test_track_pipeline.py (no GPU, no model; GMC warps stubbed to the identity)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import cvt.follow.lock as lock_mod
import cvt.track.pipeline as pl
import cvt.train.job as job
from cvt.cli import app
from cvt.data.manifest import load_manifest, load_mot
from cvt.eval.report import follow_table, results_table, tracking_table
from cvt.eval.results import RESULT_COLS, append_results
from cvt.eval.run import set_hash_of
from cvt.follow import features as feat_mod
from cvt.follow import pipeline as fp
from cvt.follow.features import VEHICLE_IDS, features_path
from cvt.follow.lock import lock_params
from tests.test_track_pipeline import SEQS, TAG, _manifest, write_det_cache

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "configs" / "trackers" / "botsort.tuned3b.src_pseudo-yolo11m-s0@1536-fp16.yaml"
VARIANTS = ("id_only", "mil", "reacq", "features", "hsv")
LOCK_METRICS = ["lock/success_auc", "lock/precision_20", "lock/norm_precision_auc",
                "lock/lock_kept", "lock/wrong_frame_rate", "lock/wrong_switches_per_target"]
GIMBAL_METRICS = ["gimbal/centre_err", "gimbal/inside", "gimbal/smooth", "gimbal/clamped_frac"]
PIPE = "det+botsort-tuned3b"


def _identity_gmc(monkeypatch):
    monkeypatch.setattr(pl, "gmc_warps",
                        lambda paths, method, downscale=2: np.tile(np.eye(2, 3), (len(paths), 1, 1)))


def _write_feats(p: dict, set_name: str, feats_dir: Path) -> Path:
    """Features of every cached vehicle detection: one-hot class + a little box position."""
    images, boxes, _ = load_manifest(p["manifest"])
    s = pl.load_set(set_name, images, boxes, load_mot(p["manifest"])[0])
    models_cfg = yaml.safe_load(p["models"].read_text())
    dets, tag = pl.cached_detections("det", models_cfg, s, images, boxes, p["preds"])
    keep = np.flatnonzero(dets.label.isin(VEHICLE_IDS).to_numpy())
    feat = np.zeros((len(keep), 8))
    feat[np.arange(len(keep)), dets.label.to_numpy()[keep]] = 1.0
    feat[:, 7] = dets.x1.to_numpy()[keep] / 128.0
    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    path = features_path(feats_dir, "det", tag, set_name)
    feat_mod._save(path, keep, feat, {"det_model": "det", "det_tag": tag,
                                      "set_hash": set_hash_of(s.im)})
    return path


def _setup(tmp_path: Path, monkeypatch, feats: bool = True) -> tuple[dict, Path, Path]:
    p = _manifest(tmp_path, monkeypatch)
    _identity_gmc(monkeypatch)
    monkeypatch.setattr(job, "no_network", lambda: None)  # no global Ultralytics patching here
    cfg = yaml.safe_load((ROOT / "configs" / "follow.yaml").read_text(encoding="utf-8"))
    # synthetic sequences: 6 frame numbers (frame 4 absent); at 1 fps every vehicle lasts >= 5 s
    cfg.update(detector="det", models=str(p["models"]), tracker=str(TRACKER),
               fps={s: 1 for s in SEQS})
    cfg_path = tmp_path / "follow.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    feats_dir = tmp_path / "feats"
    if feats:
        for s in SEQS:
            _write_feats(p, s, feats_dir)
    return p, cfg_path, feats_dir


def _opts(p: dict, feats_dir: Path) -> list[str]:
    return ["--manifest", str(p["manifest"]), "--hash-cache", str(p["hash"]),
            "--pred-cache", str(p["preds"]), "--out-root", str(p["tracks"]),
            "--warp-cache", str(p["gmc"]), "--feats-dir", str(feats_dir)]


def _kw(p: dict, feats_dir: Path) -> dict:
    return dict(models_cfg=yaml.safe_load(p["models"].read_text()), manifest_dir=p["manifest"],
                hash_cache=p["hash"], pred_cache=p["preds"], out_root=p["tracks"],
                warp_cache=p["gmc"], feats_dir=feats_dir)


def _tune(p, cfg_path, feats_dir, tmp_path):
    out = tmp_path / "follow.tuned.yaml"
    r = CliRunner().invoke(app, ["follow", "tune", "--config", str(cfg_path), "--out", str(out),
                                 "--reports-dir", str(tmp_path / "reports"),
                                 *_opts(p, feats_dir)])
    assert r.exit_code == 0, r.output
    return out


# --- config ---------------------------------------------------------------------------------

def test_follow_config_declares_the_grids():
    cfg = fp.load_follow_cfg(ROOT / "configs" / "follow.yaml")
    assert cfg["detector"] == "src_pseudo-yolo11m-s0@1536-fp16"
    assert cfg["tracker"] == "configs/trackers/botsort.tuned3b.src_pseudo-yolo11m-s0@1536-fp16.yaml"
    assert cfg["fps"] == {"MOT-tune": 30, "T-MOT": 30, "T-UAVDT": 30, "UAVDT-S": 30,
                          "UAV123-veh": 30, "UAV20L-veh": 30}
    assert cfg["min_s"] == 5 and cfg["hfov_deg"] == 84
    lock = fp.lock_grid(cfg)
    assert len(lock) == 18
    assert lock[0] == {"coast_s": 1, "reid_thr": 0.5, "gate_growth": 0.5}
    assert lock[-1] == {"coast_s": 4, "reid_thr": 0.9, "gate_growth": 1.0}
    gim = fp.gimbal_grid(cfg)
    assert len(gim) == 12
    assert gim[0] == {"kp": 0.5, "kd": 0, "target_frac": 0.15}
    assert gim[-1] == {"kp": 2, "kd": 0.2, "target_frac": 0.25}
    assert "30 fps" in (ROOT / "configs" / "follow.yaml").read_text(encoding="utf-8")


# --- feature adapter ------------------------------------------------------------------------

def test_track_feature_fn_matches_track_boxes_to_featured_detections():
    dets = pd.DataFrame({
        "seq_id": ["a", "a", "a", "b", "a"],
        "frame": [1, 1, 1, 1, 2],
        "x1": [0.0, 50.0, 0.0, 0.0, 0.0], "y1": [0.0, 0.0, 0.0, 0.0, 0.0],
        "x2": [10.0, 60.0, 12.0, 10.0, 10.0], "y2": [10.0, 10.0, 10.0, 10.0, 10.0],
        "score": 0.9, "label": 1})
    det_rows = np.array([0, 1, 2, 3])  # row 4 has no feature
    feats = np.eye(4, dtype=np.float16) * 2  # not unit length on disk: normalised on the way out
    fn = fp.track_feature_fn(dets, det_rows, feats, "a")
    out = fn(1, np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 0.0, 61.0, 10.0], [90.0, 0, 99, 9]]))
    assert out.shape == (3, 4) and out.dtype == np.float32
    np.testing.assert_allclose(out[0], [1, 0, 0, 0])  # IoU 1 with row 0 beats 10/12 with row 2
    np.testing.assert_allclose(out[1], [0, 1, 0, 0])  # IoU 10/11 >= 0.5
    assert np.isnan(out[2]).all()  # no detection overlaps
    one = fn(1, np.array([[0.0, 0.0, 12.0, 10.0]]))  # one-box query
    assert one.shape == (1, 4)
    np.testing.assert_allclose(one[0], [0, 0, 1, 0])
    assert fn(1, np.array([[90.0, 0, 99, 9]])) is None  # nothing matched
    assert fn(2, np.array([[0.0, 0.0, 10.0, 10.0]])) is None  # frame 2's detection has no feature
    assert fn(7, np.array([[0.0, 0.0, 10.0, 10.0]])) is None  # no detections at all
    assert fn(1, np.zeros((0, 4))) is None


# --- tuning ---------------------------------------------------------------------------------

@pytest.mark.parametrize("set_name", ["T-MOT", "T-UAVDT", "VD-tune-not-a-set"])
def test_tune_refuses_test_sets(set_name, tmp_path):
    cfg = fp.load_follow_cfg(ROOT / "configs" / "follow.yaml")
    with pytest.raises(ValueError, match="only on a tune set"):
        # refused before anything is read: the manifest does not even exist
        fp.tune_follow(cfg, set_name, models_cfg={}, manifest_dir=tmp_path / "none",
                       hash_cache=tmp_path / "h", pred_cache=tmp_path / "p",
                       reports_dir=tmp_path / "reports")
    assert not (tmp_path / "reports").exists()
    r = CliRunner().invoke(app, ["follow", "tune", "--set", set_name, "--out",
                                 str(tmp_path / "t.yaml"), "--manifest", str(tmp_path / "none")])
    assert r.exit_code == 1 and "only on a tune set" in r.output
    assert not (tmp_path / "t.yaml").exists()


def test_tune_writes_tuned_yaml_and_grids(tmp_path, monkeypatch):
    p, cfg_path, feats_dir = _setup(tmp_path, monkeypatch)
    assert not pl.tracks_path(p["tracks"], PIPE, "MOT-tune").exists()
    out = _tune(p, cfg_path, feats_dir, tmp_path)
    # the 3b tracker ran on MOT-tune because its tracks were missing; no test set was touched
    assert pl.tracks_path(p["tracks"], PIPE, "MOT-tune").exists()
    assert not pl.tracks_path(p["tracks"], PIPE, "T-MOT").exists()
    lock = pd.read_csv(tmp_path / "reports" / "follow_grid_lock.csv")
    gim = pd.read_csv(tmp_path / "reports" / "follow_grid_gimbal.csv")
    assert len(lock) == 18 and len(gim) == 12
    assert list(lock.columns[:3]) == ["coast_s", "reid_thr", "gate_growth"]
    assert {"success_auc", "precision_20", "lock_kept", "n_targets"} <= set(lock.columns)
    assert list(gim.columns[:3]) == ["kp", "kd", "target_frac"]
    assert {"centre_err", "inside", "smooth", "clamped_frac"} <= set(gim.columns)
    assert (lock.n_targets == 1).all()  # MOT-tune: one car (the pedestrian is not a lock test)

    text = out.read_text(encoding="utf-8")
    assert text.startswith("#") and "MOT-tune" in text.splitlines()[0]
    tuned = yaml.safe_load(text)
    assert tuned["set"] == "MOT-tune" and tuned["detector"] == "det"
    assert tuned["tracker"] == "botsort-tuned3b"
    # static objects: every lock point scores the same -> the first in grid order
    assert lock.success_auc.nunique() == 1
    assert tuned["lock"] == {"coast_s": 1, "reid_thr": 0.5, "gate_growth": 0.5}
    # gimbal: min centring error, ties within 5% of the best -> min smoothness
    near = gim[gim.centre_err <= gim.centre_err.min() * 1.05]
    best = gim.loc[near.smooth.idxmin()]
    assert tuned["gimbal"] == {"kp": best.kp, "kd": best.kd, "target_frac": best.target_frac}


def test_gimbal_choice_ties_within_five_percent_go_to_the_smoothest():
    t = pd.DataFrame({"centre_err": [0.10, 0.104, 0.20, 0.10], "smooth": [3.0, 1.0, 0.1, 3.0]})
    assert fp.pick_gimbal(t) == 1
    t = pd.DataFrame({"centre_err": [0.10, 0.2], "smooth": [3.0, 3.0]})
    assert fp.pick_gimbal(t) == 0


def test_tune_needs_the_features(tmp_path, monkeypatch):
    p, cfg_path, feats_dir = _setup(tmp_path, monkeypatch, feats=False)
    r = CliRunner().invoke(app, ["follow", "tune", "--config", str(cfg_path), "--out",
                                 str(tmp_path / "t.yaml"), "--reports-dir",
                                 str(tmp_path / "reports"), *_opts(p, feats_dir)])
    assert r.exit_code == 1 and r"scripts\follow.ps1 -Step features" in r.output
    assert not (tmp_path / "t.yaml").exists()


# --- evaluation -----------------------------------------------------------------------------

def test_eval_rows_and_variants(tmp_path, monkeypatch):
    p, cfg_path, feats_dir = _setup(tmp_path, monkeypatch)
    tuned = _tune(p, cfg_path, feats_dir, tmp_path)
    res = tmp_path / "results.csv"
    r = CliRunner().invoke(app, ["follow", "eval", "--config", str(cfg_path), "--tuned",
                                 str(tuned), "--sets", "T-MOT,T-UAVDT", "--n-boot", "50",
                                 "--results", str(res), *_opts(p, feats_dir)])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(res, dtype={"slice": str, "notes": str})
    assert list(df.columns) == RESULT_COLS
    for set_name, n_targets in (("T-MOT", 2), ("T-UAVDT", 4)):
        d = df[df.eval_set == set_name]
        want = {f"{PIPE}+lock-{v}" for v in VARIANTS} | {f"{PIPE}+gimbal-gt", f"{PIPE}+gimbal-lock"}
        assert set(d.model) == want
        assert len(d) == 5 * len(LOCK_METRICS) + 2 * len(GIMBAL_METRICS)
        for v in VARIANTS:
            assert sorted(d[d.model == f"{PIPE}+lock-{v}"].metric) == sorted(LOCK_METRICS)
        for g in ("gt", "lock"):
            assert sorted(d[d.model == f"{PIPE}+gimbal-{g}"].metric) == sorted(GIMBAL_METRICS)
        assert (d["slice"] == "all").all() and (d.n_scenes == 2).all()
        assert (d.n_images == n_targets * 5).all()  # scored frames: 5 annotated frames per target
        assert (d.model_sha == TAG).all() and (d.imgsz == 1280).all()
        assert d.ci_lo.notna().all() and d.ci_hi.notna().all()
        auc = d[(d.model == f"{PIPE}+lock-features") & (d.metric == "lock/success_auc")].iloc[0]
        assert auc.value == pytest.approx(20 / 21)  # perfect lock: IoU > t fails only at t = 1
        assert d.config_hash.nunique() == 7  # one per variant / gimbal mode
    # eval tracked the test sets with the frozen 3b tracker on its own (tune never did)
    assert pl.tracks_path(p["tracks"], PIPE, "T-MOT").exists()


def test_eval_refuses_params_tuned_for_another_pipeline(tmp_path, monkeypatch):
    p, cfg_path, feats_dir = _setup(tmp_path, monkeypatch)
    tuned = tmp_path / "t.yaml"
    tuned.write_text(yaml.safe_dump({"set": "MOT-tune", "detector": "other", "tracker":
                                     "botsort-tuned3b", "lock": {"coast_s": 1, "reid_thr": 0.5,
                                                                 "gate_growth": 0.5},
                                     "gimbal": {"kp": 1, "kd": 0, "target_frac": 0.15}}))
    r = CliRunner().invoke(app, ["follow", "eval", "--config", str(cfg_path), "--tuned",
                                 str(tuned), "--results", str(tmp_path / "r.csv"),
                                 *_opts(p, feats_dir)])
    assert r.exit_code == 1 and "cvt follow tune" in r.output
    assert not (tmp_path / "r.csv").exists()


def test_lost_track_falls_back_to_mil_on_disk_frames(tmp_path, monkeypatch):
    p, cfg_path, feats_dir = _setup(tmp_path, monkeypatch)
    path = write_det_cache(p, "T-MOT")  # same tag: the features file still applies
    dt = pd.read_parquet(path)
    gone = dt.image_id.str.endswith("uav0000002_00000_v/0000003")
    dt[~gone].to_parquet(path, index=False)
    for f in Path(p["preds"]).glob("*_T-MOT_*.parquet"):  # one cache file per set
        assert f == path
    _write_feats(p, "T-MOT", feats_dir)
    shapes: list = []

    class FakeMil:
        def init(self, img, box):
            shapes.append(img.shape)
            self.box = box

        def update(self, img):
            shapes.append(img.shape)
            return True, self.box

    monkeypatch.setattr(lock_mod, "make_mil", FakeMil)
    cfg = fp.load_follow_cfg(cfg_path)
    fs = fp.prepare_set(cfg, "T-MOT", **_kw(p, feats_dir))
    per, outs = fp.run_locks(fs, lock_params("features", 2.0, 0.5, 0.5))
    assert len(per) == 2 and set(per.seq_id) == {"uav0000002_00000_v", "uav0000003_00000_v"}
    lost = outs[("uav0000002_00000_v", 1)].set_index("frame")
    assert lost.loc[3, "source"] == "mil" and lost.loc[5, "source"] == "track"
    assert lost.loc[3, ["x1", "y1", "x2", "y2"]].tolist() == [10, 10, 40, 30]
    assert shapes and all(s == (96, 128, 3) for s in shapes)  # real frames read from disk
    assert set(outs[("uav0000003_00000_v", 1)].source) == {"track"}


def test_feature_fns_split_once_and_slice_per_sequence(monkeypatch):
    # two interleaved sequences; row 5 has no feature
    dets = pd.DataFrame({
        "seq_id": ["a", "b", "a", "b", "b", "a"],
        "frame": [1, 1, 2, 1, 2, 3],
        "x1": [0.0, 0.0, 0.0, 50.0, 0.0, 0.0], "y1": 0.0,
        "x2": [10.0, 10.0, 10.0, 60.0, 10.0, 10.0], "y2": 10.0, "score": 0.9, "label": 1})
    det_rows = np.array([0, 1, 2, 3, 4])
    feats = (np.eye(5, 6) * 3).astype(np.float16)
    calls: list = []
    real = fp.seq_feature_fn

    def spy(sub, f):
        calls.append((sorted(set(sub.seq_id)), len(sub), np.asarray(f).dtype))
        return real(sub, f)

    monkeypatch.setattr(fp, "seq_feature_fn", spy)
    fns = fp.feature_fns(dets, det_rows, feats, ["a", "b", "c"])
    # each sequence converts only its own rows (still float16 when handed over)
    assert calls == [(["a"], 2, np.float16), (["b"], 3, np.float16), ([], 0, np.float16)]
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    np.testing.assert_allclose(fns["a"](1, box)[0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(fns["a"](2, box)[0], [0, 0, 1, 0, 0, 0])
    assert fns["a"](3, box) is None  # row 5 is not featured
    np.testing.assert_allclose(fns["b"](1, box)[0], [0, 1, 0, 0, 0, 0])
    two = fns["b"](1, np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 0.0, 60.0, 10.0]]))
    np.testing.assert_allclose(two, [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert fns["c"](1, box) is None
    # the single-sequence adapter gives the same answers
    for seq in ("a", "b"):
        one = fp.track_feature_fn(dets, det_rows, feats, seq)
        for fr in (1, 2, 3):
            got, want = one(fr, box), fns[seq](fr, box)
            assert (got is None) == (want is None)
            if got is not None:
                np.testing.assert_array_equal(got, want)
    empty = fp.feature_fns(dets, np.zeros(0, np.int64), np.zeros((0, 6), np.float16), ["a"])
    assert empty["a"](1, box) is None


def test_prepare_set_offline_guards_and_per_sequence_features(tmp_path, monkeypatch):
    p, cfg_path, feats_dir = _setup(tmp_path, monkeypatch)
    events: list = []
    monkeypatch.setattr(job, "no_network", lambda: events.append("offline"))
    gmc, track = pl.gmc_warps, pl.run_tracking

    def gmc_spy(*a, **kw):
        events.append("gmc")
        return gmc(*a, **kw)

    def track_spy(*a, **kw):
        events.append("track")
        return track(*a, **kw)

    monkeypatch.setattr(pl, "gmc_warps", gmc_spy)
    monkeypatch.setattr(pl, "run_tracking", track_spy)
    slices: list = []
    real = fp.seq_feature_fn
    monkeypatch.setattr(fp, "seq_feature_fn",
                        lambda sub, f: slices.append(set(sub.seq_id)) or real(sub, f))
    cfg = fp.load_follow_cfg(cfg_path)
    fs = fp.prepare_set(cfg, "T-UAVDT", **_kw(p, feats_dir))
    assert events[:2] == ["offline", "track"]  # guarded before the tracker (and its GMC) runs
    assert len(slices) == len(fs.feats) == 2 and all(len(s) == 1 for s in slices)
    assert {next(iter(s)) for s in slices} == set(fs.feats)
    # tracks cached, warps deleted: each uncached GMC computation is guarded
    for f in Path(p["gmc"]).glob("*.npy"):
        f.unlink()
    events.clear()
    fp.prepare_set(cfg, "T-UAVDT", **_kw(p, feats_dir))
    assert events == ["offline", "gmc", "offline", "gmc"]
    events.clear()
    fp.prepare_set(cfg, "T-UAVDT", **_kw(p, feats_dir))
    assert events == []  # everything cached: Ultralytics is not touched


def _fake_set(n: int, name: str = "T-MOT") -> fp.FollowSet:
    targets = pd.DataFrame({"seq_id": "s", "track_id": np.arange(1, n + 1), "label": 1,
                            "first": 0, "last": 0})
    gt = pd.DataFrame({"seq_id": "s", "frame": 0, "track_id": np.arange(1, n + 1),
                       "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "label": 1})
    fs = fp.FollowSet(name, PIPE, None, targets, 1.0, TAG, {}, {}, "m")
    fs.frames, fs.warps, fs.paths, fs.size = {"s": [0]}, {"s": None}, {"s": {}}, {"s": (64, 48)}
    fs.tracks = {"s": pd.DataFrame({c: pd.Series(dtype=float) for c in ["frame", "track_id"]})}
    fs.gt, fs.feats = {"s": gt}, {"s": None}
    return fs


def _stub_lock(monkeypatch):
    def run_lock(tr, feats, warps, frames, first, box, params, fps, read):
        return pd.DataFrame({"frame": [0], "x1": [0.0], "y1": [0.0], "x2": [10.0], "y2": [10.0],
                             "source": ["track"], "track_id": [1]})

    monkeypatch.setattr(fp, "run_lock", run_lock)


def test_run_locks_logs_progress_every_50_targets(monkeypatch):
    _stub_lock(monkeypatch)
    fs = _fake_set(120)
    logs: list[str] = []
    per, outs = fp.run_locks(fs, lock_params("features", 1.0, 0.5, 0.5), logs.append)
    assert len(per) == 120 and len(outs) == 120
    prog = [m for m in logs if "target " in m]
    assert prog == ["[T-MOT] lock-features: target 50/120", "[T-MOT] lock-features: target 100/120"]
    assert fp.variant_of(lock_params("id_only", 1.0, 0.5, 0.5)) == "id_only"


def test_duplicate_lock_targets_are_refused(monkeypatch):
    _stub_lock(monkeypatch)
    fs = _fake_set(3)
    fs.targets = pd.concat([fs.targets, fs.targets.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"not unique.*\('s', 2\)"):
        fp.run_locks(fs, lock_params("features", 1.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="not unique"):
        fp.run_gimbal(fs, fp.GimbalParams(kp=1.0, kd=0.0, target_frac=0.2, hfov_deg=84.0))


def test_summarize_raises_on_a_nan_target_metric():
    per = pd.DataFrame({"seq_id": ["a", "b"], "track_id": [1, 9], "n_frames": [10.0, 10.0],
                        "centre_err": [0.1, float("nan")], "inside": [1.0, 1.0],
                        "smooth": [0.1, 0.1], "clamped_frac": [0.0, 0.0]})
    with pytest.raises(ValueError, match=r"centre_err.*seq b track 9"):
        fp.summarize(per, fp.GIMBAL_KEYS, n_boot=10)


# --- README ---------------------------------------------------------------------------------

def _row(**kw):
    base = {c: "" for c in RESULT_COLS}
    base.update(timestamp="2026-11-08T00:00:00", model=f"{PIPE}+lock-features", imgsz=1536,
                eval_set="T-MOT", slice="all", metric="lock/success_auc", value=0.61,
                ci_lo=0.55, ci_hi=0.66, n_scenes=14, n_images=1000,
                notes="12 lock tests (vehicle GT tracks >= 5 s at 30 fps); params tuned on MOT-tune")
    base.update(kw)
    return base


def test_follow_table_and_main_table_exclusion(tmp_path):
    assert follow_table(pd.DataFrame(columns=RESULT_COLS)) == "(no follow results yet)"
    nan = float("nan")
    rows = [
        _row(),
        _row(model=f"{PIPE}+lock-id_only", value=0.40, ci_lo=0.30, ci_hi=0.50),
        _row(eval_set="T-UAVDT", value=0.70, ci_lo=0.60, ci_hi=0.80, n_scenes=20,
             notes="30 lock tests (vehicle GT tracks >= 5 s at 30 fps); params tuned on MOT-tune"),
        _row(metric="lock/wrong_switches_per_target", value=1.25, ci_lo=0.5, ci_hi=2.0),
        _row(metric="lock/wrong_frame_rate", value=0.05, ci_lo=0.01, ci_hi=0.09),
        _row(model=f"{PIPE}+gimbal-gt", metric="gimbal/centre_err", value=0.031, ci_lo=0.02,
             ci_hi=0.04, notes="driven by the GT box; 12 lock tests (vehicle GT tracks >= 5 s)"),
        _row(model=f"{PIPE}+gimbal-lock", metric="gimbal/smooth", value=0.123, ci_lo=nan,
             ci_hi=nan),
        # detection and tracking rows of the same pipeline stay where they were
        _row(model="det", eval_set="T-VD", metric="mAP50", value=0.5, notes="mAP classes: car"),
        _row(model=PIPE, metric="HOTA", value=0.51, notes="classes: car"),
    ]
    df = pd.DataFrame(rows)
    md = follow_table(df)
    lock_md, gim_md = md.split("Virtual gimbal")
    # one pipeline: named once above the tables, variants shown by their short name
    assert lock_md.startswith(f"Pipeline `{PIPE}`")
    tab = [ln for ln in lock_md.splitlines() if ln.startswith("|")]
    assert tab[0].startswith("| set | lock variant | success AUC | prec@20px |")
    # one row per set and variant, ablation order; a set's first row names its lock tests
    rows_ = [ln.split(" | ")[:2] for ln in tab[2:]]
    assert rows_ == [["| T-MOT (12 targets, 14 seq)", "id_only"], ["| ", "features"],
                     ["| T-UAVDT (30 targets, 20 seq)", "features"]]
    assert "61.0 [55.0, 66.0]" in tab[3] and "1.25 [0.50, 2.00]" in tab[3]
    assert "5.0 [1.0, 9.0]" in tab[3] and "70.0 [60.0, 80.0]" in tab[4]
    gim_tab = [ln for ln in gim_md.splitlines() if ln.startswith("|")]
    assert gim_tab[0].startswith("| set | gimbal input | centre err | inside |")
    assert "| T-MOT (12 targets, 14 seq) | GT box | 3.1 [2.0, 4.0]" in gim_md
    # without a count in the notes only the sequences are named
    bare = follow_table(pd.DataFrame([_row(notes="")]))
    assert "| T-MOT (14 seq) | features |" in bare
    assert "0.123" in gim_md and "HOTA" not in md and "mAP" not in md
    assert "MOT-tune" in md  # where the parameters were tuned

    assert "lock-" not in results_table(df) and "lock-" not in tracking_table(df)
    res = tmp_path / "r.csv"
    append_results(res, rows)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- tracking:start -->\n<!-- tracking:end -->\n"
                      "<!-- follow:start -->\n(no follow results yet)\n<!-- follow:end -->\n",
                      encoding="utf-8")
    r = CliRunner().invoke(app, ["report", "--results", str(res), "--readme", str(readme)])
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    main, rest = text.split("<!-- results:end -->")
    tracking, follow = rest.split("<!-- tracking:end -->")
    assert "lock-" not in main and "gimbal-" not in main and "| det |" in main
    assert "lock-" not in tracking and "gimbal-" not in tracking and f"| {PIPE} |" in tracking
    assert f"Pipeline `{PIPE}`" in follow and "| features |" in follow
    assert "(no follow results yet)" not in follow


def test_results_page_has_the_follow_block():
    text = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    assert "<!-- follow:start -->" in text and "<!-- follow:end -->" in text
    assert text.index("<!-- tracking:end -->") < text.index("<!-- follow:start -->") < text.index(
        "### Deployment")


# --- script ---------------------------------------------------------------------------------

PS1 = ROOT / "scripts" / "follow.ps1"


def test_follow_ps1_is_ascii_lf():
    raw = PS1.read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell not available")
def test_follow_ps1_dryrun():
    parse = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
             f"'{PS1}', [ref]$null, [ref]$e); $e.Count")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", parse], capture_output=True,
                       text=True, timeout=120)
    assert r.stdout.strip() == "0", r.stdout + r.stderr
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
                        "-DryRun"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [ln.removeprefix(">>> ") for ln in r.stdout.splitlines() if ln.startswith(">>> ")]
    det = "src_pseudo-yolo11m-s0@1536-fp16"
    models = "configs/models.yaml,configs/models_deploy.yaml"
    assert cmds == [
        f"uv run --extra ml cvt follow features --weights-model src_pseudo-yolo11m-s0 "
        f"--boxes-of '{det}' --imgsz 1536 --sets MOT-tune,T-MOT,T-UAVDT --min-score 0.1 "
        f"--models {models}",
        f"uv run --extra ml cvt follow sot-detect --config configs/follow.yaml --sot "
        f"configs/follow_sot.yaml --models {models} --weights-model src_pseudo-yolo11m-s0 "
        "--imgsz 1536 --min-score 0.1 --sets UAVDT-S,UAV123-veh,UAV20L-veh",
        f"uv run --extra ml cvt follow tune --config configs/follow.yaml --models {models} "
        "--out configs/follow.tuned.yaml",
        f"uv run --extra ml cvt follow eval --config configs/follow.yaml --models {models} "
        "--tuned configs/follow.tuned.yaml --sets T-MOT,T-UAVDT",
        f"uv run --extra ml cvt follow sot-eval --config configs/follow.yaml --sot "
        f"configs/follow_sot.yaml --models {models} --tuned configs/follow.tuned.yaml "
        "--sets UAVDT-S,UAV123-veh,UAV20L-veh",
        "uv run --extra ml cvt report",
    ]
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
                        "-Step", "tune", "-DryRun"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.count(">>> ") == 1 and "follow tune" in r.stdout
