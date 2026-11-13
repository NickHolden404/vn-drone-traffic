"""Public lock benchmarks (UAVDT-S, UAV123 / UAV20L vehicle subsets) on tiny fixture trees in
tmp_path written in the native formats (no GPU, no model; GMC warps stubbed to the identity)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import cvt.track.pipeline as pl
from cvt.cli import app
from cvt.data.leakage import phash_u64
from cvt.data.schema import CLASS_ID
from cvt.eval.report import follow_table
from cvt.follow import pipeline as fp
from cvt.follow import sot_data as sd

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "configs" / "trackers" / "botsort.tuned3b.src_pseudo-yolo11m-s0@1536-fp16.yaml"
SOT_CFG = ROOT / "configs" / "follow_sot.yaml"
SOT_SETS = ("UAVDT-S", "UAV123-veh", "UAV20L-veh")
CAR = CLASS_ID["car"]
TUNED = {"set": "MOT-tune", "detector": "det", "tracker": "botsort-tuned3b",
         "lock": {"coast_s": 2, "reid_thr": 0.7, "gate_growth": 1.0},
         "gimbal": {"kp": 2, "kd": 0.2, "target_frac": 0.25}}


# --- fixture trees ---------------------------------------------------------------------------

def _img(path: Path, k: int, w: int = 96, h: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(k)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _boxes(n: int, x0: float = 10.0) -> np.ndarray:
    """x, y, w, h rows of a box drifting right."""
    return np.array([[x0 + 2 * i, 20.0, 16.0, 12.0] for i in range(n)])


def _write_xywh(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join("NaN" if np.isnan(v) else f"{v:g}" for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _uavdt_tree(root: Path, seqs: dict[str, int], anno_only=(), frames_only=(), seed=0) -> Path:
    for k, (name, n) in enumerate(seqs.items()):
        if name not in anno_only:
            for i in range(1, n + 1):
                _img(root / "UAV-benchmark-S" / name / f"img{i:06d}.jpg", seed + 100 * k + i)
        if name not in frames_only:
            _write_xywh(root / "UAV-benchmark-SOT_v1.0" / "anno" / f"{name}_gt.txt", _boxes(n))
    return root


CONFIG_SEQS_FIXTURE = r"""function seqs=configSeqs

seqUAV20L = {struct('name','car1','path','d:\data_seq\UAV123\car1\','startFrame',1,'endFrame',8,'nz',6,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','car9','path','d:\data_seq\UAV123\car9\','startFrame',1,'endFrame',6,'nz',6,'ext','jpg','init_rect', [0,0,0,0])};

seqUAV123={struct('name','car1_1','path','d:\data_seq\UAV123\car1\','startFrame',1,'endFrame',5,'nz',6,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','car1_2','path','d:\data_seq\UAV123\car1\','startFrame',5,'endFrame',8,'nz',6,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','car9','path','d:\data_seq\UAV123\car9\','startFrame',1,'endFrame',6,'nz',6,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','person1','path','d:\data_seq\UAV123\person1\','startFrame',1,'endFrame',9,'nz',6,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','truck2','path','d:\data_seq\UAV123\truck2\','startFrame',1,'endFrame',4,'nz',6,'ext','jpg','init_rect', [0,0,0,0])};

seqs = seqUAV20L; %seqUAV123, seqUAV20L
"""
UAV_RANGES = {"car1_1": ("car1", 1, 5), "car1_2": ("car1", 5, 8), "car9": ("car9", 1, 6),
              "truck2": ("truck2", 1, 4)}


def _uav_tree(root: Path) -> Path:
    """car1 (8 frames; UAV20L car1, UAV123 pieces car1_1 = 1..5 and car1_2 = 5..8, with a NaN
    row and its last row 1 px off the UAV20L one, as on the real car1_2), car9 (6 frames, single
    piece in both subsets), truck2 (6 frames, officially 1..4: 4 annotated rows), configSeqs.m."""
    full = _boxes(8)
    full[6] = np.nan
    anno = root / "anno"
    _write_xywh(anno / "UAV20L" / "car1.txt", full)
    _write_xywh(anno / "UAV123" / "car1_1.txt", full[0:5])
    piece2 = full[4:8].copy()
    piece2[-1, 0] += 1.0
    _write_xywh(anno / "UAV123" / "car1_2.txt", piece2)
    _write_xywh(anno / "UAV20L" / "car9.txt", _boxes(6, 30.0))
    _write_xywh(anno / "UAV123" / "car9.txt", _boxes(6, 30.0))
    _write_xywh(anno / "UAV123" / "truck2.txt", _boxes(4))
    for v, n in (("car1", 8), ("car9", 6), ("truck2", 6)):
        for i in range(1, n + 1):
            _img(root / "data_seq" / "UAV123" / v / f"{i:06d}.jpg", sum(map(ord, v)) + i)
    (root / "configSeqs.m").write_text(CONFIG_SEQS_FIXTURE, encoding="utf-8")
    return root


# --- readers ---------------------------------------------------------------------------------

def test_uavdt_sot_reader(tmp_path):
    root = _uavdt_tree(tmp_path / "uavdt_sot", {"S0101": 4, "S0102": 3, "S0201": 2, "S0301": 3},
                       anno_only=("S0201",), frames_only=("S0301",))
    # a frame-count mismatch is excluded too
    (root / "UAV-benchmark-S" / "S0102" / "img000003.jpg").unlink()
    logs: list[str] = []
    seqs = sd.load_uavdt_sot(root, fps=30.0, log=logs.append)
    assert [s.name for s in seqs] == ["S0101"]
    s = seqs[0]
    assert s.fps == 30.0 and s.video == "UAVDT-S/S0101" and s.start == 1
    assert [Path(p).name for p in s.frame_paths] == [f"img{i:06d}.jpg" for i in range(1, 5)]
    # x, y, w, h -> xyxy
    np.testing.assert_allclose(s.gt[0], [10, 20, 26, 32])
    np.testing.assert_allclose(s.gt[3], [16, 20, 32, 32])
    excl = sd.uavdt_sot_inventory(root)[1]
    assert set(excl) == {"S0102", "S0201", "S0301"}
    assert "no frame folder" in excl["S0201"] and "no annotation" in excl["S0301"]
    assert "2 frames" in excl["S0102"] and "3 annotation rows" in excl["S0102"]
    assert any("S0201" in m for m in logs)


def test_uav123_reader_nan_rows(tmp_path):
    root = _uav_tree(tmp_path / "UAV123")
    seqs = sd.load_uav123(root, ["car9"], fps=30.0)
    assert len(seqs) == 1 and seqs[0].start == 1 and len(seqs[0].frame_paths) == 6
    logs: list[str] = []
    pieces = sd.load_uav123(root, ["car1_1", "car1_2", "truck2"], ranges=UAV_RANGES,
                            log=logs.append)
    p2 = pieces[1]
    assert p2.video == "UAV123/car1" and p2.start == 5
    assert [Path(p).name for p in p2.frame_paths] == [f"{i:06d}.jpg" for i in range(5, 9)]
    assert p2.gt.shape == (4, 4)
    assert np.isnan(p2.gt[2]).all() and not np.isnan(p2.gt[[0, 1, 3]]).any()
    # the official range of truck2 covers 4 of its 6 frames: fine
    assert len(pieces[2].frame_paths) == 4 and pieces[2].end == 4
    # the UAV20L alignment is a logged cross-check: car1_2 agrees despite its 1 px last row
    assert any("car1_2" in m and "agrees" in m and "3/4 rows" in m for m in logs)
    # UAV20L full video, NaN row kept
    full = sd.load_uav123(root, ["car1"], subset="UAV20L")[0]
    assert full.video == "UAV123/car1" and np.isnan(full.gt[6]).all()
    # annotation rows must equal the range length
    with pytest.raises(ValueError, match="car1_2.*5 frames, its annotation 4 rows"):
        sd.load_uav123(root, ["car1_2"], ranges={"car1_2": ("car1", 4, 8)})
    # a range off the alignment loads, but the cross-check says so
    logs.clear()
    sd.load_uav123(root, ["car1_1"], ranges={"car1_1": ("car1", 2, 6)}, log=logs.append)
    assert any("DISAGREES" in m for m in logs)
    # every frame of the range must exist
    (root / "data_seq" / "UAV123" / "car1" / "000006.jpg").unlink()
    with pytest.raises(ValueError, match="car1_2.*missing.*000006.jpg"):
        sd.load_uav123(root, ["car1_2"], ranges=UAV_RANGES)


def test_config_seqs_parser_and_alignment_crosscheck(tmp_path):
    rng = np.random.default_rng(0)
    full = rng.uniform(0, 500, (40, 4))
    full[12] = np.nan
    assert sd.align_piece(full[10:25], full) == [10]
    assert sd.align_piece(full[0:1], full) == [0]
    noisy = full[10:25].copy()
    noisy[3, 0] += 1e-3  # beyond 1e-6: no exact match, but the best offset is still 10
    assert sd.align_piece(noisy, full) == [] and sd.best_alignment(noisy, full) == (10, 14)
    nan_off = full[10:25].copy()
    nan_off[5] = np.nan  # NaN where the full annotation has a box: no exact match
    assert sd.align_piece(nan_off, full) == []
    rep = np.tile(full[:3], (4, 1))  # periodic annotation: ambiguous
    assert sd.align_piece(full[:3], rep) == [0, 3, 6, 9]

    root = _uav_tree(tmp_path / "UAV123")
    cells = sd.parse_config_seqs(root / "configSeqs.m")
    assert cells["seqUAV20L"] == {"car1": ("car1", 1, 8), "car9": ("car9", 1, 6)}
    assert {k: v for k, v in cells["seqUAV123"].items() if k != "person1"} == UAV_RANGES
    assert sd.official_ranges(root, ["car1_2", "truck2"]) == {"car1_2": ("car1", 5, 8),
                                                              "truck2": ("truck2", 1, 4)}
    with pytest.raises(ValueError, match="truck4_1"):
        sd.official_ranges(root, ["truck4_1"])
    assert sd.crosscheck_uav20l(root, "car1_2", "car1", 5).startswith("agrees")
    assert sd.crosscheck_uav20l(root, "truck2", "truck2", 1) is None  # not in UAV20L


# --- config ----------------------------------------------------------------------------------

def test_sot_lists_are_vehicle_only():
    cfg = sd.load_sot_cfg(SOT_CFG)
    veh = re.compile(r"^(car|truck)\d+(_\d+)?$")
    u123, u20 = cfg["uav123_veh"], cfg["uav20l_veh"]
    assert all(veh.match(n) for n in u123["sequences"]) and all(veh.match(n) for n in u20["sequences"])
    assert sorted(u20["sequences"]) == sorted(["car1", "car3", "car6", "car8", "car9", "car16"])
    for sim in ("car1_s", "car2_s", "car3_s", "car4_s"):
        assert sim not in u123["sequences"] and "simulator" in u123["excluded"][sim]
    assert set(u123["excluded"]) == {"car1_s", "car2_s", "car3_s", "car4_s"}
    assert len(u123["sequences"]) == 31  # every real car/truck sequence of configSeqs.m
    assert u123["sequences"]["truck2"] == ["truck2", 1, 385]
    assert u123["sequences"]["car1_2"] == ["car1", 751, 1627]
    assert u123["sequences"]["truck4_2"] == ["truck4", 577, 1261]
    for name, (video, start, end) in u123["sequences"].items():
        assert video == name.split("_")[0] and 1 <= start < end
        if name == video:
            assert start == 1
    assert len(cfg["uavdt_s"]["sequences"]) == 50
    assert all(re.fullmatch(r"S\d{4}", n) for n in cfg["uavdt_s"]["sequences"])
    assert cfg["classes"] == ["car", "truck", "bus", "motorbike"]
    text = SOT_CFG.read_text(encoding="utf-8")
    assert "configSeqs.m" in text and "30 fps" in text and "T-UAVDT" in text
    follow = fp.load_follow_cfg(ROOT / "configs" / "follow.yaml")
    assert all(follow["fps"][s] == 30 for s in SOT_SETS)
    # the declared sets are checked by the loaders: names outside car*/truck* are refused
    bad = {**cfg, "uav123_veh": {**u123, "sequences": {**u123["sequences"],
                                                      "person1": ["person1", 1, 10]}}}
    with pytest.raises(ValueError, match="person1"):
        sd.check_sot_cfg(bad)


def test_declared_uav123_ranges_must_equal_config_seqs(tmp_path):
    root = _uav_tree(tmp_path / "UAV123")
    seqs = {n: list(r) for n, r in UAV_RANGES.items()}
    sot = {"classes": ["car"], "uavdt_s": {"root": "x", "sequences": [], "excluded": {}},
           "uav123_veh": {"root": str(root), "sequences": seqs, "excluded": {}},
           "uav20l_veh": {"root": str(root), "sequences": ["car1"], "excluded": {}}}
    follow = {"fps": {"UAV123-veh": 30}}
    assert [s.name for s in sd.sot_sequences(sot, "UAV123-veh", follow)] == list(UAV_RANGES)
    seqs["car1_2"] = ["car1", 4, 7]
    with pytest.raises(ValueError, match="car1_2.*configSeqs.m"):
        sd.sot_sequences(sot, "UAV123-veh", follow)


@pytest.mark.skipif(not (ROOT / "data/external/uav123/UAV123/configSeqs.m").exists(),
                    reason="UAV123 toolkit files not present")
def test_sot_config_ranges_are_config_seqs_and_crosscheck_on_the_real_annotations():
    cfg = sd.load_sot_cfg(SOT_CFG)
    root = ROOT / cfg["uav123_veh"]["root"]
    declared = {k: tuple(v) for k, v in cfg["uav123_veh"]["sequences"].items()}
    assert sd.official_ranges(root, list(declared)) == declared
    checks = {n: sd.crosscheck_uav20l(root, n, v, s) for n, (v, s, _) in declared.items()}
    checks = {n: c for n, c in checks.items() if c is not None}
    # 12 pieces of UAV20L videos + car3, car9 (whole UAV20L videos)
    assert len(checks) == 14 and all(c.startswith("agrees") for c in checks.values()), checks


# --- leakage ---------------------------------------------------------------------------------

def test_uavdt_s_leakage_drops_matched_sequences(tmp_path):
    root = _uavdt_tree(tmp_path / "uavdt_sot", {"S0101": 3, "S0102": 3})
    seqs = sd.load_uavdt_sot(root)
    hit = phash_u64(seqs[1].frame_paths[2])
    train = pd.DataFrame({"image_id": ["uavdt/M0101/000001", "visdrone/x/1"],
                          "set_name": ["UD-pool", "VD-train"],
                          "phash": np.array([hit, hit ^ 0xFFFFFFFFFFFFFFFF], np.uint64)})
    report, dropped = sd.sot_leakage(seqs, train, tmp_path / "phash_sot.parquet")
    assert dropped == ["S0102"]
    assert list(report.columns) == sd.LEAK_COLS
    assert report.iloc[0].test_id == "UAVDT-S/S0102/3" and report.iloc[0].dist == 0
    assert report.iloc[0].train_set == "UD-pool"
    assert (tmp_path / "phash_sot.parquet").exists()


def test_uavdt_s_overlap_with_test_sets_is_not_leakage(tmp_path, monkeypatch):
    """Only training/tuning-side sets count; T-UAVDT (the UAVDT MOT test set, same source
    videos) or any other test set never drops a UAVDT-S sequence."""
    root = _uavdt_tree(tmp_path / "uavdt_sot", {"S0101": 3, "S0102": 3})
    seqs = sd.load_uavdt_sot(root)
    hits = [phash_u64(s.frame_paths[0]) for s in seqs]
    test_side = pd.DataFrame({"image_id": ["uavdt/M0101/1", "uavdt/M0102/1"],
                              "set_name": ["T-UAVDT", "T-MOT"],
                              "phash": np.array(hits, np.uint64)})
    report, dropped = sd.sot_leakage(seqs, test_side, tmp_path / "phash_sot.parquet")
    assert dropped == [] and report.empty
    assert list(sd.training_side(pd.DataFrame({"set_name": [
        "VD-train", "VD-tune", "P-train", "P-val", "UD-pool", "MOT-tune", "T-UAVDT", "T-MOT",
        "T-VD", "T-VN", "UAVDT-S"]}))) == [True] * 6 + [False] * 5
    # training_hashes reads only training/tuning-side images of the manifest
    from tests.test_track_pipeline import _manifest

    p = _manifest(tmp_path, monkeypatch)
    th = fp.training_hashes(p["manifest"], tmp_path / "phash.parquet")
    assert set(th.set_name) == {"MOT-tune"}  # the fixture's T-MOT / T-UAVDT are test sets


# --- detection cache (sot-detect) ------------------------------------------------------------

class _FakeDet:
    """Detects the GT box of every frame (looked up by path) as a car, plus a far-away truck."""

    name, cache_tag = "det", "abc123"

    def __init__(self, truth: dict[str, np.ndarray]):
        self.truth, self.calls = truth, 0

    def predict_paths(self, paths):
        from cvt.detect.base import Detections

        for p in paths:
            self.calls += 1
            b = self.truth.get(str(p))
            boxes = np.array([[70.0, 2.0, 90.0, 14.0]] + ([] if b is None or np.isnan(b).any()
                                                           else [b]), np.float32)
            yield Detections(boxes, np.full(len(boxes), 0.9, np.float32),
                             np.array([CLASS_ID["truck"]] + [CAR] * (len(boxes) - 1)))


class _FakeExtractor:
    imgsz, stride, channels = 64, 8, 4

    def __init__(self):
        self.frames = 0

    def fmap(self, bgr):
        import torch

        self.frames += 1
        return torch.ones((1, 4, 8, 8))


def _truth(seqs) -> dict[str, np.ndarray]:
    return {p: s.gt[i] for s in seqs for i, p in enumerate(s.frame_paths)}


def _sot_world(tmp_path: Path):
    """Fixture trees, configs and a tuned file in tmp_path."""
    uavdt = _uavdt_tree(tmp_path / "uavdt_sot", {"S0101": 12, "S0102": 10})
    uav = _uav_tree(tmp_path / "UAV123")
    sot = {
        "classes": ["car", "truck", "bus", "motorbike"],
        "uavdt_s": {"root": str(uavdt), "sequences": ["S0101", "S0102"], "excluded": {}},
        "uav123_veh": {"root": str(uav),
                       "sequences": {n: list(r) for n, r in UAV_RANGES.items()},
                       "excluded": {}},
        "uav20l_veh": {"root": str(uav), "sequences": ["car1", "car9"], "excluded": {}},
    }
    sot_path = tmp_path / "follow_sot.yaml"
    sot_path.write_text(yaml.safe_dump(sot), encoding="utf-8")
    cfg = yaml.safe_load((ROOT / "configs" / "follow.yaml").read_text(encoding="utf-8"))
    cfg.update(detector="det", models=str(tmp_path / "models.yaml"), tracker=str(TRACKER))
    cfg_path = tmp_path / "follow.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"pt")
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(
        {"det": {"arch": "yolo", "weights": str(tmp_path / "det.engine"), "imgsz": 1536},
         "wpt": {"arch": "yolo", "weights": str(weights)}}), encoding="utf-8")
    tuned = tmp_path / "follow.tuned.yaml"
    tuned.write_text(yaml.safe_dump(TUNED), encoding="utf-8")
    return cfg_path, sot_path, tuned


def _detect_all(tmp_path, cfg_path, sot_path, monkeypatch):
    cfg, sot = fp.load_follow_cfg(cfg_path), sd.load_sot_cfg(sot_path)
    seqs = {s: sd.sot_sequences(sot, s, cfg) for s in SOT_SETS}
    det = _FakeDet(_truth([q for v in seqs.values() for q in v]))
    ext = _FakeExtractor()
    monkeypatch.setattr(fp, "_open_sot_detector", lambda *a, **k: det)
    monkeypatch.setattr(fp, "_open_extractor", lambda *a, **k: ext)
    cache = tmp_path / "sot_cache"
    models_cfg = yaml.safe_load((tmp_path / "models.yaml").read_text())
    fp.detect_sot(cfg, sot, list(SOT_SETS), models_cfg=models_cfg, weights_model="wpt",
                  imgsz=64, conf=0.01, min_score=0.1, cache_dir=cache)
    return seqs, det, ext, cache, models_cfg


def test_sot_videos_span_gaps_between_pieces(tmp_path):
    """car1_2 is excluded on the real data: car1_1 + car1_3 leave a gap, detected all the same."""
    root = _uav_tree(tmp_path / "UAV123")
    a, b = sd.load_uav123(root, ["car1_1", "car1_2"],
                          ranges={"car1_1": ("car1", 1, 5), "car1_2": ("car1", 5, 8)})
    b.frame_paths, b.start, b.gt = b.frame_paths[2:], 7, b.gt[2:]  # frames 7..8 only
    a.frame_paths, a.gt = a.frame_paths[:3], a.gt[:3]  # frames 1..3 only
    lo, hi, paths = fp.sot_videos([a, b])["UAV123/car1"]
    assert (lo, hi) == (1, 8)
    assert [Path(p).name for p in paths] == [f"{i:06d}.jpg" for i in range(1, 9)]
    assert all(Path(p).exists() for p in paths)
    (root / "data_seq" / "UAV123" / "car1" / "000005.jpg").unlink()
    with pytest.raises(ValueError, match="000005.jpg"):
        fp.sot_videos([a, b])


def test_sot_detect_caches_each_video_once(tmp_path, monkeypatch):
    cfg_path, sot_path, _ = _sot_world(tmp_path)
    seqs, det, ext, cache, models_cfg = _detect_all(tmp_path, cfg_path, sot_path, monkeypatch)
    # UAV20L car1 / car9 and UAV123 car1_1, car1_2, car9 share two videos: each frame once;
    # truck2 adds its 4 official frames (1..4 of 6)
    assert det.calls == 12 + 10 + 8 + 6 + 4 and ext.frames == det.calls
    vids = sorted(p.stem for p in (cache / "det").rglob("*.parquet"))
    assert vids == ["S0101", "S0102", "car1", "car9", "truck2"]
    d, meta = fp.load_sot_dets(cache, "det", "UAV123/car1")
    assert meta["det_tag"] == "abc123" and (meta["first"], meta["last"]) == (1, 8)
    assert sorted(d.frame.unique()) == list(range(1, 9))
    rows, feat = fp.load_sot_feats(cache, "det", "UAV123/car1", meta)
    assert feat.shape == (len(d), 4)  # every detection is a vehicle with score 0.9
    # resumable: a second run detects nothing
    det.calls = ext.frames = 0
    fp.detect_sot(fp.load_follow_cfg(cfg_path), sd.load_sot_cfg(sot_path),
                  list(SOT_SETS), models_cfg=models_cfg, weights_model="wpt", imgsz=64,
                  conf=0.01, min_score=0.1, cache_dir=cache)
    assert det.calls == 0 and ext.frames == 0


# --- evaluation (sot-eval) -------------------------------------------------------------------

def _eval(tmp_path, cfg_path, sot_path, tuned, models_cfg, cache, monkeypatch, train=None):
    monkeypatch.setattr(pl, "gmc_warps",
                        lambda paths, method, downscale=2: np.tile(np.eye(2, 3), (len(paths), 1, 1)))
    train = train if train is not None else pd.DataFrame(
        {"image_id": [], "set_name": [], "phash": np.array([], np.uint64)})
    return list(fp.eval_sot_sets(
        fp.load_follow_cfg(cfg_path), yaml.safe_load(tuned.read_text()), sd.load_sot_cfg(sot_path),
        list(SOT_SETS), models_cfg=models_cfg, cache_dir=cache, warp_cache=tmp_path / "gmc",
        train_hashes=train, sot_hash_cache=tmp_path / "phash_sot.parquet",
        reports_dir=tmp_path / "reports", n_boot=20))


def test_sot_eval_uses_frozen_params(tmp_path, monkeypatch):
    cfg_path, sot_path, tuned = _sot_world(tmp_path)
    _, _, _, cache, models_cfg = _detect_all(tmp_path, cfg_path, sot_path, monkeypatch)
    seen_lock, seen_gim = [], []
    real_lock, real_follow = fp.run_lock, fp.follow

    def spy_lock(tr, feats, warps, frames, first, box, params, fps, read):
        seen_lock.append((params, fps, first, tuple(np.round(box, 3))))
        return real_lock(tr, feats, warps, frames, first, box, params, fps, read)

    def spy_follow(boxes, w, h, fps, params):
        seen_gim.append(params)
        return real_follow(boxes, w, h, fps, params)

    monkeypatch.setattr(fp, "run_lock", spy_lock)
    monkeypatch.setattr(fp, "follow", spy_follow)
    out = _eval(tmp_path, cfg_path, sot_path, tuned, models_cfg, cache, monkeypatch)
    rows = pd.DataFrame([r for part in out for r in part])
    assert sorted(rows.eval_set.unique()) == sorted(SOT_SETS)
    # every lock run uses the frozen lock point; all 5 variants; GT box of the first frame
    assert {(p.coast_s, p.reid_thr, p.gate_growth) for p, *_ in seen_lock} == {(2, 0.7, 1.0)}
    assert {fp.variant_of(p) for p, *_ in seen_lock} == set(fp.VARIANTS)
    assert {fps for _, fps, *_ in seen_lock} == {30.0}
    assert all(first == 0 for _, _, first, _ in seen_lock)
    assert (0, *[10.0, 20.0, 26.0, 32.0]) in {(f, *b) for _, _, f, b in seen_lock}
    assert {(g.kp, g.kd, g.target_frac) for g in seen_gim} == {(2.0, 0.2, 0.25)}
    # rows: 5 variants x 4 OPE metrics + 2 gimbal modes x 4 metrics, per set
    for s, n in (("UAVDT-S", 2), ("UAV123-veh", 4), ("UAV20L-veh", 2)):
        g = rows[rows.eval_set == s]
        assert len(g) == 5 * 4 + 2 * 4
        assert set(g.metric) == {*(f"lock/{k}" for k in fp.SOT_LOCK_KEYS),
                                 *(f"gimbal/{k}" for k in fp.GIMBAL_KEYS)}
        assert (g.n_scenes == n).all() and g.notes.str.contains(f"(?:^|; ){n} lock tests").all()
        assert g.notes.str.contains("tuned on MOT-tune").all()
    perfect = rows[(rows.eval_set == "UAV20L-veh") & (rows.metric == "lock/lock_kept")]
    assert (perfect.value > 0.9).all()  # detections equal the GT: the lock holds
    md = follow_table(rows.assign(timestamp="2026-11-09T00:00:00"))
    assert "UAV123-veh (4 targets, 4 seq)" in md and "UAVDT-S" in md
    assert "wrong-target metrics are not defined" in md
    leak = pd.read_csv(tmp_path / "reports" / "follow_sot_leakage.csv")
    assert leak.empty and list(leak.columns) == sd.LEAK_COLS


def test_sot_eval_refuses_params_not_tuned_on_mot_tune(tmp_path, monkeypatch):
    cfg_path, sot_path, tuned = _sot_world(tmp_path)
    tuned.write_text(yaml.safe_dump({**TUNED, "set": "T-MOT"}), encoding="utf-8")
    called = []
    monkeypatch.setattr(sd, "sot_sequences", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="T-MOT"):
        _eval(tmp_path, cfg_path, sot_path, tuned, {}, tmp_path / "none", monkeypatch)
    assert not called and not (tmp_path / "reports").exists()
    res = tmp_path / "r.csv"
    r = CliRunner().invoke(app, ["follow", "sot-eval", "--config", str(cfg_path), "--sot",
                                 str(sot_path), "--tuned", str(tuned), "--results", str(res)])
    assert r.exit_code == 1 and "T-MOT" in r.output and not res.exists()
    r = CliRunner().invoke(app, ["follow", "sot-eval", "--config", str(cfg_path), "--sot",
                                 str(sot_path), "--tuned", str(tmp_path / "missing.yaml"),
                                 "--results", str(res)])
    assert r.exit_code == 1 and "follow tune" in r.output and not res.exists()


def test_sot_eval_drops_leaked_uavdt_s_sequences(tmp_path, monkeypatch):
    cfg_path, sot_path, tuned = _sot_world(tmp_path)
    _, _, _, cache, models_cfg = _detect_all(tmp_path, cfg_path, sot_path, monkeypatch)
    seq = sd.load_uavdt_sot(tmp_path / "uavdt_sot")[0]
    train = pd.DataFrame({"image_id": ["uavdt/M0101/000004"], "set_name": ["UD-pool"],
                          "phash": np.array([phash_u64(seq.frame_paths[3])], np.uint64)})
    out = _eval(tmp_path, cfg_path, sot_path, tuned, models_cfg, cache, monkeypatch, train)
    rows = pd.DataFrame([r for part in out for r in part])
    u = rows[rows.eval_set == "UAVDT-S"]
    assert (u.n_scenes == 1).all() and u.notes.str.contains("S0101").all()
    leak = pd.read_csv(tmp_path / "reports" / "follow_sot_leakage.csv")
    assert set(leak.test_id.str.split("/").str[1]) == {"S0101"}


def test_sot_eval_needs_the_detection_cache(tmp_path, monkeypatch):
    cfg_path, sot_path, tuned = _sot_world(tmp_path)
    with pytest.raises(ValueError, match="sot-detect"):
        _eval(tmp_path, cfg_path, sot_path, tuned, {"det": {"imgsz": 1536}},
              tmp_path / "empty", monkeypatch)


def test_sot_detect_cli_loads_no_model_for_a_bad_set(tmp_path):
    cfg_path, sot_path, _ = _sot_world(tmp_path)
    r = CliRunner().invoke(app, ["follow", "sot-detect", "--config", str(cfg_path), "--sot",
                                 str(sot_path), "--sets", "T-MOT", "--weights-model", "wpt",
                                 "--cache", str(tmp_path / "c")])
    assert r.exit_code == 1 and "UAVDT-S" in r.output and not (tmp_path / "c").exists()


# --- script ----------------------------------------------------------------------------------

PS1 = ROOT / "scripts" / "follow.ps1"


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell not available")
@pytest.mark.parametrize("step,expect", [
    ("sot-detect", "uv run --extra ml cvt follow sot-detect --config configs/follow.yaml --sot "
                   "configs/follow_sot.yaml --models configs/models.yaml,configs/models_deploy.yaml "
                   "--weights-model src_pseudo-yolo11m-s0 --imgsz 1536 --min-score 0.1 "
                   "--sets UAVDT-S,UAV123-veh,UAV20L-veh"),
    ("sot-eval", "uv run --extra ml cvt follow sot-eval --config configs/follow.yaml --sot "
                 "configs/follow_sot.yaml --models configs/models.yaml,configs/models_deploy.yaml "
                 "--tuned configs/follow.tuned.yaml --sets UAVDT-S,UAV123-veh,UAV20L-veh"),
])
def test_follow_ps1_sot_steps_dryrun(step, expect):
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
                        "-Step", step, "-DryRun"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [ln.removeprefix(">>> ") for ln in r.stdout.splitlines() if ln.startswith(">>> ")]
    assert cmds == [expect]

