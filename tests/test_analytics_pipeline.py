from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from cvt.analytics import pipeline
from cvt.analytics.pipeline import (
    analyze_clip,
    detect_clip,
    interp_homographies,
    load_cfg,
    paths,
    track_clip,
)
from cvt.cli import app
from tests.test_demo import CFG, FakeDetector, N, _write_video


def _cfg(tmp_path, **over):
    cfg = load_cfg(Path("configs/analytics.yaml"))
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(exist_ok=True)
    return {**cfg, "clips_dir": str(clips_dir), "tracker": str(CFG), "stab_every": 5,
            "labels_dir": str(tmp_path / "labels"), **over}


def test_load_cfg_names_a_missing_key(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("detector: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracker"):
        load_cfg(p)


def test_interp_homographies():
    Hk = np.stack([np.eye(3), np.array([[1, 0, 10.0], [0, 1, 0], [0, 0, 1]]) * 2.0])
    H, valid = interp_homographies(Hk, np.array([True, True]), np.array([0, 5]), 8)
    assert H[2, 0, 2] == pytest.approx(4.0) and H[5, 2, 2] == pytest.approx(1.0)
    assert H[7, 0, 2] == pytest.approx(10.0) and valid.all()
    _, v = interp_homographies(Hk, np.array([True, False]), np.array([0, 5]), 8)
    assert v[0] and not v[1:].any()


SCALE = {"camera": None, "height_med": float("nan"), "height_min": float("nan"),
         "height_max": float("nan"), "zoom": float("nan"), "ref_s": 0, "mpp_telemetry": float("nan"),
         "mpp_car": 0.05, "n_cars": 9, "mpp": 0.05, "source": "car-length (uncalibrated)"}


def test_detect_track_analyze_on_a_synthetic_clip(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    cache = tmp_path / "cache"
    det = FakeDetector()
    dets = detect_clip(video, cfg, det, cache, print)
    assert det.seen == N and set(dets.frame) == set(range(N))
    p = paths(cache, cfg, "clip")
    assert json.loads(p["dets_meta"].read_text())["n_frames"] == N
    again = FakeDetector()
    detect_clip(video, cfg, again, cache, print)
    assert again.seen == 0                                    # resumed from the cache
    track_clip(video, cfg, cache, print)
    assert p["tracks"].exists() and p["stab"].exists() and not p["frames_dir"].exists()
    r = analyze_clip(video, cfg, cache)       # no telemetry, square boxes: no scale at all
    assert np.isnan(r.scale["mpp"]) and r.kin.track_id.nunique() == 2
    assert r.kin.speed_kmh.isna().all() and r.maps == {} and not r.scores["both"].any()
    assert r.od.empty and r.od_summary["arms"] == 0           # two tracks: no arm
    monkeypatch.setattr(pipeline, "clip_scale", lambda *a, **k: dict(SCALE))
    r = analyze_clip(video, cfg, cache)
    assert r.kin.speed_kmh.notna().any() and len(r.scores["both"]) == N
    assert set(r.maps) == {"density", "speed", "stationary"}


def test_interrupted_track_leaves_no_half_cache(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    cache = tmp_path / "cache"
    detect_clip(video, cfg, FakeDetector(), cache, print)
    monkeypatch.setattr(pipeline, "reference_homographies",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        track_clip(video, cfg, cache, print)
    p = paths(cache, cfg, "clip")
    assert not p["tracks_meta"].exists() and not p["frames_dir"].exists()
    monkeypatch.undo()
    track_clip(video, cfg, cache, print)
    assert p["tracks_meta"].exists()


def test_missing_caches_give_a_hint(tmp_path):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    with pytest.raises(FileNotFoundError, match="-Step detect"):
        track_clip(video, cfg, tmp_path / "cache", print)
    with pytest.raises(FileNotFoundError, match="-Step"):
        analyze_clip(video, cfg, tmp_path / "cache")


def test_cli_analyze_writes_per_clip_outputs(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    cache = tmp_path / "cache"
    detect_clip(video, cfg, FakeDetector(), cache, print)
    track_clip(video, cfg, cache, print)
    monkeypatch.setattr(pipeline, "clip_scale", lambda *a, **k: dict(SCALE))
    cfg_path = tmp_path / "analytics.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["analytics", "analyze", "--config", str(cfg_path), "--cache",
                                 str(cache), "--out", str(out)])
    assert r.exit_code == 0, r.output
    raw = (out / "clip" / "scale.json").read_bytes()
    assert b"\r" not in raw and b"NaN" not in raw
    assert json.loads(raw)["height_med"] is None
    assert {f.name for f in (out / "clip").iterdir()} >= {
        "od.csv", "events.csv", "scale.json", "maps_speed.jpg", "maps_density.jpg",
        "maps_stationary.jpg"}
    r = CliRunner().invoke(app, ["analytics", "analyze", "--config", str(cfg_path), "--cache",
                                 str(cache), "--out", str(out), "--clips", "nope"])
    assert r.exit_code != 0 and "nope" in r.output


def test_stale_frames_removed_and_reference_jpeg_kept(tmp_path):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    cache = tmp_path / "cache"
    detect_clip(video, cfg, FakeDetector(), cache, print)
    stale = cache / "frames" / "other_clip"
    stale.mkdir(parents=True)
    (stale / "000000.jpg").write_bytes(b"x")
    track_clip(video, cfg, cache, print)
    p = paths(cache, cfg, "clip")
    assert not stale.exists() and p["ref_jpg"].exists()
    assert cv2.imread(str(p["ref_jpg"])).shape == (480, 640, 3)


def test_analyze_refuses_tracks_of_other_detections(tmp_path):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    cache = tmp_path / "cache"
    detect_clip(video, cfg, FakeDetector(), cache, print)
    track_clip(video, cfg, cache, print)
    analyze_clip(video, cfg, cache)
    p = paths(cache, cfg, "clip")
    meta = json.loads(p["dets_meta"].read_text())
    p["dets_meta"].write_text(json.dumps({**meta, "conf": 0.5}))
    with pytest.raises(FileNotFoundError, match="-Step track"):
        analyze_clip(video, cfg, cache)
    p["tracks_meta"].unlink()
    with pytest.raises(FileNotFoundError, match="-Step"):
        analyze_clip(video, cfg, cache)
