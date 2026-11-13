from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from cvt import cli
from cvt.cli import _demo_pick
from cvt.follow import pipeline as fp
from cvt.follow.demo import (
    SUMMARY_COLS,
    demo_plan,
    load_demo_params,
    pick_auto,
    render_follow,
    render_follows,
    run_summary,
    select_target,
)
from cvt.follow.gimbal import GimbalParams
from cvt.follow.lock import LockParams
from cvt.track.run import load_tracker_cfg
from tests.test_demo import CFG, FPS, FakeDetector, N, _centres, _write_video

LOCK = LockParams(2.0, 0.7, 1.0, True, True, "none")
GIMBAL = GimbalParams(kp=1.0, kd=0.2, target_frac=0.25)
COLS = ["frame", "t", "source", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "zoom",
        "pan_rate", "tilt_rate", "zoom_rate", "yaw_rate", "pitch_rate", "clamped"]


def _tracks() -> pd.DataFrame:
    return pd.DataFrame({
        "frame": [0, 0, 0], "track_id": [1, 2, 3], "label": [1, 1, 1],
        "x1": [0.0, 100.0, 400.0], "y1": [0.0, 100.0, 400.0],
        "x2": [200.0, 120.0, 420.0], "y2": [200.0, 120.0, 420.0],
    })


def test_click_selects_containing_box():
    # inside the big box 1 and the small box 2: the smaller (more specific) box wins
    assert select_target(_tracks(), 0, (110, 110))[0] == 2
    assert select_target(_tracks(), 0, (10, 10))[0] == 1
    # outside every box: nearest box centre (box 3 centre = 410, 410)
    tid, box = select_target(_tracks(), 0, (300, 300))
    assert tid == 3 and list(box) == [400.0, 400.0, 420.0, 420.0]
    with pytest.raises(ValueError, match="no track"):
        select_target(_tracks().iloc[:0], 0, (1, 1))


def test_render_follow_writes_video_and_command_log(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    out = tmp_path / "out" / "follow.mp4"
    cx, cy, _ = _centres(0)[1]  # the car
    res = render_follow(video, (cx, cy), 0.0, N / FPS, out, detector=FakeDetector(),
                        tracker_cfg=load_tracker_cfg(CFG), lock=LOCK, gimbal=GIMBAL,
                        hfov_deg=84.0, tmp_root=tmp_path / "cache")
    assert Path(res) == out
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, im = cap.read()
    cap.release()
    assert n == N and ok
    assert im.shape[1] == 1920 and im.shape[0] == 720  # 960 px full view + 960x540 follow view
    log = pd.read_csv(out.with_suffix(".csv"))
    assert list(log.columns) == COLS
    assert len(log) == N
    assert (log["source"] == "track").all() and (log["track_id"] == log["track_id"].iloc[0]).all()
    assert log["zoom"].max() > 1.0  # the small target makes the virtual gimbal zoom in
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())


def test_load_demo_params_requires_tuned(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="-Step tune"):
        load_demo_params(Path("configs/follow.yaml"), tmp_path / "missing.yaml")
    t = tmp_path / "t.yaml"
    cfg = fp.load_follow_cfg(Path("configs/follow.yaml"))
    tname = load_tracker_cfg(Path(cfg["tracker"]))["name"]
    t.write_text(f"set: MOT-tune\ndetector: {cfg['detector']}\ntracker: {tname}\n"
                 "lock: {coast_s: 2, reid_thr: 0.7, gate_growth: 1.0}\n"
                 "gimbal: {kp: 1, kd: 0.2, target_frac: 0.25}\n", encoding="utf-8")
    lock, gim, hfov = load_demo_params(Path("configs/follow.yaml"), t)
    assert lock.reacquire and lock.use_mil and lock.appearance == "none" and lock.coast_s == 2
    assert gim.kp == 1 and hfov == 84.0


def test_missing_video(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        render_follow(tmp_path / "no.mp4", (1, 1), 0.0, 1.0, tmp_path / "o.mp4",
                      detector=FakeDetector(), tracker_cfg=load_tracker_cfg(CFG), lock=LOCK,
                      gimbal=GIMBAL, hfov_deg=84.0, tmp_root=tmp_path / "c")



def _auto_tracks() -> pd.DataFrame:
    """Frame 0 of a 100x100 frame: car 1 (2 frames, still), car 2 (3 frames, moving), car 3 (3
    frames, still), motorbike 4 (1 frame), car 5 at the edge (5 frames), pedestrian 6."""
    rows = []
    def add(tid, label, frames, x0, dx=0.0):
        for f in frames:
            x = x0 + dx * f
            rows.append((f, tid, label, x - 2, 48.0, x + 2, 52.0))
    add(1, 1, [0, 1], 50.0)
    add(2, 1, [0, 1, 2], 30.0, dx=5.0)
    add(3, 1, [0, 1, 2], 70.0)
    add(4, 0, [0], 40.0)
    add(5, 1, [0, 1, 2, 3, 4], 5.0)        # centre x = 5 < 10 % of the width: excluded
    add(6, 5, [0, 1, 2, 3, 4, 5], 60.0)    # pedestrian: not a vehicle
    return pd.DataFrame(rows, columns=["frame", "track_id", "label", "x1", "y1", "x2", "y2"])


def test_pick_auto_prefers_long_then_moving_tracks_away_from_the_edge():
    t = _auto_tracks()
    tid, box = pick_auto(t, 0, (100, 100))
    assert tid == 2 and list(box) == [28.0, 48.0, 32.0, 52.0]  # 3 frames and moving beats 3
    assert pick_auto(t, 0, (100, 100), label=0)[0] == 4         # the only motorbike
    assert pick_auto(t, 0, (100, 100), label=3) is None          # no truck
    assert pick_auto(t, 0, (100, 100), margin=0.0)[0] == 5       # the edge car once allowed
    assert pick_auto(t[t.track_id != 2], 0, (100, 100))[0] == 3


def test_run_summary_counts_sources_reacquisitions_and_travel():
    log = pd.DataFrame({
        "source": ["track", "track", "lost", "mil", "track", "track"],
        "track_id": [7, 7, None, None, 9, 9],
        "x1": [0.0, 30.0, None, 30.0, 30.0, 30.0], "y1": [0.0, 40.0, None, 40.0, 40.0, 40.0],
        "x2": [0.0, 30.0, None, 30.0, 30.0, 30.0], "y2": [0.0, 40.0, None, 40.0, 40.0, 40.0],
        "zoom": [1.0, 2.0, 2.0, 2.0, 2.0, 3.0], "clamped": [False, False, True, True, False, False],
    })
    s = run_summary(log, (60, 80))
    assert s["frames"] == 6 and s["frac_track"] == 4 / 6 and s["frac_mil"] == 1 / 6
    assert s["frac_lost"] == 1 / 6 and s["reacquisitions"] == 1
    assert s["travel_diag"] == pytest.approx(50.0 / 100.0)  # (0,0) -> (30,40), then still
    assert s["clamped_frac"] == 2 / 6 and s["mean_zoom"] == 2.0
    # last held box (30, 40) is mid-frame: a drop, not an exit
    assert s["frac_lost_drop"] == 1 / 6 and s["frac_lost_exit"] == 0.0 and s["drops"] == 1
    # held at the right edge, then lost twice in a row: an exit; lost from the start: a drop
    edge = pd.DataFrame({
        "source": ["lost", "track", "lost", "lost"], "track_id": [None, 3, None, None],
        "x1": [None, 50.0, None, None], "y1": [None, 30.0, None, None],
        "x2": [None, 59.5, None, None], "y2": [None, 40.0, None, None],
        "zoom": [1.0] * 4, "clamped": [False] * 4,
    })
    s = run_summary(edge, (60, 80))
    assert s["frac_lost_exit"] == 2 / 4 and s["frac_lost_drop"] == 1 / 4 and s["drops"] == 1


def test_render_follows_one_pass_a_different_vehicle_per_pick(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    det = FakeDetector()
    outs = [tmp_path / "o" / f"{k}.mp4" for k in ("car", "motorbike", "truck", "any")]
    rows = render_follows(video, ["car", "motorbike", "truck", "any"], 0.0, N / FPS, outs,
                          detector=det, tracker_cfg=load_tracker_cfg(CFG), lock=LOCK,
                          gimbal=GIMBAL, hfov_deg=84.0, tmp_root=tmp_path / "cache")
    assert det.seen == N  # detected once for every pick
    assert [r["status"] for r in rows] == ["ok", "ok", "no candidate", "no candidate"]
    assert [r["label"] for r in rows[:2]] == ["car", "motorbike"]
    assert rows[0]["track_id"] != rows[1]["track_id"]  # 'any' finds nothing left to lock
    assert all(o.exists() and o.with_suffix(".csv").exists() for o in outs[:2])
    assert not outs[2].exists() and not outs[3].exists()
    assert rows[0]["frac_track"] == 1.0 and rows[0]["travel_diag"] > 0.1
    assert set(rows[0]) <= set(SUMMARY_COLS)
    with pytest.raises(ValueError, match="no track to lock"):
        render_follow(video, "truck", 0.0, N / FPS, tmp_path / "t.mp4", detector=FakeDetector(),
                      tracker_cfg=load_tracker_cfg(CFG), lock=LOCK, gimbal=GIMBAL,
                      hfov_deg=84.0, tmp_root=tmp_path / "cache")
    with pytest.raises(ValueError, match="unknown class"):
        render_follow(video, "tank", 0.0, N / FPS, tmp_path / "t.mp4", detector=FakeDetector(),
                      tracker_cfg=load_tracker_cfg(CFG), lock=LOCK, gimbal=GIMBAL,
                      hfov_deg=84.0, tmp_root=tmp_path / "cache")


def test_demo_plan_takes_the_middle_segment(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_video(video)  # N / FPS = 2 s
    (p,) = demo_plan([video], 1.0, ["car", "bus"], tmp_path / "out")
    assert p["t0_s"] == 0.5 and p["picks"] == ["car", "bus"]
    assert p["outs"] == [tmp_path / "out" / "clip_car.mp4", tmp_path / "out" / "clip_bus.mp4"]
    assert demo_plan([video], 5.0, ["car"], tmp_path)[0]["t0_s"] == 0.0  # shorter: whole clip


def test_demo_pick_parsing():
    assert _demo_pick("auto") == "any" and _demo_pick(" Car ") == "car"
    assert _demo_pick("1920,1080") == (1920.0, 1080.0)
    with pytest.raises(typer.BadParameter):
        _demo_pick("pedestrian")


def test_demo_all_cli_writes_summary_and_resumes(tmp_path: Path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    _write_video(clips / "a.mp4")
    (clips / "notes.txt").write_text("not a video", encoding="utf-8")
    calls = []

    def fake_detector(config, models, imgsz, conf):
        calls.append(1)
        cfg = fp.load_follow_cfg(config)
        return {**cfg, "tracker": str(CFG)}, FakeDetector()

    monkeypatch.setattr(cli, "_follow_detector", fake_detector)
    out = tmp_path / "out"
    args = ["follow", "demo-all", "--videos", str(clips), "--classes", "car,truck",
            "--seconds", str(N / FPS), "--out-dir", str(out)]
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    s = pd.read_csv(out / "summary.csv")
    assert list(s.columns) == SUMMARY_COLS
    assert list(s.pick) == ["car", "truck"] and list(s.status) == ["ok", "no candidate"]
    assert (out / "a_car.mp4").exists() and len(calls) == 1
    r = CliRunner().invoke(cli.app, args)  # resume: nothing left, no detector opened
    assert r.exit_code == 0 and len(calls) == 1 and "1 already done" in r.output
    r = CliRunner().invoke(cli.app, [*args[:4], "--classes", "car,pedestrian"])
    assert r.exit_code != 0 and "classes must be among" in r.output
