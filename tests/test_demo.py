from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cvt.detect.base import Detections
from cvt.track.count import Line
from cvt.track.demo import _inverse_line, _live_counts, render_demo
from cvt.track.run import load_tracker_cfg

CFG = Path(__file__).resolve().parent.parent / "configs" / "trackers" / "bytetrack.yaml"
W, H, N, FPS = 640, 480, 20, 10.0
SIZE = 24


def _texture(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fine = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.5)
    img = (fine - fine.min()) / (fine.max() - fine.min())
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _centres(i: int) -> list[tuple[float, float, int]]:
    """(cx, cy, label): a motorbike moving down and a car moving up, both across y = 240."""
    return [(160.0, 120.0 + 8.0 * i, 0), (480.0, 360.0 - 8.0 * i, 1)]


def _write_video(path: Path) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened()
    bg = _texture()
    for i in range(N):
        im = bg.copy()
        for cx, cy, _ in _centres(i):
            cv2.rectangle(im, (int(cx - SIZE / 2), int(cy - SIZE / 2)),
                          (int(cx + SIZE / 2), int(cy + SIZE / 2)), (255, 255, 255), -1)
        vw.write(im)
    vw.release()


class FakeDetector:
    """Returns the fixed boxes of frame k on the k-th image it is given (calls in frame order)."""

    def __init__(self):
        self.seen = 0

    def predict_arrays(self, ims, imgsz=None):
        out = []
        for im in ims:
            assert im.shape == (H, W, 3)
            c = _centres(self.seen)
            self.seen += 1
            boxes = np.array([[x - SIZE / 2, y - SIZE / 2, x + SIZE / 2, y + SIZE / 2]
                              for x, y, _ in c], np.float32)
            out.append(Detections(boxes, np.full(len(c), 0.9, np.float32),
                                  np.array([lab for _, _, lab in c], np.int64)))
        return out


def _scene(**kw) -> dict:
    return {"seq_id": "demo", "set_name": "demo", "ref_frame": 0,
            "lines": [Line("mid", (0.0, 240.0), (640.0, 240.0))], **kw}


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def test_render_demo_writes_video_and_counts(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    out = tmp_path / "out" / "demo.mp4"
    res = render_demo(video, 0.0, N / FPS, FakeDetector(), load_tracker_cfg(CFG), _scene(), out,
                      fps_out=30.0, width_out=320, tmp_root=tmp_path / "cache")
    assert res["counts"] == {"mid": {"motorbike": {"pos": 1, "neg": 0},
                                     "car": {"pos": 0, "neg": 1}}}
    assert res["total"] == 2 and res["frames"] == N
    assert _frame_count(out) == N
    cap = cv2.VideoCapture(str(out))
    ok, im = cap.read()
    cap.release()
    assert ok and im.shape[1] == 320 and im.shape[0] == 240
    stills = [out.parent / f"demo_still{k}.png" for k in (1, 2, 3)]
    assert [Path(p) for p in res["stills"]] == stills
    assert all(p.exists() and cv2.imread(str(p)) is not None for p in stills)
    # the README still: the middle still as JPEG quality 88
    jpg = out.parent / "demo_still.jpg"
    assert Path(res["still"]) == jpg
    want = cv2.imencode(".jpg", cv2.imread(str(stills[1])), [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
    assert jpg.read_bytes() == want.tobytes()
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())


def test_disk_path_rereads_jpegs_with_the_same_counts(tmp_path: Path, monkeypatch):
    import cvt.track.demo as demo

    monkeypatch.setattr(demo, "MEM_BUDGET", 0)  # nothing fits: every frame is re-read from disk
    seen = []
    read = demo._read_segment

    def spy(*a, **k):
        out = read(*a, **k)
        seen.append(out[0])
        return out

    monkeypatch.setattr(demo, "_read_segment", spy)
    video = tmp_path / "in.mp4"
    _write_video(video)
    out = tmp_path / "out" / "demo.mp4"
    res = render_demo(video, 0.0, N / FPS, FakeDetector(), load_tracker_cfg(CFG), _scene(), out,
                      fps_out=30.0, width_out=320, tmp_root=tmp_path / "cache")
    assert len(seen) == 1 and seen[0].mem is None
    assert res["counts"] == {"mid": {"motorbike": {"pos": 1, "neg": 0},
                                     "car": {"pos": 0, "neg": 1}}}
    assert res["total"] == 2 and res["frames"] == N and _frame_count(out) == N
    assert not any((tmp_path / "cache").iterdir())


def test_failure_after_frames_are_written_leaves_no_temp_dir(tmp_path: Path):
    class Broken:
        def predict_arrays(self, ims, imgsz=None):
            jpgs = list((tmp_path / "cache").rglob("*.jpg"))
            assert len(jpgs) == N  # the segment is on disk when detection starts
            raise RuntimeError("detector exploded")

    video = tmp_path / "in.mp4"
    _write_video(video)
    with pytest.raises(RuntimeError, match="exploded"):
        render_demo(video, 0.0, N / FPS, Broken(), load_tracker_cfg(CFG), _scene(),
                    tmp_path / "d.mp4", tmp_root=tmp_path / "cache")
    assert (tmp_path / "cache").is_dir() and not any((tmp_path / "cache").iterdir())


def _fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def test_output_fps_defaults_to_the_source_fps(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    cfg, kw = load_tracker_cfg(CFG), {"width_out": 320, "tmp_root": tmp_path / "cache"}
    res = render_demo(video, 0.0, N / FPS, FakeDetector(), cfg, _scene(), tmp_path / "a.mp4", **kw)
    assert res["fps_out"] == pytest.approx(FPS) and _fps(tmp_path / "a.mp4") == pytest.approx(FPS)
    res = render_demo(video, 0.0, N / FPS, FakeDetector(), cfg, _scene(), tmp_path / "b.mp4",
                      fps_out=25.0, **kw)
    assert res["fps_out"] == 25.0 and _fps(tmp_path / "b.mp4") == pytest.approx(25.0)
    with pytest.raises(ValueError, match="fps_out"):
        render_demo(video, 0.0, N / FPS, FakeDetector(), cfg, _scene(), tmp_path / "c.mp4",
                    fps_out=0.0, **kw)


def test_keep_frames_keeps_the_segment_jpegs(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    render_demo(video, 0.0, N / FPS, FakeDetector(), load_tracker_cfg(CFG), _scene(),
                tmp_path / "demo.mp4", width_out=320, tmp_root=tmp_path / "cache",
                keep_frames=True)
    jpgs = list((tmp_path / "cache").rglob("*.jpg"))
    assert len(jpgs) == N


def test_detector_without_predict_arrays_raises(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    with pytest.raises(ValueError, match="predict_arrays"):
        render_demo(video, 0.0, 1.0, object(), load_tracker_cfg(CFG), _scene(), tmp_path / "d.mp4",
                    tmp_root=tmp_path / "cache")


def test_bad_inputs_raise(tmp_path: Path):
    video = tmp_path / "in.mp4"
    _write_video(video)
    cfg = load_tracker_cfg(CFG)
    kw = {"tmp_root": tmp_path / "cache"}
    with pytest.raises(ValueError, match="ref_frame"):
        render_demo(video, 0.0, N / FPS, FakeDetector(), cfg, _scene(ref_frame=99),
                    tmp_path / "d.mp4", **kw)
    with pytest.raises(ValueError, match="lines"):
        render_demo(video, 0.0, N / FPS, FakeDetector(), cfg, _scene(lines=[]),
                    tmp_path / "d.mp4", **kw)
    with pytest.raises(FileNotFoundError):
        render_demo(tmp_path / "missing.mp4", 0.0, 1.0, FakeDetector(), cfg, _scene(),
                    tmp_path / "d.mp4", **kw)
    with pytest.raises(ValueError, match="no frames"):
        render_demo(video, 100.0, 1.0, FakeDetector(), cfg, _scene(), tmp_path / "d.mp4", **kw)


def test_inverse_line_maps_reference_points_into_the_frame():
    # frame -> reference is a shift by (+10, +5); so reference (100, 50) is frame (90, 45)
    Hm = np.array([[1.0, 0, 10], [0, 1, 5], [0, 0, 1]])
    p1, p2 = _inverse_line(Line("l", (100.0, 50.0), (200.0, 50.0)), Hm)
    assert np.allclose(p1, (90, 45)) and np.allclose(p2, (190, 45))


def test_live_counts_are_cumulative_by_frame():
    import pandas as pd

    events = pd.DataFrame({"line": ["a", "a", "a"], "track_id": [1, 2, 3], "label": [1, 1, 0],
                           "frame": [2, 5, 5], "direction": ["pos", "neg", "pos"]})
    assert _live_counts(events, "a", 1) == {}
    assert _live_counts(events, "a", 2) == {1: {"pos": 1, "neg": 0}}
    c = _live_counts(events, "a", 5)
    assert c == {1: {"pos": 1, "neg": 1}, 0: {"pos": 1, "neg": 0}}
    assert _live_counts(events, "other", 5) == {}


def test_cli_demo_renders_with_the_opened_detector(tmp_path: Path, monkeypatch):
    import json

    import yaml
    from typer.testing import CliRunner

    import cvt.eval.run as er
    from cvt.cli import app

    video = tmp_path / "in.mp4"
    _write_video(video)
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump({"seq_id": "d", "set_name": "demo", "ref_frame": 0, "lines": [
        {"name": "mid", "p1": [0, 240], "p2": [640, 240]}]}), encoding="utf-8")
    seen = {}

    def fake_open(model, cfg, imgsz, conf, images, boxes):
        seen.update(model=model, imgsz=imgsz, conf=conf)
        return FakeDetector(), "sha"

    monkeypatch.setattr(er, "open_detector", fake_open)
    models = tmp_path / "models.yaml"
    models.write_text("{}", encoding="utf-8")
    out = tmp_path / "r" / "demo.mp4"
    args = ["demo", "--video", str(video), "--scene", str(scene), "--model", "m", "--tracker",
            str(CFG), "--out", str(out), "--start", "0", "--duration", str(N / FPS),
            "--width-out", "320", "--models", str(models)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert seen == {"model": "m", "imgsz": 1280, "conf": 0.05}
    assert out.exists() and (tmp_path / "r" / "demo_still3.png").exists()
    assert _fps(out) == pytest.approx(FPS)  # no --fps-out: the source video's rate
    counts = json.loads(r.stdout[: r.stdout.rindex("}") + 1])
    assert counts["mid"]["car"] == {"pos": 0, "neg": 1}

    bad = CliRunner().invoke(app, [*args[:4], str(tmp_path / "nope.yaml"), *args[5:]])
    assert bad.exit_code != 0 and "nope.yaml" in bad.output


def test_writer_prefers_media_foundation_on_windows(tmp_path: Path, monkeypatch):
    import cvt.track.demo as demo

    calls = []

    class FakeWriter:
        def __init__(self, path, *args):
            calls.append(args)
            self.ok = len(args) == 4  # (api, fourcc, fps, size): the Media Foundation form

        def isOpened(self):  # noqa: N802 (OpenCV name)
            return self.ok

        def release(self):
            pass

    monkeypatch.setattr(demo.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(demo.os, "name", "nt")
    _, code = demo._open_writer(tmp_path / "x.mp4", 30.0, (64, 48))
    assert code == "H264 (Media Foundation)" and calls[0][0] == demo.cv2.CAP_MSMF
    monkeypatch.setattr(demo.os, "name", "posix")
    calls.clear()
    with pytest.raises(RuntimeError):  # only the FFmpeg forms are tried, and the fake refuses them
        demo._open_writer(tmp_path / "y.mp4", 30.0, (64, 48))
    assert all(len(a) == 3 for a in calls)
