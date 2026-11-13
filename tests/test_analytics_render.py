from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cvt.analytics.pipeline import analyze_clip, detect_clip, track_clip
from cvt.analytics.render import render_clip, still
from tests.test_analytics_pipeline import _cfg
from tests.test_demo import FakeDetector, N, _write_video


def test_render_clip_writes_a_full_length_video(tmp_path):
    cfg = _cfg(tmp_path)
    video = Path(cfg["clips_dir"]) / "clip.mp4"
    _write_video(video)
    cache = tmp_path / "cache"
    detect_clip(video, cfg, FakeDetector(), cache, print)
    track_clip(video, cfg, cache, print)
    r = analyze_clip(video, cfg, cache)
    out = render_clip(video, r, tmp_path / "o" / "clip_analytics.mp4")
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N
    assert (int(cap.get(3)), int(cap.get(4))) == (640, 480)
    ok, im = cap.read()
    cap.release()
    src = cv2.VideoCapture(str(video)).read()[1]
    assert ok and np.abs(im.astype(int) - src.astype(int)).sum() > 0  # something was drawn
    s = still(video, r, 5)
    assert s.shape == (480, 640, 3)
    short = render_clip(video, r, tmp_path / "o" / "s.mp4", max_frames=7)
    assert int(cv2.VideoCapture(str(short)).get(cv2.CAP_PROP_FRAME_COUNT)) == 7


def test_scale_line_and_exit_counts():
    import pandas as pd

    from cvt.analytics.pipeline import ClipResult
    from cvt.analytics.render import _exits, scale_line

    assert scale_line({"mpp": 0.045, "source": "telemetry", "height_med": 61.2}) == (
        "4.5 cm/px, telemetry 61 m"
    )
    assert scale_line({"mpp": 0.045, "source": "car_length", "height_med": np.nan}) == (
        "4.5 cm/px, car length (uncalibrated)"
    )
    assert scale_line({"mpp": np.nan, "source": "car_length", "height_med": np.nan}) == (
        "uncalibrated"
    )
    ep = pd.DataFrame({
        "track_id": [1, 2, 3], "label": [1, 1, 2], "first": [0, 0, 0], "last": [30, 30, 40],
        "entry": [0.1, 0.1, 0.1], "exit": [0.6, np.nan, 0.9], "complete": [True, False, True],
    })
    r = ClipResult(*([None] * 8), ep, np.array([0.1, 0.6]), *([None] * 5))
    assert _exits(r, 0.06) == {30: [("A", "B", "car")]}   # 3 ends 0.3 from any arm: not counted


def test_stopped_label_counts_from_when_the_stop_began():
    import pandas as pd

    from cvt.analytics.render import _stop_starts, _stopped_for

    ev = pd.DataFrame({"track_id": [3, 3, 4], "kind": ["stopped", "stopped", "wrong_way"],
                       "t_start": [50.0, 10.0, 0.0]})
    starts = _stop_starts(ev)
    assert set(starts) == {3}
    assert _stopped_for(starts, 3, 16.0) == 6.0 and _stopped_for(starts, 3, 60.0) == 10.0
    assert _stopped_for(starts, 4, 5.0) == 0.0 and _stop_starts(ev.iloc[:0]) == {}


def test_cli_render_reports_missing_caches(tmp_path):
    from typer.testing import CliRunner

    from cvt.cli import app

    cfg = _cfg(tmp_path)
    _write_video(Path(cfg["clips_dir"]) / "clip.mp4")
    import yaml
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    res = CliRunner().invoke(app, ["analytics", "render", "--config", str(path),
                                   "--cache", str(tmp_path / "none"), "--out", str(tmp_path)])
    assert res.exit_code == 1 and "error:" in res.output
