from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cvt.analytics import telemetry
from cvt.analytics.telemetry import (
    camera_model,
    car_scale,
    clip_scale,
    hfov_deg,
    metres_per_pixel,
    read_telemetry,
    ref_second,
)
from cvt.track.run import DET_COLS

REC = ("F/2.8, SS 1562.21, ISO 100, EV 0, DZOOM 1.000, GPS (106.6152, 10.7517, 25), "
       "D 302.15m, H {h:.2f}m, H.S 0.10m/s, V.S -0.10m/s ")


def _video(tmp_path, heights, pad=1000, camera=b"FC7303", name="v.mp4"):
    """A fake mp4: junk, the camera tag, then one telemetry record per second between junk."""
    p = tmp_path / name
    body = b"\x00" * pad + camera + b"".join(
        b"\x01" * pad + REC.format(h=h).encode() for h in heights)
    p.write_bytes(body)
    return p


def test_read_telemetry_and_camera(tmp_path):
    tel = read_telemetry(_video(tmp_path, [60.8, 61.0, 61.3]))
    assert list(tel.t_s) == [0, 1, 2] and list(tel.height_m) == [60.8, 61.0, 61.3]
    assert tel.lon[0] == 106.6152 and tel.zoom[0] == 1.0 and tel.vs_ms[0] == -0.1
    assert camera_model(_video(tmp_path, [50.0])) == "FC7303"


def test_records_across_read_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "CHUNK", 257)   # records straddle chunk boundaries
    assert len(read_telemetry(_video(tmp_path, [50.0] * 40, pad=300))) == 40


def test_scale_formula():
    assert hfov_deg(83.0, (4, 3)) == pytest.approx(70.58, abs=0.02)
    assert metres_per_pixel(61.0, hfov_deg(83.0, (4, 3)), 1920) == pytest.approx(0.04497, abs=1e-4)


def test_car_scale_uses_axis_aligned_confident_cars():
    # label 1 = car; long side 100 px for the two that count
    dets = pd.DataFrame({"frame": 0, "x1": [0, 0, 0, 0], "y1": [0, 0, 0, 0],
                         "x2": [100, 40, 100, 70], "y2": [40, 100, 40, 70],
                         "score": [0.9, 0.8, 0.3, 0.9], "label": [1, 1, 1, 1]})
    mpp, n = car_scale(dets, 4.5)       # low score and the square (diagonal) box are dropped
    assert n == 2 and mpp == pytest.approx(0.045)
    assert np.isnan(car_scale(dets.iloc[:0], 4.5)[0])


def test_ref_second_and_clip_scale(tmp_path):
    cfg = yaml.safe_load(Path("configs/analytics.yaml").read_text(encoding="utf-8"))
    v = _video(tmp_path, [70.8, 70.5, 50.9, 50.9, 50.5, 51.0])
    assert ref_second(read_telemetry(v), 0.5) == 2          # median 50.95
    s = clip_scale(v, pd.DataFrame(columns=DET_COLS), 1920, cfg)
    assert s["source"] == "telemetry" and s["ref_s"] == 2 and s["camera"] == "FC7303"
    assert s["mpp"] == pytest.approx(metres_per_pixel(50.9, hfov_deg(83.0, (4, 3)), 1920))


def test_no_telemetry_or_unknown_camera_falls_back(tmp_path):
    cfg = yaml.safe_load(Path("configs/analytics.yaml").read_text(encoding="utf-8"))
    cars = pd.DataFrame({"frame": 0, "x1": [0.0], "y1": [0.0], "x2": [90.0], "y2": [40.0],
                         "score": [0.9], "label": [1]})
    for v in (_video(tmp_path, [], name="a.mp4"),
              _video(tmp_path, [50.0], camera=b"FC9999", name="b.mp4")):
        s = clip_scale(v, cars, 1920, cfg)
        assert s["source"] == "car-length (uncalibrated)" and s["mpp"] == pytest.approx(0.05)
    assert np.isnan(clip_scale(_video(tmp_path, []), cars.iloc[:0], 1920, cfg)["mpp"])
