"""Appearance features for cached detections: letterbox, ROI pooling, resumable CLI."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from typer.testing import CliRunner

import cvt.train.job as job
from cvt.cli import app
from cvt.follow.features import (
    VEHICLE_IDS,
    features_path,
    letterbox_boxes,
    load_features,
    pool_features,
)
from tests.test_track_pipeline import TAG, _manifest, write_det_cache

C, IMGSZ = 4, 64


def test_letterbox_boxes_matches_ultralytics_letterbox():
    # 3840x2160 at 1536: r = 0.4, content 1536x864, pad 336 top and bottom
    out = letterbox_boxes(np.array([[0, 0, 3840, 2160]], np.float32), 3840, 2160, 1536)
    np.testing.assert_allclose(out, [[0, 336, 1536, 1200]])
    out = letterbox_boxes(np.array([[10, 20, 30, 40]], np.float32), 100, 50, 200)  # r = 2
    np.testing.assert_allclose(out, [[20, 90, 60, 130]])


def test_pool_features_unit_norm_and_shape():
    g = torch.Generator().manual_seed(0)
    fmap = torch.randn(1, 16, 40, 40, generator=g)
    boxes = np.array([[0, 0, 80, 80], [40, 40, 200, 120], [100, 100, 300, 300]], np.float32)
    f = pool_features(fmap, boxes, 8)
    assert f.shape == (3, 16) and f.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5)
    prof = torch.arange(1, 17, dtype=torch.float32)
    const = torch.zeros(1, 16, 40, 40)
    const[:, :, 5:30, 5:30] = prof[None, :, None, None]
    f = pool_features(const, np.array([[80, 80, 200, 200]], np.float32), 8)
    np.testing.assert_allclose(f[0], (prof / prof.norm()).numpy(), atol=1e-5)


def _install_fake_ultralytics(monkeypatch, calls):
    class Head(torch.nn.Module):
        stride = torch.tensor([8.0, 16.0, 32.0])

        def forward(self, x):
            return x

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(Head())
            ys = torch.arange(IMGSZ // 8, dtype=torch.float32)[:, None]
            xs = torch.arange(IMGSZ // 8, dtype=torch.float32)[None, :]
            self.fm = torch.stack([ys + xs + c + 1 for c in range(C)])[None]

        def forward(self, x):
            assert x.shape == (1, 3, IMGSZ, IMGSZ)
            return self.model[-1]([self.fm])

    class FakeYOLO:
        def __init__(self, weights, task=None):
            calls.append(weights)
            self.model = Net()

    ul = types.ModuleType("ultralytics")
    ul.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    monkeypatch.setattr(job, "no_network", lambda: calls.append("offline"))


def test_features_cli_resumes_and_filters(tmp_path: Path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    weights = tmp_path / "absent.pt"
    weights.write_bytes(b"w")
    os.utime(weights, (1, 1))  # older than the detection caches
    low = write_det_cache(p, "MOT-tune")
    dt = pd.read_parquet(low)
    dt.loc[0, "score"] = 0.05  # below --min-score
    dt.to_parquet(low, index=False)
    calls: list = []
    _install_fake_ultralytics(monkeypatch, calls)
    out = tmp_path / "feats"
    args = ["follow", "features", "--weights-model", "det", "--boxes-of", "det",
            "--imgsz", str(IMGSZ), "--sets", "MOT-tune,T-MOT", "--min-score", "0.1",
            "--models", str(p["models"]), "--manifest", str(p["manifest"]),
            "--hash-cache", str(p["hash"]), "--pred-cache", str(p["preds"]),
            "--out-dir", str(out)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert calls == ["offline", str(weights)]  # offline guard first, one model load for both sets

    for name in ("MOT-tune", "T-MOT"):
        path = features_path(out, "det", TAG, name)
        assert path.exists()
        row, feat = load_features(path)
        cache = pd.read_parquet(next(Path(p["preds"]).glob(f"det_{TAG}_{name}_*.parquet")))
        want = np.flatnonzero(cache.label.isin(VEHICLE_IDS).to_numpy() & (cache.score >= 0.1))
        assert row.dtype == np.int64 and feat.dtype == np.float16
        np.testing.assert_array_equal(row, want)
        assert feat.shape == (len(want), C)
        np.testing.assert_allclose(np.linalg.norm(feat.astype(np.float32), axis=1), 1.0, atol=2e-3)
    assert 0 not in load_features(features_path(out, "det", TAG, "MOT-tune"))[0]
    assert len(load_features(features_path(out, "det", TAG, "T-MOT"))[0]) < len(
        pd.read_parquet(next(Path(p["preds"]).glob(f"det_{TAG}_T-MOT_*.parquet"))))  # pedestrians out

    calls.clear()
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert calls == []  # every set skipped: the model is not even loaded
    r = CliRunner().invoke(app, [*args[:-2], "--out-dir", str(out), "--min-score", "0.5"])
    assert r.exit_code == 0 and calls == ["offline", str(weights)]  # meta changed: recomputed


def test_features_cli_missing_weights_is_one_line_error(tmp_path: Path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    r = CliRunner().invoke(app, [
        "follow", "features", "--weights-model", "det", "--boxes-of", "det", "--sets", "T-MOT",
        "--models", str(p["models"]), "--manifest", str(p["manifest"]),
        "--hash-cache", str(p["hash"]), "--pred-cache", str(p["preds"]),
        "--out-dir", str(tmp_path / "f")])
    assert r.exit_code == 1 and "weights not found" in r.output


def test_features_empty_set_keeps_the_channel_dimension(tmp_path: Path, monkeypatch):
    p = _manifest(tmp_path, monkeypatch)
    weights = tmp_path / "absent.pt"
    weights.write_bytes(b"w")
    os.utime(weights, (1, 1))
    write_det_cache(p, "T-MOT")
    calls: list = []
    _install_fake_ultralytics(monkeypatch, calls)
    out = tmp_path / "feats"
    r = CliRunner().invoke(app, [
        "follow", "features", "--weights-model", "det", "--boxes-of", "det",
        "--imgsz", str(IMGSZ), "--sets", "T-MOT", "--min-score", "2.0",  # no detection passes
        "--models", str(p["models"]), "--manifest", str(p["manifest"]),
        "--hash-cache", str(p["hash"]), "--pred-cache", str(p["preds"]), "--out-dir", str(out)])
    assert r.exit_code == 0, r.output
    row, feat = load_features(features_path(out, "det", TAG, "T-MOT"))
    assert len(row) == 0 and feat.shape == (0, C) and feat.dtype == np.float16


def test_extract_set_casts_each_frame_to_float16(monkeypatch):
    from cvt.follow import features as fm

    class Ext:
        imgsz, stride, channels = IMGSZ, 8, C

        def fmap(self, img):
            return torch.ones(1, C, IMGSZ // 8, IMGSZ // 8)

    monkeypatch.setattr("cv2.imread", lambda path: np.zeros((48, 64, 3), np.uint8))
    s = types.SimpleNamespace(name="x", frames=pd.DataFrame(
        {"seq_id": ["a", "a"], "frame": [1, 2], "path": ["p1", "p2"]}))
    dets = pd.DataFrame({"seq_id": ["a", "a", "a"], "frame": [1, 1, 2],
                         "x1": [0.0, 10, 5], "y1": 0.0, "x2": [8.0, 30, 20], "y2": 8.0,
                         "score": 0.9, "label": VEHICLE_IDS[0]})
    casts: list = []

    class Pooled(np.ndarray):
        def astype(self, dtype, *a, **kw):
            casts.append(np.dtype(dtype))
            return np.asarray(self).astype(dtype, *a, **kw)

    real = fm.pool_features
    monkeypatch.setattr(fm, "pool_features", lambda *a: real(*a).view(Pooled))
    row, feat = fm.extract_set(Ext(), s, dets, 0.1, lambda m: None)
    assert casts == [np.float16, np.float16]  # each frame's block is cast as it is collected
    assert feat.dtype == np.float16 and feat.shape == (3, C)
    np.testing.assert_array_equal(row, [0, 1, 2])
    none_row, none_feat = fm.extract_set(Ext(), s, dets, 2.0, lambda m: None)
    assert none_feat.shape == (0, C) and none_feat.dtype == np.float16 and len(none_row) == 0
