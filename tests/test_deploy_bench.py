"""End-to-end latency benchmark (fakes only: no GPU, no Ultralytics models)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cvt.data.manifest import IMAGE_COLS

NAME = "m@640-fp16"


class Recorder:
    """Fake clock + sync + detector sharing one event log; predict advances the fake clock."""

    def __init__(self, durations_ms: list[float]):
        self.events: list[str] = []
        self.now = 0.0
        self.durations = list(durations_ms)  # one per predict call (warm-up calls first)

    def clock(self) -> float:
        self.events.append("clock")
        return self.now

    def sync(self) -> None:
        self.events.append("sync")

    def predict_arrays(self, ims, imgsz=None):
        assert len(ims) == 1
        self.events.append("predict")
        self.now += self.durations.pop(0) / 1000.0
        return []


def test_bench_excludes_warmup_and_synchronises():
    from cvt.deploy.bench import time_detector

    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(5)]
    rec = Recorder([900.0, 900.0, 900.0] + [10.0, 20.0, 30.0, 40.0, 50.0])  # slow warm-up
    res = time_detector(rec, frames, warmup=3, sync=rec.sync, clock=rec.clock)
    assert rec.durations == []
    assert rec.events[:3] == ["predict"] * 3  # warm-up: no clock, no sync
    timed = rec.events[3:]
    assert timed == ["sync", "clock", "predict", "sync", "clock"] * 5
    assert res["median_ms"] == pytest.approx(30.0)
    assert res["p90_ms"] == pytest.approx(46.0)  # linear percentile of 10..50
    assert res["fps"] == pytest.approx(1000.0 / 30.0)
    assert res["n_frames"] == 5 and res["warmup"] == 3
    with pytest.raises(ValueError, match="no frames"):
        time_detector(rec, [], 1, rec.sync, rec.clock)
    with pytest.raises(ValueError, match="warmup"):
        time_detector(rec, frames, -1, rec.sync, rec.clock)


def test_time_detector_warmup_cycles_frames():
    from cvt.deploy.bench import time_detector

    seen = []

    class Det:
        def predict_arrays(self, ims):
            seen.append(int(ims[0][0, 0, 0]))
            return []

    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(3)]
    ticks = iter(range(100))
    time_detector(Det(), frames, warmup=4, sync=lambda: None, clock=lambda: next(ticks) * 0.001)
    assert seen == [0, 1, 2, 0] + [0, 1, 2]


def _images(extra_sets: bool = True) -> pd.DataFrame:
    rows = []
    for i in [7, 3, 11, 0, 5, 9, 1, 2, 8, 4, 10, 6]:  # unsorted on purpose
        rows.append((f"vn/{i:02d}", f"/frames/vn_{i:02d}.jpg", 8, 8, "uit", "s", "test", "T-VN"))
    if extra_sets:
        rows += [("vd/00", "/frames/vd_00.jpg", 8, 8, "vd", "s", "test", "T-VD"),
                 ("aa/00", "/frames/aa_00.jpg", 8, 8, "vd", "s", "train", "VD-train")]
    return pd.DataFrame(rows, columns=IMAGE_COLS)


def test_bench_frames_every_30th_sorted():
    from cvt.deploy.bench import bench_frames, frames_hash

    im = _images()
    assert bench_frames(im, every=3, n=3) == [
        "/frames/vn_00.jpg", "/frames/vn_03.jpg", "/frames/vn_06.jpg"]
    assert bench_frames(im, every=3, n=2) == ["/frames/vn_00.jpg", "/frames/vn_03.jpg"]
    # the real protocol: every 30th, first 200, from a sorted list of 6000
    big = pd.DataFrame({"image_id": [f"vn/{i:05d}" for i in range(6000)][::-1]})
    big["path"] = "/f/" + big.image_id.str.replace("/", "_") + ".jpg"
    big["set_name"] = "T-VN"
    paths = bench_frames(big, 30, 200)
    assert len(paths) == 200 and paths[0] == "/f/vn_00000.jpg" and paths[199] == "/f/vn_05970.jpg"
    # order of the frame table does not change the hash; other frames do
    assert frames_hash(im, 3, 3) == frames_hash(im.sample(frac=1, random_state=0), 3, 3)
    assert frames_hash(im, 3, 3) != frames_hash(im, 3, 2) != frames_hash(im, 2, 3)
    with pytest.raises(ValueError, match="fewer than the 5 frames"):
        bench_frames(im, every=3, n=5)
    # T-VN has 5,950 images: every 30th gives 199 frames (configs/deploy.yaml bench.n)
    tvn = big.iloc[:5950]
    assert len(bench_frames(tvn, 30, 199)) == 199
    with pytest.raises(ValueError, match="gives 199, fewer than the 200 frames"):
        bench_frames(tvn, 30, 200)
    with pytest.raises(ValueError, match=">= 1"):
        bench_frames(im, every=0, n=1)


# --- run_bench with fakes -----------------------------------------------------------------------

def _cfg(every: int = 3, n: int = 3, warmup: int = 2) -> dict:
    return {"students": ["m"], "sizes": [640], "precisions": ["fp16"], "fp32_sizes": [640],
            "tiled": {"full_imgsz": 640, "tile": 640, "overlap": 0.2},
            "bench": {"every": every, "n": n, "warmup": warmup}}


def _models(tmp: Path, fp16_bytes: bytes = b"engine-A") -> dict:
    (tmp / "w.pt").write_bytes(b"pt-weights")
    (tmp / "m@640-fp16.engine").write_bytes(fp16_bytes)
    (tmp / "m@640-fp16.json").write_text(json.dumps({"tensorrt": "10.16.0"}), encoding="utf-8")
    return {
        "m@640-fp16": {"arch": "yolo", "weights": (tmp / "m@640-fp16.engine").as_posix(),
                       "names_map": "cvt", "imgsz": 640},
        "m@640-fp32": {"arch": "yolo", "weights": (tmp / "w.pt").as_posix(),
                       "names_map": "cvt", "imgsz": 640},
        "m@640+tiles640-fp16": {"arch": "tiled", "base": "m@640-fp16",
                                "tile_base": "m@640-fp16", "tile": 640, "overlap": 0.2},
    }


def _device(**kw):
    from cvt.deploy.bench import Device

    d = Device(
        name="FakeGPU", driver="999.1", ultralytics="9.9", torch="2.x+fake", sync=lambda: None,
        begin=lambda: None, used_mb=lambda: 1234.56, release=lambda: None,
    )
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def _install_fake_open(monkeypatch, opened: list[str], tick_ms: float = 5.0):
    from cvt.deploy import bench

    class Det:
        def predict_arrays(self, ims, imgsz=None):
            return []

    def fake_open(model, models_cfg, imgsz, conf, images, boxes):
        opened.append(f"{model}@{imgsz}")
        return Det(), "sha"

    monkeypatch.setattr(bench, "open_detector", fake_open)
    state = {"t": 0.0}

    def clock():
        state["t"] += tick_ms / 1000.0  # t1 - t0 is one tick
        return state["t"]

    return clock


def _decode_fake(paths):
    return [np.zeros((4, 4, 3), np.uint8) for _ in paths]


def test_latency_rows_skip_existing(tmp_path, monkeypatch):
    from cvt.deploy.bench import LATENCY_COLS, run_bench, select_points

    opened: list[str] = []
    clock = _install_fake_open(monkeypatch, opened)
    cfg, models, im = _cfg(), _models(tmp_path), _images()
    out = tmp_path / "reports" / "latency.csv"
    pts = select_points(cfg, None)
    assert len(pts) == 3
    logs: list[str] = []
    rows = run_bench(pts, cfg, models, im, None, out, _device(), log=logs.append, clock=clock,
                     decode=_decode_fake)
    assert [r["model"] for r in rows] == ["m@640-fp16", "m@640-fp32", "m@640+tiles640-fp16"]
    warm = ["m@640-fp16@640", "m@640-fp32@640"]  # pre-warm builds (not measured)
    assert opened == [*warm, "m@640-fp16@640", "m@640-fp32@640", "m@640+tiles640-fp16@640"]
    assert sum("decoded 3 frames once" in ln for ln in logs) == 1
    assert sum("pre-warmed" in ln for ln in logs) == 2
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df.columns) == LATENCY_COLS and len(df) == 3
    r = df.iloc[0]
    assert (r.model, r.imgsz, r.precision, r.tiled) == ("m@640-fp16", "640", "fp16", "False")
    assert (r.n_frames, r.warmup) == ("3", "2")
    # decode is timed once with the bench clock (one 5 ms tick for 3 frames); sizes from the arrays
    assert float(r.decode_ms_per_frame) == pytest.approx(5.0 / 3, abs=1e-3)
    assert r.frame_sizes == "4x4:3" and df.frame_sizes.nunique() == 1
    assert float(r.median_ms) == pytest.approx(5.0) and float(r.fps) == pytest.approx(200.0)
    assert r.gpu_mem_mb == "1234.6" and r.gpu == "FakeGPU" and r.driver == "999.1"
    assert r.tensorrt == "10.16.0" and df.iloc[1].tensorrt == "" and df.iloc[2].tensorrt == "10.16.0"
    assert (r.ultralytics, r.torch) == ("9.9", "2.x+fake")
    assert len(r.frames_hash) == 16 and len(r.weights_sha) == 64
    assert df.iloc[2].tiled == "True" and df.iloc[2].weights_sha != r.weights_sha
    assert not list(out.parent.glob("*.tmp"))

    # same frames + same weights: nothing is opened, timed or decoded again
    opened.clear()
    logs.clear()
    assert run_bench(pts, cfg, models, im, None, out, _device(), log=logs.append, clock=clock,
                     decode=_decode_fake) == []
    assert opened == [] and not any("decoded" in ln for ln in logs)
    assert sum("already benchmarked" in ln for ln in logs) == 3
    assert len(pd.read_csv(out)) == 3

    # a rebuilt engine (other sha) re-benchmarks that point, and its tiled user too
    models = _models(tmp_path, fp16_bytes=b"engine-B")
    run_bench(pts, cfg, models, im, None, out, _device(), clock=clock, decode=_decode_fake)
    assert opened == ["m@640-fp16@640"] * 2 + ["m@640+tiles640-fp16@640"]  # pre-warm + 2
    assert len(pd.read_csv(out)) == 5

    # other frames (different frames_hash) re-benchmark everything
    opened.clear()
    run_bench(pts, _cfg(every=2), models, im, None, out, _device(), clock=clock,
              decode=_decode_fake)
    assert len(opened) == 3 + 2 and len(pd.read_csv(out)) == 8

    # another GPU, or another warm-up, is not "already benchmarked" either
    opened.clear()
    run_bench(pts[:1], cfg, models, im, None, out, _device(name="OtherGPU"), clock=clock,
              decode=_decode_fake)
    assert opened == ["m@640-fp16@640"] * 2  # pre-warm + measured
    run_bench(pts[:1], _cfg(warmup=5), models, im, None, out, _device(), clock=clock,
              decode=_decode_fake)
    assert opened == ["m@640-fp16@640"] * 4 and len(pd.read_csv(out)) == 10
    run_bench(pts[:1], cfg, models, im, None, out, _device(name="OtherGPU"), clock=clock,
              decode=_decode_fake)
    assert len(opened) == 4  # that exact combination is now recorded


def test_run_bench_fails_before_timing_and_without_mem(tmp_path, monkeypatch):
    from cvt.deploy.bench import Point, run_bench

    opened: list[str] = []
    clock = _install_fake_open(monkeypatch, opened)
    cfg, models, im = _cfg(), _models(tmp_path), _images()
    out = tmp_path / "latency.csv"
    (tmp_path / "m@640-fp16.engine").unlink()
    pts = [Point("m", 640, "fp32", False), Point("m", 640, "fp16", False)]
    with pytest.raises(FileNotFoundError, match="m@640-fp16.*cvt deploy export"):
        run_bench(pts, cfg, models, im, None, out, _device(), clock=clock, decode=_decode_fake)
    assert opened == [] and not out.exists()
    with pytest.raises(KeyError, match="cvt deploy register"):
        run_bench([Point("zz", 640, "fp16", False)], cfg, models, im, None, out, _device())
    # no CUDA memory counters: the column is empty, the benchmark still runs
    rows = run_bench(pts[:1], cfg, models, im, None, out,
                     _device(begin=None, used_mb=None, release=None), clock=clock, decode=_decode_fake)
    assert np.isnan(rows[0]["gpu_mem_mb"])
    assert pd.read_csv(out, dtype=str, keep_default_na=False).gpu_mem_mb.tolist() == [""]


def test_memory_is_measured_around_the_build_and_released_between_points(tmp_path, monkeypatch):
    from cvt.deploy import bench

    events: list[str] = []

    class Det:
        def predict_arrays(self, ims, imgsz=None):
            return []

    def fake_open(model, *a):
        events.append("open")
        return Det(), "sha"

    monkeypatch.setattr(bench, "open_detector", fake_open)
    dev = _device(sync=lambda: events.append("sync"), begin=lambda: events.append("begin"),
                  used_mb=lambda: events.append("used") or 7.0,
                  release=lambda: events.append("release"))
    cfg = _cfg(n=1, warmup=0)
    pts = bench.select_points(cfg, "m@640-fp16,m@640-fp32")
    rows = bench.run_bench(pts, cfg, _models(tmp_path), _images(), None, tmp_path / "l.csv", dev,
                           decode=_decode_fake)
    assert [r["gpu_mem_mb"] for r in rows] == [7.0, 7.0]
    warm = ["open", "sync", "release"]  # pre-warm (warmup 0: no predictions)
    one = ["begin", "open", "sync", "sync", "used", "release"]  # begin precedes the build
    assert events == warm * 2 + one * 2


def test_prewarm_builds_one_engine_and_one_fp32_point_before_measuring(tmp_path, monkeypatch):
    from cvt.deploy import bench

    events: list[str] = []

    class Det:
        def __init__(self, model):
            self.model = model

        def predict_arrays(self, ims, imgsz=None):
            events.append(f"predict {self.model}")
            return []

    def fake_open(model, *a):
        events.append(f"open {model}")
        return Det(model), "sha"

    def decode(paths):
        events.append("decode")
        return _decode_fake(paths)

    monkeypatch.setattr(bench, "open_detector", fake_open)
    dev = _device(sync=lambda: events.append("sync"), begin=lambda: events.append("begin"),
                  used_mb=lambda: events.append("used") or 7.0,
                  release=lambda: events.append("release"))
    cfg = _cfg(n=1, warmup=2)
    fp32, fp16, tiled = "m@640-fp32", "m@640-fp16", "m@640+tiles640-fp16"
    pts = bench.select_points(cfg, f"{fp32},{tiled},{fp16}")
    rows = bench.run_bench(pts, cfg, _models(tmp_path), _images(), None, tmp_path / "l.csv", dev,
                           decode=decode)
    assert [r["model"] for r in rows] == [fp32, tiled, fp16]  # only measured points get rows

    def warm(m):  # build, `warmup` predictions, drain, delete, release; nothing measured
        return [f"open {m}", f"predict {m}", f"predict {m}", "sync", "release"]

    def timed(m):
        return ["begin", f"open {m}", f"predict {m}", f"predict {m}", "sync", f"predict {m}",
                "sync", "used", "release"]

    assert events == ["decode", *warm(fp16), *warm(fp32), *timed(fp32), *timed(tiled),
                      *timed(fp16)]
    # only a tiled engine point pending: it is the engine pre-warm; no FP32 pre-warm
    events.clear()
    bench.run_bench(bench.select_points(cfg, tiled), cfg, _models(tmp_path), _images(), None,
                    tmp_path / "l2.csv", dev, decode=decode)
    assert events == ["decode", *warm(tiled), *timed(tiled)]
    # nothing pending: nothing is built
    events.clear()
    bench.run_bench(bench.select_points(cfg, tiled), cfg, _models(tmp_path), _images(), None,
                    tmp_path / "l2.csv", dev, decode=decode)
    assert events == []


def test_cuda_device_memory_is_device_wide_free_memory_delta(monkeypatch):
    import torch

    from cvt.deploy import bench

    calls: list[str] = []
    free = iter([10 * 2**30, 10 * 2**30 - 300 * 2**20])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake RTX")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (next(free), 12 * 2**30))
    monkeypatch.setattr(bench, "driver_version", lambda: "1")
    dev = bench.cuda_device()
    with pytest.raises(RuntimeError, match="before begin"):
        dev.used_mb()
    dev.begin()
    assert calls == ["empty_cache"]
    assert dev.used_mb() == pytest.approx(300.0)
    assert calls == ["empty_cache", "synchronize"]
    dev.release()
    assert calls[-1] == "empty_cache" and dev.name == "Fake RTX"


def test_append_latency_atomic_and_refuses_header_mismatch(tmp_path):
    from cvt.deploy.bench import LATENCY_COLS, append_latency, run_bench

    out = tmp_path / "r" / "latency.csv"
    row = dict.fromkeys(LATENCY_COLS, "x") | {"tiled": False, "median_ms": 1.5}
    append_latency(out, [row])
    append_latency(out, [row | {"model": "y"}])
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert df.model.tolist() == ["x", "y"] and df.median_ms.tolist() == ["1.5", "1.5"]
    assert df.tiled.tolist() == ["False", "False"]
    assert b"\r" not in out.read_bytes() and not list(out.parent.glob("*.tmp"))
    old = out.read_bytes()
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,model\n1,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from LATENCY_COLS"):
        append_latency(bad, [row])
    assert bad.read_text(encoding="utf-8") == "timestamp,model\n1,a\n"  # untouched
    with pytest.raises(ValueError, match="differs from LATENCY_COLS"):  # also before any work
        run_bench([], _cfg(), {}, _images(), None, bad, _device())
    assert out.read_bytes() == old
    assert LATENCY_COLS.index("weights_sha") == LATENCY_COLS.index("frames_hash") + 1
    w = LATENCY_COLS.index("warmup")
    assert LATENCY_COLS[w + 1:w + 3] == ["decode_ms_per_frame", "frame_sizes"]


def test_frame_sizes_counts_most_common_first():
    from cvt.deploy.bench import frame_sizes

    ims = [np.zeros((2160, 3840, 3), np.uint8)] * 3 + [np.zeros((1080, 1920, 3), np.uint8)]
    assert frame_sizes(ims) == "3840x2160:3;1920x1080:1"
    assert frame_sizes([np.zeros((4, 6, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)]) == (
        "4x4:1;6x4:1")  # ties: by size string


def test_driver_version(monkeypatch):
    from cvt.deploy import bench

    seen = {}

    def ok(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="581.42\n581.42\n", stderr="")

    monkeypatch.setattr(subprocess, "run", ok)
    assert bench.driver_version() == "581.42"
    assert seen["cmd"] == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    def failing(cmd, **kw):
        raise subprocess.CalledProcessError(9, cmd)

    for fn in (missing, failing):
        monkeypatch.setattr(subprocess, "run", fn)
        assert bench.driver_version() == ""
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    assert bench.driver_version() == ""


def test_select_points():
    from cvt.deploy.bench import select_points

    cfg = _cfg()
    assert len(select_points(cfg, None)) == 3
    assert [p.precision for p in select_points(cfg, "m@640-fp32, m@640-fp16")] == ["fp32", "fp16"]
    assert len(select_points(cfg, "m@640-fp16,m@640-fp16")) == 1
    with pytest.raises(ValueError, match="nope"):
        select_points(cfg, "m@640-fp16,nope")
    with pytest.raises(ValueError, match="--only"):
        select_points(cfg, " , ")


def test_weights_sha_tiled_combines_base_and_tile_base(tmp_path):
    from cvt.deploy.bench import tensorrt_of, weights_sha

    models = _models(tmp_path)
    a = weights_sha("m@640+tiles640-fp16", models)
    models["m@640+tiles640-fp16"]["tile_base"] = "m@640-fp32"
    assert weights_sha("m@640+tiles640-fp16", models) != a
    assert tensorrt_of("m@640-fp32", models) == ""
    (tmp_path / "m@640-fp16.json").unlink()
    with pytest.raises(FileNotFoundError, match="no sidecar"):
        tensorrt_of("m@640-fp16", models)


# --- tiled detectors accept frames --------------------------------------------------------------

def test_tiled_predict_arrays_matches_predict_paths():
    from cvt.detect.base import Detections
    from cvt.detect.composite import TiledDetector

    class Base:
        cache_tag = "b"

        def predict_arrays(self, ims, imgsz=None):
            return [Detections(np.array([[1, 1, 5, 5]], np.float32),
                               np.array([0.9], np.float32), np.array([0])) for _ in ims]

    frame = np.zeros((20, 30, 3), np.uint8)
    det = TiledDetector("t", Base(), tile=16, overlap=0.0, reader=lambda p: frame)
    via_paths = list(det.predict_paths(["x", "y"]))
    via_arrays = det.predict_arrays([frame, frame])
    assert len(via_arrays) == 2
    for a, b in zip(via_paths, via_arrays, strict=True):
        assert np.array_equal(a.boxes, b.boxes) and np.array_equal(a.scores, b.scores)
        assert np.array_equal(a.labels, b.labels)
    assert det.predict_arrays([]) == []


# --- CLI ----------------------------------------------------------------------------------------

def _cli_setup(tmp_path: Path, monkeypatch):
    """Config, registry, manifest and 12 real tiny T-VN images; fake detector and device."""
    import cv2

    from cvt.deploy import bench
    from cvt.deploy.engines import write_registry

    cfg = _cfg()
    cfg_path = tmp_path / "deploy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    models = _models(tmp_path)
    src = {"m": {"arch": "yolo", "weights": (tmp_path / "w.pt").as_posix(), "names_map": "cvt"}}
    reg = write_registry(cfg, src, tmp_path / "models_deploy.yaml", tmp_path)
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(src), encoding="utf-8")
    assert set(yaml.safe_load(reg.read_text(encoding="utf-8"))) == set(models)
    mdir = tmp_path / "manifest"
    mdir.mkdir()
    im = _images(extra_sets=False)
    paths = []
    for i, image_id in enumerate(im.image_id):
        p = tmp_path / f"{image_id.replace('/', '_')}.png"
        cv2.imwrite(str(p), np.full((8, 8, 3), i, np.uint8))
        paths.append(p.as_posix())
    im["path"] = paths
    im.to_parquet(mdir / "images.parquet")
    pd.DataFrame({"image_id": [], "x1": [], "y1": [], "x2": [], "y2": [], "label": [],
                  "is_ignore": []}).to_parquet(mdir / "boxes.parquet")
    opened: list[str] = []
    _install_fake_open(monkeypatch, opened)
    monkeypatch.setattr(bench, "cuda_device", lambda: _device())
    args = ["--config", str(cfg_path), "--manifest", str(mdir), "--out",
            str(tmp_path / "latency.csv"), "--models",
            f"{tmp_path / 'models.yaml'},{tmp_path / 'models_deploy.yaml'}"]
    return args, opened


def test_deploy_cli_bench(tmp_path, monkeypatch):
    from cvt.cli import app

    args, opened = _cli_setup(tmp_path, monkeypatch)
    runner = CliRunner()
    r = runner.invoke(app, ["deploy", "bench", "--only", "m@640-fp16,m@640+tiles640-fp16", *args])
    assert r.exit_code == 0, r.output
    assert opened == ["m@640-fp16@640", "m@640-fp16@640", "m@640+tiles640-fp16@640"]
    assert r.output.count("decoded 3 frames once") == 1
    df = pd.read_csv(tmp_path / "latency.csv", dtype=str, keep_default_na=False)
    assert df.model.tolist() == ["m@640-fp16", "m@640+tiles640-fp16"]
    assert df.gpu.tolist() == ["FakeGPU", "FakeGPU"]
    r = runner.invoke(app, ["deploy", "bench", "--only", "m@640-fp16", *args])
    assert r.exit_code == 0 and "already benchmarked" in r.output and len(opened) == 3
    r = runner.invoke(app, ["deploy", "bench", "--only", "m@999-fp16", *args])
    assert r.exit_code == 1 and "not a point of the grid" in r.output
    r = runner.invoke(app, ["deploy", "bench", *args[:-1], str(tmp_path / "models.yaml")])
    assert r.exit_code == 1 and "m@640-fp16" in r.output and "cvt deploy register" in r.output


def test_deploy_cli_bench_refuses_without_cuda(tmp_path, monkeypatch):
    import torch

    from cvt.cli import app

    args, opened = _cli_setup(tmp_path, monkeypatch)
    monkeypatch.undo()  # drop the fake device: use the real guard with CUDA reported absent
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = CliRunner().invoke(app, ["deploy", "bench", *args])
    assert r.exit_code == 1 and "needs a CUDA GPU" in r.output
    assert not (tmp_path / "latency.csv").exists()
