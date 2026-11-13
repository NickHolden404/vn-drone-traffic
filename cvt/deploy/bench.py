"""End-to-end latency benchmark of the deployment grid (GPU, user-run via `cvt deploy bench`).

Protocol: 199 fixed T-VN frames (every 30th of the image_id-sorted
list), decoded once before any timing, 20 warm-up calls, batch 1, `predict_arrays` end to end
(letterbox, inference, NMS, class mapping; tiles for tiled points), CUDA synchronised around every
timed call, median / p90, fps = 1000 / median. Rows go to reports/latency.csv.

Before the first measured point, one pending TensorRT point and one pending PyTorch FP32 point are
built and run for the warm-up calls, then deleted (pre-warm, not recorded): the one-off CUDA
context, TensorRT runtime and cuDNN initialisation then never lands in a measured point's latency
or memory figure.

Each row also records the untimed decode cost (`decode_ms_per_frame`, cv2.imread of the frames
once, by the bench clock) and the frame-size mix (`frame_sizes`, e.g. `3840x2160:187;1920x1080:12`)
for the README caption.

`gpu_mem_mb` is the steady-state device footprint of the point, measured the same way for PyTorch
and TensorRT: free device memory (`torch.cuda.mem_get_info`) before the detector is built minus
free memory after the timed loop, so it includes engine weights, TensorRT context / scratch memory
and the torch caching allocator. It is device-wide: close other GPU applications while benchmarking.
(`torch.cuda.max_memory_allocated` would miss TensorRT's own allocations.)"""

from __future__ import annotations

import csv
import gc
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.deploy.engines import Point, grid_points, grid_tile, registry_name
from cvt.eval.run import open_detector
from cvt.utils import file_sha256, git_commit, stable_hash

LATENCY = Path("reports/latency.csv")
BENCH_SET = "T-VN"
LATENCY_COLS = [
    "timestamp", "git_commit", "model", "imgsz", "precision", "tiled", "n_frames", "warmup",
    "decode_ms_per_frame", "frame_sizes", "median_ms", "p90_ms", "fps", "gpu_mem_mb", "gpu", "driver", "tensorrt", "ultralytics",
    "torch", "frames_hash", "weights_sha",
]
Log = Callable[[str], None]


def _quiet(msg: str) -> None:
    pass


# --- frames and timing --------------------------------------------------------------------------

def _bench_images(images: pd.DataFrame, every: int, n: int) -> pd.DataFrame:
    if every < 1 or n < 1:
        raise ValueError(f"bench every and n must be >= 1, got every={every}, n={n}")
    tvn = images[images.set_name == BENCH_SET].sort_values("image_id")
    sel = tvn.iloc[::every].iloc[:n]
    if len(sel) < n:
        raise ValueError(f"{BENCH_SET} has {len(tvn)} images: every {every}th gives {len(sel)}, "
                         f"fewer than the {n} frames to benchmark")
    return sel


def bench_frames(images: pd.DataFrame, every: int, n: int) -> list[str]:
    """Paths of every `every`-th T-VN image of the list sorted by image_id, the first `n`."""
    return _bench_images(images, every, n).path.tolist()


def frames_hash(images: pd.DataFrame, every: int, n: int) -> str:
    """Stable hash of the benchmark frames' image_ids (in order)."""
    return stable_hash(_bench_images(images, every, n).image_id.tolist())


def decode_frames(paths: Sequence[str]) -> list[np.ndarray]:
    """Decode to BGR arrays (cv2.imread), failing loudly on an unreadable image."""
    import cv2

    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            raise OSError(f"unreadable image: {p}")
        out.append(im)
    return out


def frame_sizes(frames: Sequence[np.ndarray]) -> str:
    """`WxH:count` per frame size, most common first (ties by size), `;`-separated."""
    counts: dict[str, int] = {}
    for f in frames:
        key = f"{f.shape[1]}x{f.shape[0]}"
        counts[key] = counts.get(key, 0) + 1
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def time_detector(
    detector, frames: list[np.ndarray], warmup: int, sync: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Per-frame end-to-end latency (batch 1). Warm-up calls (cycling over the frames) are not
    timed; each timed call is `sync(); t0; predict_arrays([frame]); sync(); t1`, so the first
    timed call's leading sync also drains any queued warm-up work."""
    if not frames:
        raise ValueError("no frames to time")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    for i in range(warmup):
        detector.predict_arrays([frames[i % len(frames)]])
    ms = []
    for frame in frames:
        sync()
        t0 = clock()
        detector.predict_arrays([frame])
        sync()
        ms.append((clock() - t0) * 1000.0)
    median = float(np.median(ms))
    if median <= 0:
        raise RuntimeError(f"non-positive median latency {median} ms: broken clock")
    return {"median_ms": median, "p90_ms": float(np.percentile(ms, 90)), "fps": 1000.0 / median,
            "n_frames": len(ms), "warmup": warmup}


# --- environment --------------------------------------------------------------------------------

def driver_version() -> str:
    """NVIDIA driver version from nvidia-smi; "" when it cannot be queried."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = r.stdout.strip().splitlines()
    return lines[0].strip() if lines else ""


@dataclass
class Device:
    """What the benchmark needs from the GPU; tests pass fakes. `begin` (before the detector is
    built) and `used_mb` (after timing) bracket the device memory measurement; both None when it
    cannot be measured (the column is then empty). `release` frees cached memory between points."""
    name: str
    driver: str
    ultralytics: str
    torch: str
    sync: Callable[[], None]
    begin: Callable[[], None] | None = None
    used_mb: Callable[[], float] | None = None
    release: Callable[[], None] | None = None


def cuda_device() -> Device:
    """The CUDA device 0; refuses to run without CUDA (this is a GPU benchmark)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("deploy bench needs a CUDA GPU: torch.cuda.is_available() is False "
                           "(run it where the engines were exported)")
    free0: list[int] = []

    def release() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def begin() -> None:
        release()
        free0[:] = [torch.cuda.mem_get_info()[0]]

    def used_mb() -> float:
        if not free0:
            raise RuntimeError("used_mb() called before begin()")
        torch.cuda.synchronize()
        return (free0[0] - torch.cuda.mem_get_info()[0]) / 2**20

    return Device(
        name=torch.cuda.get_device_name(0), driver=driver_version(),
        ultralytics=metadata.version("ultralytics"), torch=torch.__version__,
        sync=torch.cuda.synchronize, begin=begin, used_mb=used_mb, release=release,
    )


# --- grid points, weights, sidecars -------------------------------------------------------------

def select_points(cfg: dict, only: str | None) -> list[Point]:
    """Every grid point, or the comma-separated registry names of `only`."""
    tile = grid_tile(cfg)
    pts = grid_points(cfg)
    if only is None:
        return pts
    names = {registry_name(p, tile): p for p in pts}
    want = [s.strip() for s in only.split(",") if s.strip()]
    bad = [s for s in want if s not in names]
    if not want or bad:
        raise ValueError(f"--only {bad or only!r} is not a point of the grid ({sorted(names)})")
    return [names[s] for s in dict.fromkeys(want)]


def weights_sha(name: str, models_cfg: dict) -> str:
    """sha256 of the engine / checkpoint behind a registry entry (a tiled entry: hash of the
    shas of its base and tile_base)."""
    entry = models_cfg[name]
    if entry["arch"] == "tiled":
        refs = [entry["base"], entry.get("tile_base", entry["base"])]
        return stable_hash([weights_sha(r, models_cfg) for r in refs])
    w = Path(entry["weights"])
    if not w.exists():
        hint = " (run `cvt deploy export`)" if w.suffix == ".engine" else ""
        raise FileNotFoundError(f"{name}: weights not found: {w}{hint}")
    return file_sha256(w)


def tensorrt_of(name: str, models_cfg: dict) -> str:
    """TensorRT version recorded in the engine sidecar; "" for a PyTorch FP32 point."""
    entry = models_cfg[name]
    if entry["arch"] == "tiled":
        return tensorrt_of(entry["base"], models_cfg)
    w = Path(entry["weights"])
    if w.suffix != ".engine":
        return ""
    sidecar = w.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"{name}: engine {w} has no sidecar {sidecar}")
    return str(json.loads(sidecar.read_text(encoding="utf-8")).get("tensorrt", ""))


# --- latency.csv --------------------------------------------------------------------------------

def _read_latency(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LATENCY_COLS, dtype=str)
    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])
    if header != LATENCY_COLS:
        raise ValueError(f"{path}: header {header} differs from LATENCY_COLS {LATENCY_COLS}; "
                         "migrate the file before appending")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def append_latency(path: Path, rows: list[dict]) -> None:
    """Append `rows`: read the existing file, concat, write a temp file, os.replace. Refuses a
    file with a different header."""
    path = Path(path)
    old = _read_latency(path)
    new = pd.DataFrame(rows, columns=LATENCY_COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.concat([old, new], ignore_index=True).to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def already_benchmarked(
    existing: pd.DataFrame, model: str, fhash: str, wsha: str, gpu: str, warmup: int,
) -> bool:
    """A row with this registry name, frames hash, weights sha, GPU and warm-up is recorded."""
    return bool(((existing.model == model) & (existing.frames_hash == fhash)
                 & (existing.weights_sha == wsha) & (existing.gpu == gpu)
                 & (existing.warmup == str(warmup))).any())


# --- the benchmark ------------------------------------------------------------------------------

def _release(device: Device) -> None:
    gc.collect()
    if device.release is not None:
        device.release()


def prewarm_points(pending: list[tuple]) -> list[tuple]:
    """The first pending TensorRT point (a full-frame engine, else a tiled one) and the first
    pending FP32 point, in that order (absent kinds are skipped)."""
    engine = (next((x for x in pending if x[0].is_engine), None)
              or next((x for x in pending if x[0].tiled), None))
    fp32 = next((x for x in pending if x[0].precision == "fp32" and not x[0].tiled), None)
    return [x for x in (engine, fp32) if x is not None]


def run_bench(
    points: list[Point], cfg: dict, models_cfg: dict, images: pd.DataFrame,
    boxes: pd.DataFrame, out: Path, device: Device, *, log: Log = _quiet,
    clock: Callable[[], float] = time.perf_counter,
    decode: Callable[[Sequence[str]], list[np.ndarray]] = decode_frames,
) -> list[dict]:
    """Time every point not yet recorded in `out` (same registry name, frames hash, weights sha,
    GPU and warm-up), appending one row per point as it finishes. Returns the new rows."""
    b, tile = cfg["bench"], grid_tile(cfg)
    every, n, warmup = int(b["every"]), int(b["n"]), int(b["warmup"])
    fhash = frames_hash(images, every, n)
    existing = _read_latency(out)  # refuses a header mismatch before any work
    pending = []
    for p in points:
        name = registry_name(p, tile)
        if name not in models_cfg:
            raise KeyError(f"{name} not in the models config: run `cvt deploy register`")
        wsha = weights_sha(name, models_cfg)  # all weights must exist before anything is timed
        if already_benchmarked(existing, name, fhash, wsha, device.name, warmup):
            log(f"{name}: already benchmarked, skipped")
        else:
            pending.append((p, name, wsha))
    if not pending:
        return []
    t0 = clock()
    frames = decode(bench_frames(images, every, n))
    decode_s = clock() - t0
    decode_ms, sizes = round(1000.0 * decode_s / len(frames), 3), frame_sizes(frames)
    log(f"decoded {len(frames)} frames once in {decode_s:.1f} s ({decode_ms} ms per frame, not "
        f"timed): {sizes}")
    for p, name, _ in prewarm_points(pending):  # one-off GPU initialisation, nothing measured
        detector, _ = open_detector(name, models_cfg, p.imgsz, 0.01, images, boxes)
        for i in range(warmup):
            detector.predict_arrays([frames[i % len(frames)]])
        device.sync()
        del detector
        _release(device)
        log(f"{name}: pre-warmed ({warmup} calls, not recorded)")
    commit, rows = git_commit(), []
    for p, name, wsha in pending:
        if device.begin is not None:  # before the build: the engine loads inside open_detector
            device.begin()
        detector, _ = open_detector(name, models_cfg, p.imgsz, 0.01, images, boxes)
        res = time_detector(detector, frames, warmup, device.sync, clock)
        used = device.used_mb() if device.used_mb is not None else float("nan")
        row = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": commit,
            "model": name, "imgsz": p.imgsz, "precision": p.precision, "tiled": p.tiled,
            "n_frames": res["n_frames"], "warmup": warmup, "decode_ms_per_frame": decode_ms,
            "frame_sizes": sizes, "median_ms": round(res["median_ms"], 3),
            "p90_ms": round(res["p90_ms"], 3), "fps": round(res["fps"], 2),
            "gpu_mem_mb": round(used, 1), "gpu": device.name, "driver": device.driver,
            "tensorrt": tensorrt_of(name, models_cfg), "ultralytics": device.ultralytics,
            "torch": device.torch, "frames_hash": fhash, "weights_sha": wsha,
        }
        append_latency(out, [row])
        rows.append(row)
        log(f"{name}: median {row['median_ms']} ms, p90 {row['p90_ms']} ms, "
            f"{row['fps']} fps, {row['gpu_mem_mb']} MB")
        del detector
        _release(device)
    return rows
