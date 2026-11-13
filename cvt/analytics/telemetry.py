"""DJI flight telemetry from the mp4 subtitle track, and the metres-per-pixel scale."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.schema import CLASS_ID

CHUNK = 64_000_000
TAIL = 300
CAMERA_SCAN = 4_000_000
COLS = ["t_s", "lon", "lat", "alt_m", "dist_m", "height_m", "hs_ms", "vs_ms", "zoom"]

REC = re.compile(rb"DZOOM ([0-9.]+), GPS \(([0-9.\-]+), ([0-9.\-]+), ([0-9.\-]+)\), "
                 rb"D ([0-9.\-]+)m, H ([0-9.\-]+)m, H\.S ([0-9.\-]+)m/s, V\.S ([0-9.\-]+)m/s")
CAMERA = re.compile(rb"FC[0-9]{4}[A-Z]?")


def read_telemetry(video: Path) -> pd.DataFrame:
    """One row per telemetry record; record k has t_s = k. Empty frame if none."""
    rows: list[tuple[float, ...]] = []
    seen: set[int] = set()
    tail = b""
    pos = 0
    with open(video, "rb") as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            base = pos - len(tail)
            buf = tail + data
            for m in REC.finditer(buf):
                start = base + m.start()
                if start in seen:
                    continue
                seen.add(start)
                zoom, lon, lat, alt, dist, h, hs, vs = (float(g) for g in m.groups())
                rows.append((len(rows), lon, lat, alt, dist, h, hs, vs, zoom))
            pos += len(data)
            tail = buf[-TAIL:]
    return pd.DataFrame(rows, columns=COLS)


def camera_model(video: Path) -> str | None:
    """First camera tag (e.g. FC7303) in the first or last 4 MB of the file."""
    size = Path(video).stat().st_size
    with open(video, "rb") as f:
        head = f.read(CAMERA_SCAN)
        f.seek(max(size - CAMERA_SCAN, 0))
        end = f.read(CAMERA_SCAN)
    for part in (head, end):
        m = CAMERA.search(part)
        if m:
            return m.group().decode()
    return None


def hfov_deg(dfov_deg: float, aspect: tuple[int, int]) -> float:
    a, b = aspect
    half = math.atan(math.tan(math.radians(dfov_deg) / 2) * a / math.hypot(a, b))
    return math.degrees(2 * half)


def metres_per_pixel(height_m: float, hfov: float, width_px: int) -> float:
    return 2 * height_m * math.tan(math.radians(hfov) / 2) / width_px


def car_scale(dets: pd.DataFrame, car_length_m: float) -> tuple[float, int]:
    """Metres per pixel from confident, axis-aligned car boxes (long side = car length)."""
    cars = dets[(dets["label"] == CLASS_ID["car"]) & (dets["score"] >= 0.5)]
    w = (cars["x2"] - cars["x1"]).to_numpy(float)
    h = (cars["y2"] - cars["y1"]).to_numpy(float)
    long_, short = np.maximum(w, h), np.minimum(w, h)
    keep = (short > 0) & (long_ / np.where(short > 0, short, 1) > 1.6)
    if not keep.any():
        return float("nan"), 0
    return car_length_m / float(np.median(long_[keep])), int(keep.sum())


def ref_second(tel: pd.DataFrame, tol_m: float) -> int:
    """First second whose height is within tol_m of the median height."""
    dev = (tel["height_m"] - tel["height_m"].median()).abs()
    ok = tel[dev <= tol_m]
    if len(ok):
        return int(ok["t_s"].iloc[0])
    return int(tel["t_s"].iloc[int(dev.to_numpy().argmin())])


def clip_scale(video: Path, dets: pd.DataFrame, width_px: int, cfg: dict) -> dict:
    nan = float("nan")
    tel = read_telemetry(video)
    cam = camera_model(video)
    mpp_car, n_cars = car_scale(dets, cfg["car_length_m"])
    out = {"camera": cam, "height_med": nan, "height_min": nan, "height_max": nan,
           "zoom": nan, "ref_s": 0, "mpp_telemetry": nan, "mpp_car": mpp_car,
           "n_cars": n_cars}
    if len(tel):
        out["height_med"] = float(tel["height_m"].median())
        out["height_min"] = float(tel["height_m"].min())
        out["height_max"] = float(tel["height_m"].max())
        out["zoom"] = float(tel["zoom"].median())
        out["ref_s"] = ref_second(tel, cfg["ref_height_tol_m"])
        cam_cfg = cfg["cameras"].get(cam) if cam else None
        if cam_cfg:
            h_ref = float(tel.loc[tel["t_s"] == out["ref_s"], "height_m"].iloc[0])
            hfov = hfov_deg(cam_cfg["dfov_deg"], tuple(cam_cfg["aspect"]))
            out["mpp_telemetry"] = metres_per_pixel(h_ref, hfov, width_px)
    if not math.isnan(out["mpp_telemetry"]):
        out["mpp"], out["source"] = out["mpp_telemetry"], "telemetry"
    else:
        out["mpp"], out["source"] = mpp_car, "car-length (uncalibrated)"
    return out
