"""Ground-plane tracks (metres) and smoothed speed / heading / acceleration per track."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from cvt.analytics.events import VEHICLE_IDS
from cvt.track.stabilize import stabilize_tracks

KINEMATIC_COLS = ["xs_m", "ys_m", "speed_kmh", "heading_rad", "accel_ms2"]


def ground_tracks(
    tracks: pd.DataFrame, H: np.ndarray, valid: np.ndarray, frames: Sequence[int], mpp: float
) -> pd.DataFrame:
    """Track rows plus `x_m, y_m` (stabilized reference pixels x mpp) and `valid`."""
    g = stabilize_tracks(tracks, H, valid, frames).reset_index(drop=True)
    g["x_m"] = g["cx_ref"] * mpp
    g["y_m"] = g["cy_ref"] * mpp
    return g.drop(columns=["cx_ref", "cy_ref"])


def _runs(frame: np.ndarray, ok: np.ndarray) -> list[np.ndarray]:
    """Index arrays of maximal runs of consecutive frames where `ok` (frame sorted ascending)."""
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frame[idx]) != 1) + 1
    return np.split(idx, breaks)


def kinematics(g: pd.DataFrame, fps: float, smooth_s: float) -> pd.DataFrame:
    """Add `t_s, xs_m, ys_m, speed_kmh, heading_rad, accel_ms2` (NaN where undefined).

    `g` is one clip (rows are grouped by track_id only). Per run of consecutive valid frames a
    quadratic Savitzky-Golay filter (window = smallest odd >= max(5, round(smooth_s * fps)),
    mode="interp") gives the smoothed positions and, analytically, the velocity; it is unbiased
    at the run ends. Runs shorter than the window get no kinematics. speed = hypot(vx, vy),
    heading_rad = atan2(vy, vx) in image axes (y down), accel_ms2 = gradient of speed.
    """
    out = g.copy()
    out["t_s"] = out["frame"].to_numpy() / fps
    cols = {c: np.full(len(out), np.nan) for c in KINEMATIC_COLS}
    window = max(5, round(smooth_s * fps))
    window += 1 - window % 2
    frame_all, valid_all = out["frame"].to_numpy(), out["valid"].to_numpy(bool)
    t_all, x_all, y_all = out["t_s"].to_numpy(), out["x_m"].to_numpy(), out["y_m"].to_numpy()
    for rows in out.groupby("track_id", sort=False).indices.values():
        rows = rows[np.argsort(frame_all[rows], kind="stable")]
        for run in _runs(frame_all[rows], valid_all[rows]):
            if len(run) < window:
                continue
            r = rows[run]
            kw = {"polyorder": 2, "mode": "interp"}
            xs = savgol_filter(x_all[r], window, **kw)
            ys = savgol_filter(y_all[r], window, **kw)
            vx = savgol_filter(x_all[r], window, deriv=1, delta=1 / fps, **kw)
            vy = savgol_filter(y_all[r], window, deriv=1, delta=1 / fps, **kw)
            speed = np.hypot(vx, vy)
            cols["xs_m"][r], cols["ys_m"][r] = xs, ys
            cols["speed_kmh"][r] = speed * 3.6
            cols["heading_rad"][r] = np.arctan2(vy, vx)
            cols["accel_ms2"][r] = np.gradient(speed, t_all[r])
    for c, v in cols.items():
        out[c] = v
    return out


def noise_floor(kin: pd.DataFrame, n_frames: int, min_span_frac: float, max_disp_m: float) -> dict:
    """Speed statistics of parked vehicles (vehicle-class tracks that are long-lived and whose
    smoothed position stays near its median)."""
    speeds = []
    n_parked = 0
    for _, t in kin[kin["label"].isin(VEHICLE_IDS)].groupby("track_id"):
        if t["frame"].max() - t["frame"].min() + 1 < min_span_frac * n_frames:
            continue
        p = t.dropna(subset=["xs_m", "ys_m"])
        if p.empty:
            continue
        d = np.hypot(p["xs_m"] - p["xs_m"].median(), p["ys_m"] - p["ys_m"].median())
        if d.max() > max_disp_m:
            continue
        n_parked += 1
        speeds.append(p["speed_kmh"].to_numpy())
    s = np.concatenate(speeds) if speeds else np.array([])
    return {
        "n_parked": n_parked,
        "median_kmh": float(np.median(s)) if len(s) else float("nan"),
        "p95_kmh": float(np.percentile(s, 95)) if len(s) else float("nan"),
    }
