"""Virtual gimbal: a smooth crop window that keeps a target centred.

Everything is in normalized frame units. The crop keeps the frame's aspect ratio, so its width and
height are both `1 / zoom` of the frame. Commands are pan/tilt rate (fraction of frame width/height
per second) and zoom rate (relative rate, d ln(zoom)/dt, i.e. ~factor per second); degrees only
appear in `GimbalCommand.to_mavlink`. Metrics are computed in each sequence's native pixels.

Control: a PD loop on the target-centre offset from the crop centre (hard deadband), then rate and
acceleration limits. The zoom loop drives the target box height to `target_frac` of the crop height.
When the crop would leave the frame it is clamped, that axis's velocity is zeroed (anti-windup: the
integrator stops) and the frame is reported as clamped. No target: zero command, state unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import pandas as pd

BOX = ["x1", "y1", "x2", "y2"]
_EPS = 1e-9


@dataclass(frozen=True)
class GimbalCommand:
    pan_rate: float
    tilt_rate: float
    zoom_rate: float

    def to_mavlink(self, hfov_deg: float, aspect: float) -> dict:
        """Convert to deg/s (`GIMBAL_MANAGER_SET_PITCHYAW`) and a zoom rate (`SET_CAMERA_ZOOM`).

        `aspect` is frame width / height. yaw_rate = pan_rate * hfov; the vertical FOV follows from
        the rectilinear relation vfov = 2 atan(tan(hfov / 2) / aspect), and pitch (positive up) is
        minus the tilt (image y grows downward): pitch_rate = -tilt_rate * vfov.
        """
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect))
        return {
            "yaw_rate": self.pan_rate * hfov_deg,
            "pitch_rate": -self.tilt_rate * vfov,
            "zoom_rate": self.zoom_rate,
        }


@dataclass(frozen=True)
class GimbalParams:
    kp: float
    kd: float
    target_frac: float
    deadband: float = 0.02
    max_rate: float = 0.5
    max_accel: float = 1.0
    max_zoom: float = 8.0
    hfov_deg: float = 84.0


@dataclass(frozen=True)
class GimbalState:
    """Crop centre and zoom; the remaining fields are controller memory and do not affect equality."""

    cx: float
    cy: float
    zoom: float
    vx: float = field(default=0.0, compare=False)
    vy: float = field(default=0.0, compare=False)
    vz: float = field(default=0.0, compare=False)
    ex: float = field(default=math.nan, compare=False)
    ey: float = field(default=math.nan, compare=False)


def _limit(v: float, prev: float, max_rate: float, max_accel: float, dt: float) -> float:
    v = min(max(v, -max_rate), max_rate)
    da = max_accel * dt
    return min(max(v, prev - da), prev + da)


def _pd(e: float, prev_e: float, params: GimbalParams, dt: float) -> float:
    if abs(e) < params.deadband:
        return 0.0
    d = 0.0 if math.isnan(prev_e) else (e - prev_e) / dt
    return params.kp * e + params.kd * d


def step(
    state: GimbalState, target_box_norm, params: GimbalParams, dt: float
) -> tuple[GimbalState, GimbalCommand, bool]:
    """One control step. `target_box_norm` is (x1, y1, x2, y2) in frame fractions, or None."""
    if target_box_norm is None or not np.all(np.isfinite(target_box_norm)):
        held = replace(state, vx=0.0, vy=0.0, vz=0.0, ex=math.nan, ey=math.nan)
        return held, GimbalCommand(0.0, 0.0, 0.0), False
    x1, y1, x2, y2 = (float(v) for v in target_box_norm)
    ex = (x1 + x2) / 2 - state.cx
    ey = (y1 + y2) / 2 - state.cy
    vx = _limit(_pd(ex, state.ex, params, dt), state.vx, params.max_rate, params.max_accel, dt)
    vy = _limit(_pd(ey, state.ey, params, dt), state.vy, params.max_rate, params.max_accel, dt)

    zoom_goal = min(max(params.target_frac / max(y2 - y1, _EPS), 1.0), params.max_zoom)
    ez = math.log(zoom_goal / state.zoom)
    vz_des = params.kp * ez if abs(ez) >= params.deadband else 0.0
    vz = _limit(vz_des, state.vz, params.max_rate, params.max_accel, dt)
    zoom = min(max(state.zoom * math.exp(vz * dt), 1.0), params.max_zoom)
    if zoom in (1.0, params.max_zoom):
        vz = math.log(zoom / state.zoom) / dt

    half = 0.5 / zoom
    cx = state.cx + vx * dt
    cy = state.cy + vy * dt
    ccx = min(max(cx, half), 1 - half)
    ccy = min(max(cy, half), 1 - half)
    clamped = abs(ccx - cx) > 1e-12 or abs(ccy - cy) > 1e-12
    if abs(ccx - cx) > 1e-12:
        vx = 0.0
    if abs(ccy - cy) > 1e-12:
        vy = 0.0
    new = GimbalState(ccx, ccy, zoom, vx, vy, vz, ex, ey)
    return new, GimbalCommand(vx, vy, vz), clamped


def follow(boxes: pd.DataFrame, w: int, h: int, fps: float, params: GimbalParams) -> pd.DataFrame:
    """Run the gimbal over per-frame pixel boxes (`frame`, x1..y2; NaN = no target).

    One row per frame, `dt = 1 / fps`. Returns frame, t, cx, cy, zoom (state after the step),
    pan_rate, tilt_rate, zoom_rate (command) and clamped.
    """
    boxes = boxes.sort_values("frame").reset_index(drop=True)
    scale = np.array([w, h, w, h], dtype=float)
    arr = boxes[BOX].to_numpy(dtype=float) / scale
    dt = 1.0 / fps
    state = GimbalState(0.5, 0.5, 1.0)
    rows = []
    for frame, b in zip(boxes["frame"], arr, strict=True):
        state, cmd, clamped = step(state, None if np.isnan(b).any() else b, params, dt)
        rows.append((int(frame), int(frame) * dt, state.cx, state.cy, state.zoom,
                     cmd.pan_rate, cmd.tilt_rate, cmd.zoom_rate, clamped))
    cols = ["frame", "t", "cx", "cy", "zoom", "pan_rate", "tilt_rate", "zoom_rate", "clamped"]
    return pd.DataFrame(rows, columns=cols)


def gimbal_metrics(follow_df: pd.DataFrame, gt_track: pd.DataFrame, w: int, h: int) -> dict[str, float]:
    """Gimbal quality in native pixels.

    centre_err: mean distance of the GT box centre from the crop centre, per axis as a fraction of
    the crop size (crop = w/zoom x h/zoom), combined as a Euclidean norm; GT frames only.
    inside: fraction of GT frames whose GT box lies fully inside the crop.
    smooth: RMS of the second difference of the crop centre (normalized units / s^2), all frames.
    clamped_frac: fraction of all frames flagged clamped.
    """
    nan = float("nan")
    fd = follow_df.sort_values("frame").reset_index(drop=True)
    out = {"centre_err": nan, "inside": nan, "smooth": nan,
           "clamped_frac": float(fd["clamped"].mean()) if len(fd) else nan}
    if len(fd) >= 3:
        dt = float(np.median(np.diff(fd["t"].to_numpy())))
        ddx = np.diff(fd["cx"].to_numpy(), 2) / dt**2
        ddy = np.diff(fd["cy"].to_numpy(), 2) / dt**2
        out["smooth"] = float(np.sqrt(np.mean(ddx**2 + ddy**2)))
    gt = gt_track[["frame", *BOX]].dropna()
    m = fd.merge(gt, on="frame", how="inner")
    if m.empty:
        return out
    crop_w = w / m["zoom"].to_numpy()
    crop_h = h / m["zoom"].to_numpy()
    ccx = m["cx"].to_numpy() * w
    ccy = m["cy"].to_numpy() * h
    gx = (m["x1"].to_numpy() + m["x2"].to_numpy()) / 2
    gy = (m["y1"].to_numpy() + m["y2"].to_numpy()) / 2
    out["centre_err"] = float(np.mean(np.hypot((gx - ccx) / crop_w, (gy - ccy) / crop_h)))
    inside = (
        (m["x1"].to_numpy() >= ccx - crop_w / 2 - _EPS)
        & (m["x2"].to_numpy() <= ccx + crop_w / 2 + _EPS)
        & (m["y1"].to_numpy() >= ccy - crop_h / 2 - _EPS)
        & (m["y2"].to_numpy() <= ccy + crop_h / 2 + _EPS)
    )
    out["inside"] = float(inside.mean())
    return out


def render_crop(frame: np.ndarray, state: GimbalState, out_size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Resample the crop window of `frame` to `out_size` = (width, height)."""
    fh, fw = frame.shape[:2]
    cw, ch = fw / state.zoom, fh / state.zoom
    x0 = state.cx * fw - cw / 2
    y0 = state.cy * fh - ch / 2
    sx, sy = out_size[0] / cw, out_size[1] / ch
    m = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], dtype=np.float64)
    return cv2.warpAffine(frame, m, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
