"""Line-crossing counts in stabilized (reference-frame) coordinates and counting error.

Direction convention: with d = p2 - p1 and v = point - p1, side = sign(cross(d, v)) where
cross(d, v) = d.x * v.y - d.y * v.x. In image coordinates (y down), "pos" means a track moved from
the negative side to the positive side of cross(d, v); "neg" is the reverse.

Counting rule 2 (rule 1 is the plain crossing count) adds two filters, applied alike
to GT and predicted tracks: `near_border` points are invalid, and `first_crossings` keeps one
event per track per line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.config import load_yaml

EVENT_COLS = ["track_id", "label", "frame", "direction"]
CELL_NAMES = ["label", "direction"]
# counting rule 2: border margin m = max(BORDER_MIN_PX, BORDER_FRAC * min(W, H))
BORDER_MIN_PX = 2.0
BORDER_FRAC = 0.01


@dataclass(frozen=True)
class Line:
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]


def _point(value, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where}: expected a 2-D point [x, y], got {value!r}")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: non-numeric point {value!r}") from e


def load_scene(path: Path) -> dict:
    """Load a scene yaml {seq_id, set_name, ref_frame, lines: [{name, p1, p2}]}.

    Lines become `Line` objects. Whether `seq_id` exists is left to the caller.
    """
    scene = load_yaml(path)
    missing = [k for k in ("seq_id", "set_name", "ref_frame", "lines") if k not in scene]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if not isinstance(scene["lines"], list):
        raise ValueError(f"{path}: 'lines' must be a list")
    lines = []
    for i, item in enumerate(scene["lines"]):
        if not isinstance(item, dict) or not {"name", "p1", "p2"} <= item.keys():
            raise ValueError(f"{path}: lines[{i}] needs keys name, p1, p2")
        p1 = _point(item["p1"], f"{path}: lines[{i}].p1")
        p2 = _point(item["p2"], f"{path}: lines[{i}].p2")
        if p1 == p2:
            raise ValueError(f"{path}: lines[{i}] ({item['name']}) needs two distinct points")
        lines.append(Line(str(item["name"]), p1, p2))
    return {**scene, "lines": lines}


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def crossings(points: pd.DataFrame, line: Line, deadband: float = 2.0) -> pd.DataFrame:
    """Crossing events of `line` by tracks; `points` has track_id, frame, x, y, label, valid.

    Per track, points are processed in frame order; invalid points are skipped, and points within
    `deadband` px of the (infinite) line have no side. An event fires when a point's non-zero
    side differs from the track's last non-zero side and the segment from the last-sided point to
    this point intersects the line segment p1..p2. Returns track_id, label, frame (the current
    point's), direction ("pos"/"neg", see the module docstring).
    """
    (x1, y1), (x2, y2) = line.p1, line.p2
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0.0:
        raise ValueError(f"line {line.name!r} needs two distinct points")
    events = []
    pts = points[points["valid"].astype(bool)].sort_values(["track_id", "frame"], kind="stable")
    for tid, g in pts.groupby("track_id", sort=False):
        last = None  # (x, y, side) of the last point with a non-zero side
        for frame, x, y, label in zip(g["frame"], g["x"], g["y"], g["label"], strict=True):
            c = _cross(dx, dy, x - x1, y - y1)
            if abs(c) / length < deadband:
                continue
            side = 1 if c > 0 else -1
            if last is not None and side != last[2]:
                ax, ay = last[0], last[1]
                # A and the current point lie strictly on opposite sides of the line; the
                # crossing is within p1..p2 iff p1 and p2 are not strictly on the same side of AB.
                o1 = _cross(x - ax, y - ay, x1 - ax, y1 - ay)
                o2 = _cross(x - ax, y - ay, x2 - ax, y2 - ay)
                if o1 * o2 <= 0:
                    events.append((tid, label, frame, "pos" if side > 0 else "neg"))
            last = (x, y, side)
    return pd.DataFrame(events, columns=EVENT_COLS)


def border_margin(width, height) -> np.ndarray:
    """Rule-2 border margin in px of a W x H image (element-wise)."""
    return np.maximum(BORDER_MIN_PX, BORDER_FRAC * np.minimum(width, height))


def near_border(x1, y1, x2, y2, width, height) -> np.ndarray:
    """Rule 2: True where a box (image coordinates, before stabilization) lies within the
    border margin m of its W x H image: x1 <= m, y1 <= m, x2 >= W - m or y2 >= H - m."""
    x1, y1, x2, y2 = (np.asarray(v, float) for v in (x1, y1, x2, y2))
    w, h = np.asarray(width, float), np.asarray(height, float)
    m = border_margin(w, h)
    return (x1 <= m) | (y1 <= m) | (x2 >= w - m) | (y2 >= h - m)


def first_crossings(events: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: only each track's first crossing event (by frame) of the line."""
    ev = events.sort_values(["track_id", "frame"], kind="stable")
    return ev.drop_duplicates("track_id").reset_index(drop=True)


def count_table(events: pd.DataFrame) -> pd.DataFrame:
    """Events -> counts with index (label, direction) and column n; only observed cells."""
    if events.empty:
        idx = pd.MultiIndex.from_arrays([[], []], names=CELL_NAMES)
        return pd.DataFrame({"n": pd.Series([], dtype=int)}, index=idx)
    return events.groupby(CELL_NAMES).size().rename("n").to_frame()


def count_errors(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    """Compare two count tables over the union of their cells (missing cells count as 0).

    count_MAE = mean |pred - gt| over the union (NaN if empty); count_pct_err =
    sum|pred - gt| / sum gt (NaN if sum gt = 0); plus gt_total and pred_total.
    """
    g, p = gt["n"].align(pred["n"], join="outer", fill_value=0)
    diff = (p - g).abs()
    gt_total, pred_total = float(g.sum()), float(p.sum())
    return {
        "count_MAE": float(diff.mean()) if len(diff) else float("nan"),
        "count_pct_err": float(diff.sum() / gt_total) if gt_total > 0 else float("nan"),
        "gt_total": gt_total,
        "pred_total": pred_total,
    }
