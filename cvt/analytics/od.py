"""Junction arms from track entry / exit points on the frame border, and turning-movement counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

_GRID = 720


def perimeter_pos(x, y, width, height) -> np.ndarray:
    """Nearest border point as a fraction of the perimeter, clockwise from the top-left corner."""
    x = np.asarray(x, dtype=float)
    width = np.broadcast_to(np.asarray(width, dtype=float), x.shape)
    height = np.broadcast_to(np.asarray(height, dtype=float), x.shape)
    x = np.clip(x, 0.0, width)
    y = np.clip(np.asarray(y, dtype=float), 0.0, height)
    dist = np.stack([y, width - x, height - y, x])  # top, right, bottom, left
    edge = dist.argmin(axis=0)
    along = np.select(
        [edge == 0, edge == 1, edge == 2],
        [x, width + y, width + height + (width - x)],
        2 * width + height + (height - y),
    )
    return (along / (2 * (width + height))) % 1.0


def perimeter_point(pos, width, height) -> tuple[np.ndarray, np.ndarray]:
    """Border point (x, y) of a perimeter fraction: the inverse of `perimeter_pos`."""
    pos = np.atleast_1d(np.asarray(pos, dtype=float))
    along = (pos % 1.0) * 2 * (width + height)
    w, h = float(width), float(height)
    x = np.select([along < w, along < w + h, along < 2 * w + h], [along, w, w + h + w - along], 0.0)
    y = np.select(
        [along < w, along < w + h, along < 2 * w + h],
        [0.0, along - w, h],
        2 * w + 2 * h - along,
    )
    return x, y


def _size_at(sizes, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(sizes, pd.DataFrame):
        table = sizes.drop_duplicates("frame").set_index("frame")
        w = table["width"].reindex(frames).to_numpy(float)
        h = table["height"].reindex(frames).to_numpy(float)
        return w, h
    w, h = sizes
    return np.full(len(frames), float(w)), np.full(len(frames), float(h))


def _edge_pos(rows: pd.DataFrame, sizes, border_frac: float, valid: np.ndarray) -> np.ndarray:
    """Perimeter position of each row's box centre, NaN unless near the border and `valid`."""
    w, h = _size_at(sizes, rows["frame"].to_numpy())
    cx = ((rows["x1"] + rows["x2"]) / 2).to_numpy(float)
    cy = ((rows["y1"] + rows["y2"]) / 2).to_numpy(float)
    near = np.minimum.reduce([cx, w - cx, cy, h - cy]) <= border_frac * np.minimum(w, h)
    return np.where(near & valid, perimeter_pos(cx, cy, w, h), np.nan)


def endpoints(
    tracks: pd.DataFrame,
    sizes: tuple[int, int] | pd.DataFrame,
    first_frame: int,
    last_frame: int,
    border_frac: float,
    min_frames: int = 0,
) -> pd.DataFrame:
    """One row per track: entry / exit perimeter positions (NaN when not at the border, and for
    tracks spanning fewer than `min_frames` frames, which are never complete)."""
    cols = ["track_id", "label", "first", "last", "entry", "exit", "complete"]
    if tracks.empty:
        return pd.DataFrame({c: [] for c in cols}).astype({"complete": bool})
    srt = tracks.sort_values(["track_id", "frame"])
    g = srt.groupby("track_id", sort=True)
    head, tail = g.head(1).reset_index(drop=True), g.tail(1).reset_index(drop=True)
    first, last = head["frame"].to_numpy(), tail["frame"].to_numpy()
    long_enough = last - first + 1 >= min_frames
    entry = _edge_pos(head, sizes, border_frac, (first > first_frame) & long_enough)
    exit_ = _edge_pos(tail, sizes, border_frac, (last < last_frame) & long_enough)
    return pd.DataFrame(
        {
            "track_id": head["track_id"].to_numpy(),
            "label": head["label"].to_numpy(),
            "first": first,
            "last": last,
            "entry": entry,
            "exit": exit_,
            "complete": np.isfinite(entry) & np.isfinite(exit_),
        }
    )


def _circ(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def find_arms(pos: np.ndarray, bandwidth: float, min_sep: float, min_tracks: int) -> np.ndarray:
    """Arm centres (sorted): peaks of a circular Gaussian density with enough nearby positions."""
    pos = np.asarray(pos, dtype=float)
    pos = pos[np.isfinite(pos)]
    if len(pos) == 0:
        return np.empty(0)
    grid = np.arange(_GRID) / _GRID
    dens = np.exp(-0.5 * (_circ(grid[:, None], pos[None, :]) / bandwidth) ** 2).sum(axis=1)
    peaks = np.flatnonzero((dens >= np.roll(dens, 1)) & (dens > np.roll(dens, -1)))
    arms: list[float] = []
    for i in peaks[np.argsort(-dens[peaks], kind="stable")]:
        c = grid[i]
        if any(_circ(c, a) < min_sep for a in arms):
            continue
        if (_circ(c, pos) <= min_sep / 2).sum() >= min_tracks:
            arms.append(c)
    return np.array(sorted(arms), dtype=float)


def assign_arm(pos: np.ndarray, arms: np.ndarray, max_dist: float) -> np.ndarray:
    """Index of the nearest arm within `max_dist` (circular), else -1 (also for NaN)."""
    pos = np.asarray(pos, dtype=float)
    out = np.full(pos.shape, -1, dtype=int)
    if len(arms) == 0:
        return out
    d = _circ(pos[:, None], np.asarray(arms, dtype=float)[None, :])
    nearest = d.argmin(axis=1)
    ok = np.isfinite(pos) & (d[np.arange(len(pos)), nearest] <= max_dist)
    out[ok] = nearest[ok]
    return out


def _assigned(ep: pd.DataFrame, arms: np.ndarray, max_dist: float) -> pd.DataFrame:
    done = ep[ep["complete"]]
    a = pd.DataFrame(
        {
            "label": done["label"].to_numpy(),
            "origin": assign_arm(done["entry"].to_numpy(), arms, max_dist),
            "dest": assign_arm(done["exit"].to_numpy(), arms, max_dist),
        }
    )
    return a[(a["origin"] >= 0) & (a["dest"] >= 0)]


def od_table(ep: pd.DataFrame, arms: np.ndarray, max_dist: float) -> pd.DataFrame:
    """Counts `n` indexed by (label, origin, dest) over complete tracks with both ends on an arm."""
    keys = ["label", "origin", "dest"]
    a = _assigned(ep, arms, max_dist)
    if a.empty:
        idx = pd.MultiIndex.from_arrays([[], [], []], names=keys)
        return pd.DataFrame({"n": pd.Series([], dtype=int, index=idx)})
    return a.groupby(keys).size().rename("n").to_frame()


def od_summary(ep: pd.DataFrame, arms: np.ndarray, max_dist: float) -> dict:
    tracks, complete = len(ep), int(ep["complete"].sum())
    return {
        "tracks": tracks,
        "complete": complete,
        "complete_frac": complete / tracks if tracks else float("nan"),
        "assigned": len(_assigned(ep, arms, max_dist)),
        "arms": len(arms),
    }


def arm_name(i: int) -> str:
    return chr(ord("A") + i)
