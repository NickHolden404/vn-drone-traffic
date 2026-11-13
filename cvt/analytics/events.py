"""Stopped and wrong-way vehicle events from per-row kinematics, and per-frame scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvt.data.schema import CLASS_ID

VEHICLE_IDS = tuple(CLASS_ID[name] for name in ("car", "truck", "bus", "motorbike"))


def _is_vehicle(kin: pd.DataFrame) -> np.ndarray:
    return kin["label"].isin(VEHICLE_IDS).to_numpy()


def _track_rows(kin: pd.DataFrame):
    """Yield, per track, the row positions in frame order."""
    tid = kin["track_id"].to_numpy()
    frame = kin["frame"].to_numpy()
    order = np.lexsort((frame, tid))
    cuts = np.flatnonzero(np.diff(tid[order]) != 0) + 1
    yield from np.split(order, cuts)


def _runs(kin: pd.DataFrame, cond: np.ndarray, fps: float, gap_s: float):
    """Yield (row positions, frames) of each run of rows where `cond` holds, per track.

    Only a tracker gap is bridged: consecutive qualifying rows of a track stay in one run when
    everything between them is missing frames (no row) or rows with NaN speed, and they are at
    most `gap_s + 1 / fps` apart. A row with a finite speed where `cond` is false breaks the run
    at once, as does a longer gap.
    """
    frame = kin["frame"].to_numpy()
    nan_speed = ~np.isfinite(kin["speed_kmh"].to_numpy(float))
    max_step = gap_s * fps + 1 + 1e-9  # in frames
    for rows in _track_rows(kin):
        ok = cond[rows]
        if not ok.any():
            continue
        breaks = np.cumsum(~ok & ~nan_speed[rows])  # rows that break a run, counted so far
        q = rows[ok]
        f = frame[q]
        cut = (np.diff(f) > max_step) | (np.diff(breaks[ok]) > 0)
        cuts = np.flatnonzero(cut) + 1
        yield from zip(np.split(q, cuts), np.split(f, cuts), strict=True)


def _stop_cond(kin: pd.DataFrame, v_stop_kmh: float) -> np.ndarray:
    speed = kin["speed_kmh"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        return _is_vehicle(kin) & (speed < v_stop_kmh)  # NaN speed compares False


def stopped_score(kin, fps, v_stop_kmh, t_stop_s, t_sat_s, gap_s: float = 1.0) -> np.ndarray:
    """Per row, causal: a vehicle row of a stop run (speed < `v_stop_kmh`; missing frames and
    NaN-speed rows up to `gap_s` bridged, a moving row breaks it) scores min(1, elapsed / `t_sat_s`)
    once elapsed >= `t_stop_s`, 0 before; elapsed is the time since the run's first row, across
    bridged gaps."""
    out = np.zeros(len(kin))
    for rows, f in _runs(kin, _stop_cond(kin, v_stop_kmh), fps, gap_s):
        elapsed = (f - f[0]) / fps
        out[rows] = np.where(elapsed >= t_stop_s, np.minimum(1.0, elapsed / t_sat_s), 0.0)
    return out


def _cells(kin, x0, y0, cell_m):
    ix = np.floor((kin["xs_m"].to_numpy(float) - x0) / cell_m)
    iy = np.floor((kin["ys_m"].to_numpy(float) - y0) / cell_m)
    return ix, iy


def flow_field(kin, cell_m, v_move_kmh, min_passes, min_concentration) -> dict:
    """Dominant heading per ground cell; the arrays are indexed [ix, iy]."""
    xs = kin["xs_m"].to_numpy(float)
    ys = kin["ys_m"].to_numpy(float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        z = np.zeros((0, 0))
        return {"x0": 0.0, "y0": 0.0, "cell_m": cell_m, "ux": z, "uy": z, "ok": z.astype(bool)}
    x0, y0 = float(xs[finite].min()), float(ys[finite].min())
    nx = int(np.floor((xs[finite].max() - x0) / cell_m)) + 1
    ny = int(np.floor((ys[finite].max() - y0) / cell_m)) + 1
    ix, iy = _cells(kin, x0, y0, cell_m)
    head = kin["heading_rad"].to_numpy(float)
    speed = kin["speed_kmh"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        use = _is_vehicle(kin) & finite & (speed > v_move_kmh) & np.isfinite(head)
    votes = pd.DataFrame(
        {
            "track": kin["track_id"].to_numpy()[use],
            "ix": ix[use].astype(int),
            "iy": iy[use].astype(int),
            "c": np.cos(head[use]),
            "s": np.sin(head[use]),
        }
    )
    # one vote per track per cell: the mean of that track's unit headings there
    per_track = votes.groupby(["track", "ix", "iy"], as_index=False)[["c", "s"]].mean()
    per_cell = per_track.groupby(["ix", "iy"]).agg(n=("c", "size"), c=("c", "mean"), s=("s", "mean"))
    n = np.zeros((nx, ny))
    c = np.zeros((nx, ny))
    s = np.zeros((nx, ny))
    if len(per_cell):
        i = per_cell.index.get_level_values("ix").to_numpy()
        j = per_cell.index.get_level_values("iy").to_numpy()
        n[i, j], c[i, j], s[i, j] = per_cell["n"], per_cell["c"], per_cell["s"]
    length = np.hypot(c, s)
    ok = (n >= min_passes) & (length >= min_concentration)
    safe = np.where(length > 0, length, 1.0)
    return {"x0": x0, "y0": y0, "cell_m": cell_m, "ux": c / safe, "uy": s / safe, "ok": ok}


def wrong_way_score(kin, field, fps, v_move_kmh, angle_deg, t_wrong_s,
                    gap_s: float = 1.0) -> np.ndarray:
    """1 for rows of a run (lasting >= `t_wrong_s`) moving against the flow of an `ok` cell;
    missing frames and NaN-speed rows up to `gap_s` are bridged, any other row breaks the run."""
    out = np.zeros(len(kin))
    ok = field["ok"]
    if not ok.any():
        return out
    ix, iy = _cells(kin, field["x0"], field["y0"], field["cell_m"])
    with np.errstate(invalid="ignore"):
        inside = (ix >= 0) & (ix < ok.shape[0]) & (iy >= 0) & (iy < ok.shape[1])
    i = np.where(inside, ix, 0).astype(int)
    j = np.where(inside, iy, 0).astype(int)
    head = kin["heading_rad"].to_numpy(float)
    speed = kin["speed_kmh"].to_numpy(float)
    cos_angle = np.cos(np.deg2rad(angle_deg))
    with np.errstate(invalid="ignore"):
        cos_diff = np.cos(head) * field["ux"][i, j] + np.sin(head) * field["uy"][i, j]
        cond = _is_vehicle(kin) & inside & ok[i, j] & (speed > v_move_kmh) & (cos_diff < cos_angle)
    for rows, f in _runs(kin, cond, fps, gap_s):
        if (f[-1] - f[0] + 1) / fps >= t_wrong_s:
            out[rows] = 1.0
    return out


def frame_scores(kin, score: np.ndarray, n_frames: int) -> np.ndarray:
    """Maximum row score per frame, 0 where no row."""
    out = np.zeros(n_frames)
    frame = kin["frame"].to_numpy().astype(int)
    keep = (frame >= 0) & (frame < n_frames)
    np.maximum.at(out, frame[keep], np.asarray(score, float)[keep])
    return out


def event_table(kin, stopped, wrong, fps, gap_s: float = 1.0, *, v_stop_kmh) -> pd.DataFrame:
    """One row per event, per track and kind. A stopped event is a stop run (as in
    `stopped_score`) with a positive score and starts at the run's first row, when the vehicle
    stopped; a wrong-way event is a run of positive `wrong`. `t_end` is the last row's time plus one
    frame; only missing frames and NaN-speed rows up to `gap_s` are bridged."""
    t = kin["t_s"].to_numpy(float)
    label = kin["label"].to_numpy()
    tid = kin["track_id"].to_numpy()
    stopped = np.asarray(stopped, float)
    records = []
    for kind, cond, score in (("stopped", _stop_cond(kin, v_stop_kmh), stopped),
                              ("wrong_way", np.asarray(wrong, float) > 0, None)):
        for rows, _ in _runs(kin, cond, fps, gap_s):
            if score is not None and not (score[rows] > 0).any():
                continue
            t_start, t_end = t[rows[0]], t[rows[-1]] + 1.0 / fps
            records.append({"track_id": tid[rows[0]], "label": label[rows[0]], "kind": kind,
                            "t_start": t_start, "t_end": t_end, "duration_s": t_end - t_start})
    cols = ["track_id", "label", "kind", "t_start", "t_end", "duration_s"]
    table = pd.DataFrame(records, columns=cols)
    return table.sort_values(["t_start", "track_id"], kind="stable").reset_index(drop=True)
