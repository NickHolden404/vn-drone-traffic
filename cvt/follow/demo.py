"""Follow demo: click a vehicle, lock it, and follow it with the virtual gimbal.

A segment of a drone video is read, detected and tracked live (BoT-SORT with GMC warps), the track
under the click at the segment's first frame is locked with the `reacq` variant (motion / geometry
re-acquisition, no appearance features, so no feature pass is needed) at the tuned lock point, and
the virtual gimbal follows the locked box at the tuned gimbal point. The output mp4 puts the full
frame (960 px wide, with the lock box, its source and the crop outline) next to the 960x540 follow
view. A CSV next to it logs the MAVLink-shaped command of every frame. Without a click the
target is picked automatically per vehicle class (`pick_auto`), one video per class from one
detection and tracking pass; `demo_plan` lays out one mid-clip segment per video for
`cvt follow demo-all`. Demo clips are never committed.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from cvt.data.schema import CLASS_ID, CLASSES
from cvt.follow import pipeline as fp
from cvt.follow.features import VEHICLE_IDS
from cvt.follow.gimbal import GimbalCommand, GimbalParams, GimbalState, follow, render_crop
from cvt.follow.lock import LockParams, lock_params, run_lock
from cvt.track.demo import _detect, _open_writer, _read_segment, _text
from cvt.track.pipeline import _uses_gmc
from cvt.track.run import gmc_warps, track_sequence

Log = Callable[[str], None]

BOX = ["x1", "y1", "x2", "y2"]
VARIANT = "reacq"
LEFT_W = 960
VIEW = (960, 540)
SRC_BGR = {"track": (60, 200, 60), "mil": (0, 165, 255), "lost": (60, 60, 230)}
CROP_BGR = (255, 255, 0)
EDGE_EXIT = 0.03  # a loss counts as an exit if the last held box was this close to the edge
AUTO_MARGIN = 0.1  # automatic target: box centre this far (frame fraction) from the edge
TUNE_HINT = r"run `scripts\follow.ps1 -Step tune` first"
SUMMARY_COLS = ["video", "t0_s", "pick", "status", "track_id", "label", "fps", "frames",
                "frac_track", "frac_mil", "frac_lost", "frac_lost_exit",
                "frac_lost_drop", "drops", "reacquisitions", "travel_diag",
                "clamped_frac", "mean_zoom", "out"]
LOG_COLS = ["frame", "t", "source", "track_id", *BOX, "cx", "cy", "zoom", "pan_rate",
            "tilt_rate", "zoom_rate", "yaw_rate", "pitch_rate", "clamped"]


def _quiet(msg: str) -> None:
    pass


def load_demo_params(config: Path, tuned: Path) -> tuple[LockParams, GimbalParams, float]:
    """(`reacq` lock params, gimbal params, hfov) at the tuned points of `cvt follow tune`."""
    if not Path(tuned).exists():
        raise FileNotFoundError(f"{tuned} not found: {TUNE_HINT}")
    cfg = fp.load_follow_cfg(config)
    params = yaml.safe_load(Path(tuned).read_text(encoding="utf-8"))
    fp.check_tuned(cfg, params)
    return (lock_params(VARIANT, **params["lock"]), fp.gimbal_params(cfg, params["gimbal"]),
            float(cfg["hfov_deg"]))


def select_target(tracks: pd.DataFrame, frame: int, click_xy) -> tuple[int, np.ndarray]:
    """(track id, box) of the track whose box contains the click in `frame` (the smallest such
    box), else the box whose centre is nearest the click."""
    t = tracks[tracks["frame"] == frame]
    if t.empty:
        raise ValueError(f"no track in frame {frame}: nothing to lock (click {tuple(click_xy)})")
    b = t[BOX].to_numpy(float)
    x, y = float(click_xy[0]), float(click_xy[1])
    inside = (b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])
    if inside.any():
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        k = int(np.flatnonzero(inside)[np.argmin(area[inside])])
    else:
        d = np.hypot((b[:, 0] + b[:, 2]) / 2 - x, (b[:, 1] + b[:, 3]) / 2 - y)
        k = int(np.argmin(d))
    return int(t["track_id"].iloc[k]), b[k]


def _compose(im: np.ndarray, row, state: GimbalState, left_h: int, height: int) -> np.ndarray:
    s = LEFT_W / im.shape[1]
    left = cv2.resize(im, (LEFT_W, left_h),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    src = row["source"]
    if src != "lost":
        p1 = (int(round(row["x1"] * s)), int(round(row["y1"] * s)))
        p2 = (int(round(row["x2"] * s)), int(round(row["y2"] * s)))
        cv2.rectangle(left, p1, p2, SRC_BGR[src], 2)
    cw, ch = LEFT_W / state.zoom, left_h / state.zoom
    c1 = (int(round(state.cx * LEFT_W - cw / 2)), int(round(state.cy * left_h - ch / 2)))
    c2 = (int(round(state.cx * LEFT_W + cw / 2)), int(round(state.cy * left_h + ch / 2)))
    cv2.rectangle(left, c1, c2, CROP_BGR, 1)
    label = f"lock: {src}" + (f" #{int(row['track_id'])}" if src == "track" else "")
    _text(left, label, (12, 28), 0.8, 2, SRC_BGR[src])
    out = np.zeros((height, LEFT_W + VIEW[0], 3), np.uint8)
    out[:left_h, :LEFT_W] = left
    y0 = (height - VIEW[1]) // 2
    out[y0:y0 + VIEW[1], LEFT_W:] = render_crop(im, state, VIEW)
    return out


def pick_auto(tracks: pd.DataFrame, frame: int, size: tuple[int, int], label: int | None = None,
              margin: float = AUTO_MARGIN) -> tuple[int, np.ndarray] | None:
    """(track id, box) of the target to lock without a click, or None if there is no candidate.

    Candidates: vehicle tracks (`label` only, if given) present in `frame` whose box centre lies
    inside the frame minus `margin` on each side (room for the gimbal to follow). Picked: the track
    the tracker keeps longest in the segment, then the one travelling farthest (a moving vehicle),
    then the lowest track id."""
    w, h = size
    t = tracks[tracks["frame"] == frame]
    t = t[t["label"].isin([label] if label is not None else list(VEHICLE_IDS))]
    cx, cy = (t["x1"] + t["x2"]) / 2, (t["y1"] + t["y2"]) / 2
    t = t[(cx >= margin * w) & (cx <= (1 - margin) * w)
          & (cy >= margin * h) & (cy <= (1 - margin) * h)]
    if t.empty:
        return None
    best = None
    for tid in t["track_id"].unique():
        g = tracks[tracks["track_id"] == tid].sort_values("frame")
        c = np.c_[(g["x1"] + g["x2"]) / 2, (g["y1"] + g["y2"]) / 2]
        path = float(np.hypot(*np.diff(c, axis=0).T).sum()) if len(c) > 1 else 0.0
        key = (-len(g), -path, int(tid))
        if best is None or key < best[0]:
            best = (key, int(tid))
    tid = best[1]
    return tid, t[t["track_id"] == tid][BOX].to_numpy(float)[0]


def _prepare(video: Path, t0_s: float, seconds: float, work: Path, detector, tracker_cfg: dict,
             log: Log):
    """Read, detect and track the segment once: (frames, fps, (w, h), tracks, warps)."""
    if not hasattr(detector, "predict_arrays"):
        raise ValueError(f"{type(detector).__name__} has no predict_arrays")
    frames, fps, (w, h) = _read_segment(video, t0_s, seconds, work, log)
    n = len(frames)
    dets = _detect(frames, detector, log)
    warps = None
    if _uses_gmc(tracker_cfg):
        log(f"GMC {tracker_cfg['gmc_method']} over {n} frames")
        warps = gmc_warps(frames.paths, method=tracker_cfg["gmc_method"])
    tracks = track_sequence(dets, list(range(n)), tracker_cfg, seq_id="demo", fps=fps,
                            warps=warps)
    return frames, fps, (w, h), tracks, warps


def _render_one(frames, fps: float, size: tuple[int, int], tracks: pd.DataFrame, warps, tid: int,
                box: np.ndarray, out: Path, *, lock: LockParams, gimbal: GimbalParams,
                hfov_deg: float, log: Log) -> dict:
    """Lock the track `tid` (box `box` in frame 0), render `out` and its command log; returns
    the run's summary figures (`run_summary`)."""
    w, h = size
    n = len(frames)
    locked = run_lock(tracks, lambda f, b: None, warps, list(range(n)), 0, box, lock, fps,
                      frames.get)
    gim = follow(locked[["frame", *BOX]], w, h, fps, gimbal)

    left_h = max(2, round(h * LEFT_W / w) // 2 * 2)
    height = max(left_h, VIEW[1]) // 2 * 2
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, codec = _open_writer(out, fps, (LEFT_W + VIEW[0], height))
    try:
        for i in range(n):
            g = gim.iloc[i]
            state = GimbalState(float(g["cx"]), float(g["cy"]), float(g["zoom"]))
            writer.write(_compose(frames.get(i), locked.iloc[i], state, left_h, height))
    finally:
        writer.release()

    cmds = [GimbalCommand(float(a), float(b), float(c)).to_mavlink(hfov_deg, w / h)
            for a, b, c in zip(gim["pan_rate"], gim["tilt_rate"], gim["zoom_rate"], strict=True)]
    table = pd.DataFrame({
        "frame": gim["frame"], "t": gim["t"], "source": locked["source"].to_numpy(),
        "track_id": locked["track_id"].to_numpy(),
        **{c: locked[c].to_numpy() for c in BOX},
        "cx": gim["cx"], "cy": gim["cy"], "zoom": gim["zoom"],
        "pan_rate": gim["pan_rate"], "tilt_rate": gim["tilt_rate"],
        "zoom_rate": gim["zoom_rate"],
        "yaw_rate": [c["yaw_rate"] for c in cmds],
        "pitch_rate": [c["pitch_rate"] for c in cmds],
        "clamped": gim["clamped"],
    })[LOG_COLS]
    table.to_csv(out.with_suffix(".csv"), index=False, lineterminator="\n")
    log(f"wrote {out} ({n} frames, {codec}) and {out.with_suffix('.csv').name}")
    return run_summary(table, size)


def run_summary(log_table: pd.DataFrame, size: tuple[int, int]) -> dict:
    """Figures for picking demos, from a command log: share of frames held by the tracker, by MIL
    and lost; re-acquisitions (the locked track id changes to another track); target travel in
    frame diagonals (held frames only); share of frames with the crop clamped at the frame edge;
    mean zoom. Lost frames are split by where the target was last held: at the frame edge
    (`frac_lost_exit`: it left the view, which a crop cannot follow) or inside the frame
    (`frac_lost_drop`: a real loss); `drops` counts those mid-frame losses."""
    src = log_table["source"].astype(str)
    n = max(1, len(log_table))
    w, h = size
    exit_frames = drop_frames = drops = 0
    lost = (src == "lost").to_numpy()
    i = 0
    while i < len(lost):
        if not lost[i]:
            i += 1
            continue
        j = i
        while j < len(lost) and lost[j]:
            j += 1
        prev = log_table.iloc[i - 1] if i > 0 else None
        at_edge = prev is not None and (
            prev["x1"] <= EDGE_EXIT * w or prev["x2"] >= (1 - EDGE_EXIT) * w
            or prev["y1"] <= EDGE_EXIT * h or prev["y2"] >= (1 - EDGE_EXIT) * h)
        if at_edge:
            exit_frames += j - i
        else:
            drop_frames += j - i
            drops += 1
        i = j
    ids = log_table.loc[src == "track", "track_id"].dropna().astype(int).to_numpy()
    held = log_table[src != "lost"]
    c = np.c_[(held["x1"] + held["x2"]) / 2, (held["y1"] + held["y2"]) / 2]
    travel = float(np.hypot(*np.diff(c, axis=0).T).sum()) if len(c) > 1 else 0.0
    return {
        "frames": len(log_table),
        "frac_track": float((src == "track").sum() / n),
        "frac_mil": float((src == "mil").sum() / n),
        "frac_lost": float((src == "lost").sum() / n),
        "frac_lost_exit": exit_frames / n,
        "frac_lost_drop": drop_frames / n,
        "drops": drops,
        "reacquisitions": int((np.diff(ids) != 0).sum()) if len(ids) > 1 else 0,
        "travel_diag": travel / float(np.hypot(*size)),
        "clamped_frac": float(log_table["clamped"].astype(bool).mean()) if len(log_table) else 0.0,
        "mean_zoom": float(log_table["zoom"].mean()) if len(log_table) else 1.0,
    }


def _pick_name(pick) -> str:
    return pick if isinstance(pick, str) else f"click {pick[0]:g},{pick[1]:g}"


def _target(tracks: pd.DataFrame, size: tuple[int, int], pick) -> tuple[int, np.ndarray] | None:
    """`pick`: an (x, y) click, or a class name / 'any' for the automatic choice."""
    if isinstance(pick, str):
        if pick != "any" and pick not in CLASS_ID:
            raise ValueError(f"unknown class {pick!r}: expected one of {CLASSES} or 'any'")
        return pick_auto(tracks, 0, size, None if pick == "any" else CLASS_ID[pick])
    if tracks[tracks["frame"] == 0].empty:
        return None
    return select_target(tracks, 0, pick)


def render_follow(
    video: Path, click_xy, t0_s: float, seconds: float, out: Path, *, detector, tracker_cfg: dict,
    lock: LockParams, gimbal: GimbalParams, hfov_deg: float,
    tmp_root: Path = Path("data/cache/demo"), keep_frames: bool = False, log: Log = _quiet,
) -> Path:
    """Render the side-by-side follow video `out` and the command log (`out` with suffix .csv).

    `click_xy` is in source-frame pixels of the segment's first frame (at `t0_s`), or a class
    name / 'any' to pick the target automatically (`pick_auto`). `detector` is anything with
    `predict_arrays(list of BGR arrays) -> list[Detections]`. Returns `out`."""
    rows = render_follows(video, [click_xy], t0_s, seconds, [out], detector=detector,
                          tracker_cfg=tracker_cfg, lock=lock, gimbal=gimbal, hfov_deg=hfov_deg,
                          tmp_root=tmp_root, keep_frames=keep_frames, log=log, strict=True)
    return Path(rows[0]["out"])


def render_follows(
    video: Path, picks: list, t0_s: float, seconds: float, outs: list[Path], *, detector,
    tracker_cfg: dict, lock: LockParams, gimbal: GimbalParams, hfov_deg: float,
    tmp_root: Path = Path("data/cache/demo"), keep_frames: bool = False, log: Log = _quiet,
    strict: bool = False,
) -> list[dict]:
    """One detection and tracking pass over the segment, then one follow video per pick (a click,
    or a class name / 'any'), written to the matching path of `outs`. Each run locks a different
    vehicle: a track locked by an earlier pick is not picked again. Returns one summary row per
    pick (`status` 'ok' or 'no candidate'); `strict` raises instead of skipping a pick that has
    no candidate."""
    video = Path(video)
    if len(picks) != len(outs):
        raise ValueError(f"{len(picks)} picks but {len(outs)} output paths")
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{video.stem}_", dir=tmp_root))
    try:
        frames, fps, size, tracks, warps = _prepare(video, t0_s, seconds, work, detector,
                                                    tracker_cfg, log)
        rows, used = [], set()
        for pick, out in zip(picks, outs, strict=True):
            out = Path(out)
            base = {"video": video.name, "t0_s": t0_s, "pick": _pick_name(pick), "out": str(out)}
            hit = _target(tracks[~tracks["track_id"].isin(list(used))], size, pick)
            if hit is None:
                if strict:
                    raise ValueError(f"no track to lock for {_pick_name(pick)} in the first frame")
                log(f"{video.name}: no {_pick_name(pick)} to lock in the first frame")
                rows.append({**base, "status": "no candidate"})
                continue
            tid, box = hit
            used.add(tid)
            name = CLASSES[int(tracks.loc[tracks["track_id"] == tid, "label"].iloc[0])]
            log(f"locking track {tid} ({name}) for {_pick_name(pick)}")
            s = _render_one(frames, fps, size, tracks, warps, tid, box, out, lock=lock,
                            gimbal=gimbal, hfov_deg=hfov_deg, log=log)
            rows.append({**base, "status": "ok", "track_id": tid, "label": name, "fps": fps, **s})
        return rows
    finally:
        if not keep_frames:
            shutil.rmtree(work, ignore_errors=True)


def demo_plan(videos: list[Path], seconds: float, classes: list[str], out_dir: Path) -> list[dict]:
    """One segment per video, from its middle (the whole clip if shorter than `seconds`), and one
    output per class: [{video, t0_s, picks, outs}]."""
    plan = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        try:
            fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        dur = n / fps if fps > 0 else 0.0
        if dur <= 0:
            raise ValueError(f"{v}: cannot read its duration")
        t0 = round(max(0.0, (dur - seconds) / 2), 1)
        plan.append({"video": Path(v), "t0_s": t0, "picks": list(classes),
                     "outs": [Path(out_dir) / f"{Path(v).stem}_{c}.mp4" for c in classes]})
    return plan
