"""Overlay video of a clip's traffic analytics: boxes, speeds, trails, events, arms, counts."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from cvt.analytics.od import arm_name, assign_arm, perimeter_point
from cvt.analytics.pipeline import ClipResult
from cvt.data.schema import CLASSES
from cvt.track.demo import CLASS_BGR, _open_writer, _text

Log = Callable[[str], None]

STOP_BGR = (0, 0, 255)
WRONG_BGR = (255, 0, 255)
ARM_BGR = (0, 255, 255)
WHITE = (255, 255, 255)
PANEL_ROWS = 8  # turning-count rows shown, the largest first
LOG_EVERY = 500


def _quiet(msg: str) -> None:
    pass


def scale_line(scale: dict) -> str:
    mpp = scale["mpp"]
    if not np.isfinite(mpp):
        return "uncalibrated"
    if scale["source"] == "telemetry":
        return f"{mpp * 100:.1f} cm/px, telemetry {scale['height_med']:.0f} m"
    return f"{mpp * 100:.1f} cm/px, car length (uncalibrated)"


def _exits(r: ClipResult, max_dist: float) -> dict[int, list[tuple[str, str, str]]]:
    """frame -> (origin arm, destination arm, class) of the complete tracks that exit there."""
    ep = r.endpoints[r.endpoints["complete"]]
    if ep.empty or len(r.arms) == 0:
        return {}
    o = assign_arm(ep["entry"].to_numpy(float), r.arms, max_dist)
    d = assign_arm(ep["exit"].to_numpy(float), r.arms, max_dist)
    out: dict[int, list[tuple[str, str, str]]] = {}
    for last, lab, oi, di in zip(ep["last"], ep["label"], o, d, strict=True):
        if oi >= 0 and di >= 0:
            out.setdefault(int(last), []).append(
                (arm_name(int(oi)), arm_name(int(di)), CLASSES[int(lab)])
            )
    return out


def _stop_starts(events) -> dict[int, np.ndarray]:
    """track -> sorted start times (s) of its stopped events."""
    if not len(events):
        return {}
    ev = events[events["kind"] == "stopped"]
    return {int(t): np.sort(g["t_start"].to_numpy(float)) for t, g in ev.groupby("track_id")}


def _stopped_for(starts: dict[int, np.ndarray], track: int, now: float) -> float:
    """Seconds since the latest stopped event of `track` starting at or before `now` began."""
    s = starts.get(track)
    j = -1 if s is None else int(np.searchsorted(s, now + 1e-9, side="right")) - 1
    return now - s[j] if j >= 0 else 0.0


def _arm_points(r: ClipResult, w: int, h: int, margin: int) -> list[tuple[str, int, int]]:
    if len(r.arms) == 0:
        return []
    x, y = perimeter_point(r.arms, w, h)
    x, y = np.clip(x, margin, w - margin), np.clip(y, margin, h - margin)
    return [(arm_name(i), int(a), int(b)) for i, (a, b) in enumerate(zip(x, y, strict=True))]


def _panel(img: np.ndarray, lines: list[str], k: float) -> None:
    pad, step = int(8 * k), int(24 * k)
    width = int(max(len(s) for s in lines) * 12 * k) + 2 * pad
    height = step * len(lines) + pad
    roi = img[:height, :width]
    roi[:] = (roi * 0.45).astype(np.uint8)
    for i, s in enumerate(lines):
        _text(img, s, (pad, step * (i + 1) - int(4 * k)), 0.6 * k, 1, WHITE)


def _annotated(
    video: Path, r: ClipResult, trail_s: float, max_frames: int | None, max_dist: float
) -> Iterator[np.ndarray]:
    """Yield each source frame with the overlay drawn; one decoded frame at a time."""
    w, h, fps = int(r.meta["width"]), int(r.meta["height"]), float(r.meta["fps"])
    n = int(r.meta["n_frames"])
    n = n if max_frames is None else min(max_frames, n)
    k = max(1.0, h / 720)  # smaller text blurs under its outline
    kin = r.kin
    rows = kin.groupby("frame").indices
    x1, y1, x2, y2 = (kin[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    tid, lab = kin["track_id"].to_numpy(), kin["label"].to_numpy()
    speed = kin["speed_kmh"].to_numpy(float)
    last_of = kin.groupby("track_id")["frame"].max().to_dict()
    exits = _exits(r, max_dist)
    arms = _arm_points(r, w, h, int(16 * k))
    head = [r.clip, "", scale_line(r.scale)]
    counts: Counter = Counter()
    trails: dict[int, deque] = {}
    stop_starts = _stop_starts(r.events)
    cap = cv2.VideoCapture(str(video))
    try:
        for f in range(n):
            ok, img = cap.read()
            if not ok:
                break
            for key in exits.get(f, ()):
                counts[key] += 1
            for i in rows.get(f, ()):
                t = int(tid[i])
                if not np.all(np.isfinite([x1[i], y1[i], x2[i], y2[i]])):
                    continue
                p1, p2 = (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i]))
                trail = trails.setdefault(t, deque(maxlen=max(2, round(trail_s * fps))))
                trail.append(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
                colour = CLASS_BGR[int(lab[i])]
                stopped, wrong = r.stopped[i] > 0, r.wrong[i] > 0
                if len(trail) > 1:
                    cv2.polylines(img, [np.array(trail, np.int32)], False, colour,
                                  max(1, round(2 * k)), cv2.LINE_AA)
                event = stopped or wrong
                cv2.rectangle(img, p1, p2, STOP_BGR if event else colour,
                              max(2, round(3 * k)) if event else max(1, round(2 * k)))
                text = f"{CLASSES[int(lab[i])]} #{t}"
                if np.isfinite(speed[i]):
                    text += f" {speed[i]:.0f} km/h"
                y = p1[1] - int(6 * k)
                _text(img, text, (p1[0], max(y, int(14 * k))), 0.5 * k, 1, colour)
                if stopped:
                    y -= int(18 * k)
                    _text(img, f"STOPPED {_stopped_for(stop_starts, t, f / fps):.0f} s",
                          (p1[0], max(y, int(14 * k))), 0.5 * k, 1, STOP_BGR)
                if wrong:
                    y -= int(18 * k)
                    _text(img, "WRONG WAY", (p1[0], max(y, int(14 * k))), 0.5 * k, 1, WRONG_BGR)
            for t in [t for t in trails if last_of.get(t, -1) <= f]:
                del trails[t]
            for name, ax, ay in arms:
                _text(img, name, (ax - int(8 * k), ay + int(8 * k)), 1.0 * k, 2, ARM_BGR)
            head[1] = f"t = {f / fps:.1f} s"
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:PANEL_ROWS]
            _panel(img, head + [f"{o}->{d} {c} {num}" for (o, d, c), num in top], k)
            yield img
    finally:
        cap.release()


def render_clip(
    video: Path, r: ClipResult, out: Path, *, trail_s: float = 3.0,
    max_frames: int | None = None, max_dist: float = 0.06, log: Log = _quiet,
) -> Path:
    """Write the overlay mp4 (same size and frame rate as the source); returns `out`."""
    video, out = Path(video), Path(out)
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")
    out.parent.mkdir(parents=True, exist_ok=True)
    size = (int(r.meta["width"]), int(r.meta["height"]))
    vw, codec = _open_writer(out, float(r.meta["fps"]), size)
    try:
        for f, img in enumerate(_annotated(video, r, trail_s, max_frames, max_dist)):
            vw.write(img)
            if (f + 1) % LOG_EVERY == 0:
                log(f"  {r.clip}: {f + 1} frames ({codec})")
    finally:
        vw.release()
    return out


def still(video: Path, r: ClipResult, frame: int, *, trail_s: float = 3.0,
          max_dist: float = 0.06) -> np.ndarray:
    """The overlay of one frame (trails and counts are replayed up to it)."""
    for f, img in enumerate(_annotated(Path(video), r, trail_s, frame + 1, max_dist)):
        if f == frame:
            return img
    raise ValueError(f"{video}: no frame {frame}")
