"""Target lock: one target box (or "lost") per frame from tracker output.

The lock follows one tracker ID. While that ID is absent the lock is LOST:
- with `use_mil`, an OpenCV MIL tracker on raw frames, initialised from the last locked box on the
  last locked frame, holds a box for at most `coast_s`; after that it is dropped and the output is
  NaN ("lost"), never a stale box;
- with `reacquire`, a *new* track (one that never shared a frame with the target while locked) is
  adopted when it lies inside the motion gate, has the target's class (unless unknown) and an
  area within x/÷ 2 of the target's last area; appearance then picks among those and vetoes;
- the target's own ID coming back is always re-adopted, in every variant.

The motion gate compensates camera motion with BoT-SORT's GMC warps (`warps[i]` maps frame i-1
coordinates to frame i): the last locked centre is advanced by its camera-compensated velocity
(mean of the last 0.5 s of frames locked on the current ID; re-acquiring a different ID starts a
fresh history) and mapped through each frame's warp. Time is in seconds
from frame numbers and `fps`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import pandas as pd

Feats = Callable[[int, np.ndarray], "np.ndarray | None"]
ReadFrame = Callable[[int], np.ndarray]

BOX = ["x1", "y1", "x2", "y2"]
INIT_IOU = 0.3
VEL_WINDOW_S = 0.5
AREA_RATIO = 2.0
HSV_BINS = (16, 4)
_EPS = 1e-9


@dataclass(frozen=True)
class LockParams:
    coast_s: float
    reid_thr: float
    gate_growth: float  # gate radius grows by this many box diagonals per second lost
    use_mil: bool
    reacquire: bool
    appearance: Literal["none", "features", "hsv"]


VARIANTS: dict[str, dict] = {
    "id_only": dict(use_mil=False, reacquire=False, appearance="none"),
    "mil": dict(use_mil=True, reacquire=False, appearance="none"),
    "reacq": dict(use_mil=True, reacquire=True, appearance="none"),
    "features": dict(use_mil=True, reacquire=True, appearance="features"),
    "hsv": dict(use_mil=True, reacquire=True, appearance="hsv"),
}


def lock_params(variant: str, coast_s: float, reid_thr: float, gate_growth: float) -> LockParams:
    """The ablation variant (`VARIANTS`) at one tuning point."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown lock variant {variant!r} ({sorted(VARIANTS)})")
    return LockParams(coast_s, reid_thr, gate_growth, **VARIANTS[variant])


def make_mil():
    """OpenCV 5.0 main-module MIL tracker (no contrib, no model download); tests monkeypatch it."""
    return cv2.TrackerMIL_create()


# --- pure gating helpers ----------------------------------------------------------------------

def centres(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def areas(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def diagonal(box: np.ndarray) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, float).reshape(-1, 4)
    iw = np.minimum(box[2], b[:, 2]) - np.maximum(box[0], b[:, 0])
    ih = np.minimum(box[3], b[:, 3]) - np.maximum(box[1], b[:, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = areas(box[None])[0] + areas(b) - inter
    return np.where(union > 0, inter / np.maximum(union, _EPS), 0.0)


def init_index(boxes: np.ndarray, init_box: np.ndarray) -> int | None:
    """Max IoU >= 0.3, else the nearest centre within one diagonal of `init_box`, else None."""
    if len(boxes) == 0:
        return None
    iou = iou_one(init_box, boxes)
    if iou.max() >= INIT_IOU:
        return int(iou.argmax())
    d = np.hypot(*(centres(boxes) - centres(init_box[None])).T)
    j = int(d.argmin())
    return j if d[j] <= diagonal(init_box) else None


def apply_warp(w: np.ndarray | None, p: np.ndarray) -> np.ndarray:
    return p if w is None else w[:, :2] @ p + w[:, 2]


def invert_warp(w: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(w, np.float64))


def mean_velocity(hist: list[tuple[int, int, np.ndarray]], warps: np.ndarray | None,
                  window_frames: float) -> np.ndarray:
    """Camera-compensated velocity (px per frame) over the last `window_frames` of the history.

    `hist` holds (list index, frame number, centre) of locked frames. Only pairs adjacent in the
    frame list are used; each step is `W_k^-1(c_k) - c_{k-1}` over its frame gap."""
    if not hist:
        return np.zeros(2)
    last = hist[-1][1]
    steps = []
    for (i0, f0, c0), (i1, f1, c1) in zip(hist[:-1], hist[1:], strict=True):
        if i1 != i0 + 1 or f1 <= last - window_frames or f1 <= f0:
            continue
        c1s = c1 if warps is None else apply_warp(invert_warp(warps[i1]), c1)
        steps.append((c1s - c0) / (f1 - f0))
    return np.mean(steps, axis=0) if steps else np.zeros(2)


def gate_radius(diag: float, gate_growth: float, t_lost_s: float) -> float:
    return diag * (1.0 + gate_growth * t_lost_s)


def gate_mask(cand_centres: np.ndarray, cand_areas: np.ndarray, cand_labels: np.ndarray,
              pred: np.ndarray, radius: float, ref_area: float, cls: int | None) -> np.ndarray:
    """Inside the gate, same class (unless `cls` is None) and area within x/÷ AREA_RATIO."""
    d = np.hypot(*(cand_centres - pred).T) if len(cand_centres) else np.zeros(0)
    ok = (d <= radius) & (cand_areas <= ref_area * AREA_RATIO) & (cand_areas >= ref_area / AREA_RATIO)
    if cls is not None:
        ok &= cand_labels == cls
    return ok


def pick(dist: np.ndarray, sims: np.ndarray | None, thr: float) -> int | None:
    """Nearest when `sims` is None; else the max similarity >= thr (ties -> nearer), or None."""
    if len(dist) == 0:
        return None
    if sims is None:
        return int(np.argmin(dist))
    ok = np.isfinite(sims) & (sims >= thr)
    if not ok.any():
        return None
    order = np.lexsort((dist, -np.where(ok, sims, -np.inf)))
    return int(order[0])


def hsv_hist(img: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """(16, 4) H-S histogram of the box crop, summing to 1; None for an empty crop."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(img[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HSV_BINS), [0, 180, 0, 256]).astype(np.float64)
    s = hist.sum()
    return hist / s if s > 0 else None


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Bhattacharyya distance."""
    d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return 1.0 - float(d)


# --- state machine ----------------------------------------------------------------------------

class _Lock:
    """TRACK (on `tid`) / LOST (gate, optional MIL until `coast_s`, then no box)."""

    def __init__(self, p: LockParams, fps: float, frames: list[int], warps: np.ndarray | None,
                 feats: Feats, read_frame: ReadFrame | None):
        self.p, self.fps, self.frames, self.warps = p, fps, frames, warps
        self.feats, self.read_frame = feats, read_frame
        self.tid: int | None = None
        self.cls: int | None = None
        self.hist: list[tuple[int, int, np.ndarray]] = []
        self.coexist: set[int] = set()
        self.anchor_box = np.full(4, np.nan)  # last locked box (or the initial box)
        self.anchor_frame = 0
        self.pred = np.zeros(2)
        self.vel: np.ndarray | None = None
        self.mil = None
        self.mil_tried = False
        self.app_sum: np.ndarray | None = None
        self._img: tuple[int, np.ndarray] | None = None

    def _frame(self, f: int) -> np.ndarray:
        if self._img is None or self._img[0] != f:
            self._img = (f, self.read_frame(f))
        return self._img[1]

    # appearance ------------------------------------------------------------------------------
    def _update_appearance(self, f: int, box: np.ndarray) -> None:
        if self.p.appearance == "features":
            rows = self.feats(f, box[None])
            v = None if rows is None else np.asarray(rows, np.float64).reshape(-1)
            if v is None or not np.isfinite(v).all() or np.linalg.norm(v) < _EPS:
                return
            v = v / np.linalg.norm(v)
        elif self.p.appearance == "hsv":
            v = hsv_hist(self._frame(f), box)
            if v is None:
                return
        else:
            return
        self.app_sum = v if self.app_sum is None else self.app_sum + v

    def _similarities(self, f: int, boxes: np.ndarray) -> np.ndarray | None:
        """Per candidate similarity to the running mean; None (motion only) without a target model."""
        if self.p.appearance == "none" or self.app_sum is None:
            return None
        if self.p.appearance == "features":
            rows = self.feats(f, boxes)
            if rows is None:
                return np.full(len(boxes), np.nan)
            r = np.asarray(rows, np.float64).reshape(len(boxes), -1)
            r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), _EPS)
            m = self.app_sum / np.linalg.norm(self.app_sum)
            return r @ m
        img = self._frame(f)
        mean = self.app_sum / self.app_sum.sum()
        hs = [hsv_hist(img, b) for b in boxes]
        return np.array([np.nan if h is None else hist_similarity(mean, h) for h in hs])

    # transitions -----------------------------------------------------------------------------
    def lock(self, i: int, tid: int, box: np.ndarray, label: int, ids: np.ndarray) -> None:
        f = self.frames[i]
        if self.tid is not None and tid != self.tid:
            self.hist = []  # another object: its velocity must not mix with the previous one's
        self.tid = tid
        if self.cls is None:
            self.cls = int(label)
        self.coexist.update(int(t) for t in ids if t != tid)
        self.hist = [h for h in self.hist if h[1] > f - VEL_WINDOW_S * self.fps - 1]
        self.hist.append((i, f, centres(box)[0]))
        self.anchor_box, self.anchor_frame = box, f
        self.pred, self.vel = centres(box)[0], None
        self.mil, self.mil_tried = None, False
        self._update_appearance(f, box)

    def start_mil_only(self, i: int, box: np.ndarray) -> np.ndarray | None:
        """No track at init: anchor the gate at the initial box (zero velocity), start MIL."""
        self.anchor_box, self.anchor_frame = box, self.frames[i]
        self.pred, self.vel = centres(box)[0], np.zeros(2)
        if self._mil_available():
            self.mil_tried = True
            self.mil = self._init_mil(self.frames[i], box)
            if self.mil is not None:
                return box
        return None

    def _mil_available(self) -> bool:
        return self.p.use_mil and self.read_frame is not None

    def _init_mil(self, f: int, box: np.ndarray):
        img = self._frame(f)
        h, w = img.shape[:2]
        x1, y1 = int(round(max(0.0, box[0]))), int(round(max(0.0, box[1])))
        x2, y2 = int(round(min(float(w), box[2]))), int(round(min(float(h), box[3])))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        mil = make_mil()
        mil.init(img, (x1, y1, x2 - x1, y2 - y1))
        return mil

    def _advance(self, i: int) -> None:
        if self.vel is None:
            self.vel = mean_velocity(self.hist, self.warps, VEL_WINDOW_S * self.fps)
        dt = self.frames[i] - self.frames[i - 1]
        w = None if self.warps is None else self.warps[i]
        self.pred = apply_warp(w, self.pred + self.vel * dt)

    def _reacquire(self, i: int, ids, boxes, labels, t_lost: float) -> int | None:
        new = np.array([int(t) not in self.coexist for t in ids], bool)
        radius = gate_radius(diagonal(self.anchor_box), self.p.gate_growth, t_lost)
        ok = new & gate_mask(centres(boxes), areas(boxes), labels, self.pred, radius,
                             float(areas(self.anchor_box[None])[0]), self.cls)
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)
        dist = np.hypot(*(centres(boxes[idx]) - self.pred).T)
        j = pick(dist, self._similarities(self.frames[i], boxes[idx]), self.p.reid_thr)
        return None if j is None else int(idx[j])

    def step(self, i: int, ids: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        """(box, source, track_id) for frame index i > the initial index."""
        if self.tid is not None and (hit := np.flatnonzero(ids == self.tid)).size:
            j = int(hit[0])
            self.lock(i, self.tid, boxes[j], labels[j], ids)
            return boxes[j], "track", self.tid
        self._advance(i)
        f = self.frames[i]
        t_lost = (f - self.anchor_frame) / self.fps
        if self.p.reacquire and len(ids):
            j = self._reacquire(i, ids, boxes, labels, t_lost)
            if j is not None:  # a re-acquired track replaces MIL
                self.lock(i, int(ids[j]), boxes[j], labels[j], ids)
                return boxes[j], "track", self.tid
        if t_lost > self.p.coast_s + _EPS:
            self.mil = None
            return None, "lost", -1
        if self.mil is None and not self.mil_tried and self._mil_available():
            self.mil_tried = True
            self.mil = self._init_mil(self.anchor_frame, self.anchor_box)
        if self.mil is not None:
            ok, (x, y, w, h) = self.mil.update(self._frame(f))
            if ok:
                return np.array([x, y, x + w, y + h], float), "mil", -1
            self.mil = None
        return None, "lost", -1


def run_lock(
    tracks_seq: pd.DataFrame,
    feats: Feats,
    warps: np.ndarray | None,
    frames: list[int],
    init_frame: int,
    init_box: np.ndarray,
    params: LockParams,
    fps: float,
    read_frame: ReadFrame | None,
) -> pd.DataFrame:
    """One sequence, one target: per frame from `init_frame`, frame, x1..y2 (NaN when lost),
    source ("track" / "mil" / "lost") and track_id (-1 unless the source is "track").

    `feats(frame, boxes)` returns L2-normalised feature rows for those boxes (NaN rows or None
    when unmatched); it is called only by the "features" variant. `read_frame(frame)` returns a
    BGR image; it is called during MIL episodes and, for the "hsv" variant, on every locked frame
    and at re-acquisition."""
    frames = [int(f) for f in frames]
    if init_frame not in frames:
        raise ValueError(f"init_frame {init_frame} not in the sequence's frames")
    if warps is not None and len(warps) != len(frames):
        raise ValueError(f"{len(warps)} warps for {len(frames)} frames")
    if params.appearance == "hsv" and read_frame is None:
        raise ValueError("the hsv appearance needs read_frame")
    by_frame = {
        int(f): (g["track_id"].to_numpy(np.int64), g[BOX].to_numpy(float),
                 g["label"].to_numpy(np.int64))
        for f, g in tracks_seq.groupby("frame", sort=False)
    }
    empty = (np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0, np.int64))
    s = _Lock(params, fps, frames, warps, feats, read_frame)
    start = frames.index(init_frame)
    init_box = np.asarray(init_box, float).reshape(4)

    out_box, out_src, out_tid = [], [], []

    def emit(box, src, tid):
        out_box.append(np.full(4, np.nan) if box is None else np.asarray(box, float))
        out_src.append(src)
        out_tid.append(int(tid))

    ids, boxes, labels = by_frame.get(init_frame, empty)
    j = init_index(boxes, init_box)
    if j is not None:
        s.lock(start, int(ids[j]), boxes[j], labels[j], ids)
        emit(boxes[j], "track", ids[j])
    else:
        box = s.start_mil_only(start, init_box)
        emit(box, "mil" if box is not None else "lost", -1)
    for i in range(start + 1, len(frames)):
        emit(*s.step(i, *by_frame.get(frames[i], empty)))

    b = np.array(out_box).reshape(-1, 4)
    return pd.DataFrame({
        "frame": frames[start:], **{c: b[:, k] for k, c in enumerate(BOX)},
        "source": out_src, "track_id": np.array(out_tid, np.int64),
    })
