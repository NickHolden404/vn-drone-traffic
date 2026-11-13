"""ByteTrack / BoT-SORT on cached detections (Ultralytics 8.4.173 trackers, CPU only).

Trackers are built as `ultralytics/trackers/track.py` does (`TRACKER_MAP[type](args=namespace)`) and
fed a small numpy `_Dets` instead of Ultralytics `Boxes`. BoT-SORT's global motion compensation is
injected: `gmc_warps` runs the real Ultralytics `GMC` once over the frames (the only function here
that reads images); `track_sequence` swaps the tracker's `gmc` for a stub returning warp `i` for
frame `i`, so tracking itself never touches pixels. ReID stays off.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

TRACK_COLS = [
    "seq_id", "frame", "timestamp_s", "camera_id", "track_id",
    "x1", "y1", "x2", "y2", "score", "label", "locked",
]  # fmt: skip
DET_COLS = ["frame", "x1", "y1", "x2", "y2", "score", "label"]

_BYTE_KEYS = {
    "tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
    "track_buffer", "match_thresh", "fuse_score",
}  # fmt: skip
_BOT_KEYS = _BYTE_KEYS | {"gmc_method", "proximity_thresh", "appearance_thresh", "with_reid", "model"}
_TRACKER_KEYS = {"bytetrack": _BYTE_KEYS, "botsort": _BOT_KEYS}
_EXTRA_KEYS = {"name", "frame_rate"}  # ours; not passed to the Ultralytics trackers


def _validate_cfg(cfg: dict, where: str = "tracker cfg") -> None:
    kind = cfg.get("tracker_type")
    if kind not in _TRACKER_KEYS:
        raise ValueError(f"{where}: tracker_type must be one of {sorted(_TRACKER_KEYS)}, got {kind!r}")
    allowed = _TRACKER_KEYS[kind] | _EXTRA_KEYS
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys for {kind}: {unknown}")
    missing = sorted(allowed - set(cfg))
    if missing:
        raise ValueError(f"{where}: missing keys for {kind}: {missing}")
    if cfg.get("with_reid"):
        raise ValueError(f"{where}: with_reid must be false (tracking runs on cached detections)")


def load_tracker_cfg(path: Path) -> dict:
    """Read a tracker yaml (Ultralytics keys + `name`, `frame_rate`); unknown/missing keys raise."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping of tracker keys")
    _validate_cfg(cfg, str(path))
    return cfg


def gmc_warps(
    frame_paths: Sequence[str], method: str = "sparseOptFlow", downscale: int = 2
) -> np.ndarray:
    """(N, 2, 3) camera warps, previous frame -> this frame, from Ultralytics GMC in order.

    Row 0 is the identity. Reads the frames with cv2; nothing else in this module does.
    """
    from ultralytics.trackers.utils.gmc import GMC

    gmc = GMC(method=method, downscale=downscale)
    warps = np.zeros((len(frame_paths), 2, 3))
    for i, p in enumerate(frame_paths):
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"cannot read frame image: {p}")
        warps[i] = gmc.apply(img)  # the first call initialises GMC and returns the identity
    return warps


class _Dets:
    """The slice of an Ultralytics numpy `Boxes` the trackers read: conf, cls, xyxy, xywh, len, mask."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.xywh = np.concatenate([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], 1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> _Dets:
        return _Dets(self.xyxy[mask], self.conf[mask], self.cls[mask])


class _WarpGMC:
    """Stands in for `BOTSORT.gmc`: `apply` returns the precomputed warp for the current frame."""

    def __init__(self, method: str, warps: np.ndarray):
        self.method = method  # non-None, or BoT-SORT skips motion compensation
        self.warps = warps
        self.index = 0

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        return self.warps[self.index]

    def reset_params(self) -> None:
        pass


def _build_tracker(cfg: dict, warps: np.ndarray | None):
    from ultralytics.trackers import BOTSORT, BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    args = IterableSimpleNamespace(**{k: v for k, v in cfg.items() if k not in _EXTRA_KEYS})
    if cfg["tracker_type"] == "bytetrack":
        return BYTETracker(args=args), None
    tracker = BOTSORT(args=args)
    if tracker.gmc.method is None:  # gmc_method: none -> identity, no warps needed
        return tracker, None
    if warps is None:
        raise ValueError("botsort with gmc_method != none needs precomputed warps (see gmc_warps)")
    stub = _WarpGMC(tracker.gmc.method, warps)
    tracker.gmc = stub
    return tracker, stub


def track_sequence(
    dets: pd.DataFrame,
    frames: Sequence[int],
    cfg: dict,
    *,
    seq_id: str,
    camera_id: str = "uav0",
    fps: float = 30.0,
    warps: np.ndarray | None = None,
    classes: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run one tracker over `frames` (in order, empty frames included) on cached `dets`.

    `dets`: frame, x1, y1, x2, y2, score, label. `warps[i]` is the camera warp for `frames[i]`
    (BoT-SORT only). Emits what Ultralytics emits: activated tracks updated in that frame. Every
    row of a track carries the track's majority label (ties -> highest summed score).
    """
    _validate_cfg(cfg)
    frames = [int(f) for f in frames]
    if any(b <= a for a, b in zip(frames, frames[1:], strict=False)):
        raise ValueError("frames must be strictly increasing")
    missing = [c for c in DET_COLS if c not in dets.columns]
    if missing:
        raise ValueError(f"dets is missing columns {missing}")
    if classes is not None:
        dets = dets[dets["label"].isin(list(classes))]
    stray = sorted(set(dets["frame"].astype(int)) - set(frames))
    if stray:
        raise ValueError(f"dets has frames not in frames: {stray[:5]}")
    if warps is not None:
        warps = np.asarray(warps, dtype=float)
        if warps.shape != (len(frames), 2, 3):
            raise ValueError(f"warps shape {warps.shape} != {(len(frames), 2, 3)}")

    tracker, stub = _build_tracker(cfg, warps)
    by_frame = {int(f): g for f, g in dets.groupby("frame")}
    dummy = np.zeros((1, 1, 3), np.uint8)  # BYTETracker only checks img is not None to apply GMC
    rows = []
    for i, f in enumerate(frames):
        g = by_frame.get(f)
        if g is None:
            d = _Dets(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
        else:
            d = _Dets(
                g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                g["score"].to_numpy(np.float32),
                g["label"].to_numpy(np.float32),
            )
        if stub is not None:
            stub.index = i
        out = tracker.update(d, dummy if stub is not None else None)
        for x1, y1, x2, y2, tid, score, cls, _idx in out:  # [x1,y1,x2,y2,id,score,cls,det_idx]
            rows.append((f, int(tid), x1, y1, x2, y2, score, int(cls)))

    res = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score", "label"])
    res = res.astype({c: "float64" for c in ("x1", "y1", "x2", "y2", "score")})
    res["frame"] = res["frame"].astype("int64")
    res["track_id"] = res["track_id"].astype("int64")
    res["label"] = res["label"].astype("int64")
    if len(res):
        tally = res.groupby(["track_id", "label"])["score"].agg(["size", "sum"]).reset_index()
        best = tally.sort_values(["track_id", "size", "sum"], ascending=[True, False, False])
        majority = best.drop_duplicates("track_id").set_index("track_id")["label"]
        res["label"] = res["track_id"].map(majority).astype("int64")
    first = frames[0] if frames else 0
    res["seq_id"] = seq_id
    res["camera_id"] = camera_id
    res["timestamp_s"] = (res["frame"] - first) / fps
    res["locked"] = False
    return res[TRACK_COLS].reset_index(drop=True)
