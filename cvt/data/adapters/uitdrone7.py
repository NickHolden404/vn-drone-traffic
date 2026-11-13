"""UIT-Drone7: YOLO-normalised `cls cx cy w h` labels, one .txt per image, train/test folders."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cvt.data.classmaps import UITDRONE7_RAW
from cvt.data.schema import FrameAnn, clip_boxes
from cvt.utils import image_size

SOURCE = "uitdrone7"
_STEM = re.compile(r"^(?P<scene>.+)_(?P<frame>\d+)$")


def scene_of(stem: str) -> str:
    """Clip id = file stem without its trailing `_<frame number>`."""
    m = _STEM.match(stem)
    if not m:
        raise ValueError(f"cannot parse scene from UIT-Drone7 stem {stem!r}")
    return m["scene"]


def _parse(path: Path) -> np.ndarray:
    """(N, 5) rows `cls cx cy w h`; exact duplicate rows are dropped (order kept)."""
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
        vals = [float(v) for v in parts]
        if not vals[0].is_integer():
            raise ValueError(f"{path}:{lineno}: non-integer class id {parts[0]!r}")
        rows.append(vals)
    arr = np.array(rows, np.float64).reshape(-1, 5)
    if len(arr) == 0:
        return arr
    _, first = np.unique(arr, axis=0, return_index=True)
    return arr[np.sort(first)]


def load_uitdrone7(images_dir: Path, labels_dir: Path, split: str) -> list[FrameAnn]:
    """Load one official split. Images without a label file are skipped (unlabelled)."""
    frames: list[FrameAnn] = []
    for lab in sorted((Path(labels_dir) / split).glob("*.txt")):
        img = Path(images_dir) / split / f"{lab.stem}.jpg"
        if not img.exists():
            raise FileNotFoundError(f"label without image: {img}")
        w, h = image_size(img)
        vals = _parse(lab)
        raw = vals[:, 0].astype(int)
        unknown = sorted(set(raw.tolist()) - set(UITDRONE7_RAW))
        if unknown:
            raise ValueError(f"{lab}: unknown UIT-Drone7 class ids {unknown}")
        cx, cy, bw, bh = vals[:, 1] * w, vals[:, 2] * h, vals[:, 3] * w, vals[:, 4] * h
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes, keep = clip_boxes(xyxy, w, h)
        labels = np.array([UITDRONE7_RAW[c] for c in raw[keep]], np.int64)
        frames.append(
            FrameAnn(
                image_id=f"{SOURCE}/{split}/{lab.stem}",
                path=str(img),
                width=w,
                height=h,
                source=SOURCE,
                scene_id=f"{SOURCE}/{scene_of(lab.stem)}",
                split=split,
                boxes=boxes,
                labels=labels,
            )
        )
    return frames
