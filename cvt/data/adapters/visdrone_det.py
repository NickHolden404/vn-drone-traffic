"""VisDrone2019-DET: `x,y,w,h,score,category,truncation,occlusion` per line, one .txt per image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cvt.data.classmaps import (
    IGNORE,
    VISDRONE_PEOPLE,
    VISDRONE_RAW,
    VISDRONE_RIDEABLE,
    VISDRONE_TRICYCLES,
)
from cvt.data.riders import merge_riders
from cvt.data.schema import FrameAnn, clip_boxes
from cvt.utils import image_size

SOURCE = "visdrone"


def _parse(path: Path) -> np.ndarray:
    """(N, 6) rows `x y w h score category`; exact duplicate rows are dropped (order kept)."""
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"{path}:{lineno}: expected >= 6 comma-separated fields, got {len(parts)}")
        rows.append([int(v) for v in parts[:6]])
    arr = np.array(rows, dtype=np.int64).reshape(-1, 6)
    if len(arr) == 0:
        return arr
    _, first = np.unique(arr, axis=0, return_index=True)
    return arr[np.sort(first)]


def _frame_from_rows(
    xywh: np.ndarray,
    cats: np.ndarray,
    score: np.ndarray,
    w: int,
    h: int,
    rider_merge: bool,
    raw_map: dict[int, int] = VISDRONE_RAW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame ignore + rider-merge logic shared by the VisDrone DET/MOT adapters.

    Input rows are `x y w h`, raw category and score. Returns (boxes, labels, ignore, src):
    clipped object boxes, our class ids, clipped ignore regions, and for every returned box the
    index of the input row it came from (a merged rider's row is dropped; the vehicle keeps its own).
    """
    xyxy = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1)
    xyxy = xyxy.astype(np.float32)
    src = np.arange(len(xyxy))
    if rider_merge:
        xyxy, cats, kept, _ = merge_riders(xyxy, cats, VISDRONE_PEOPLE, VISDRONE_TRICYCLES)
        score, src = score[kept], src[kept]
    is_ignore = (score == 0) | np.array([raw_map[c] == IGNORE for c in cats], bool)

    ignore, _ = clip_boxes(xyxy[is_ignore], w, h)
    boxes, keep = clip_boxes(xyxy[~is_ignore], w, h)
    raw, src = cats[~is_ignore][keep], src[~is_ignore][keep]
    if rider_merge:
        boxes, raw, kept, _ = merge_riders(boxes, raw, VISDRONE_PEOPLE, VISDRONE_RIDEABLE)
        src = src[kept]
    labels = np.array([raw_map[c] for c in raw], np.int64)
    return boxes, labels, ignore, src


def load_visdrone_det(root: Path, split: str, rider_merge: bool) -> list[FrameAnn]:
    """Load `root/images` + `root/annotations`. Scene = source sequence (leading stem number),
    shared across splits so train/val overlap stays detectable."""
    root = Path(root)
    frames: list[FrameAnn] = []
    for ann in sorted((root / "annotations").glob("*.txt")):
        img = root / "images" / f"{ann.stem}.jpg"
        if not img.exists():
            raise FileNotFoundError(f"annotation without image: {img}")
        w, h = image_size(img)
        a = _parse(ann)
        unknown = sorted(set(a[:, 5].tolist()) - set(VISDRONE_RAW))
        if unknown:
            raise ValueError(f"{ann}: unknown VisDrone categories {unknown}")
        cats, score = a[:, 5], a[:, 4]
        boxes, labels, ignore, _ = _frame_from_rows(a[:, :4], cats, score, w, h, rider_merge)
        frames.append(
            FrameAnn(
                image_id=f"{SOURCE}/{split}/{ann.stem}",
                path=str(img),
                width=w,
                height=h,
                source=SOURCE,
                scene_id=f"{SOURCE}/{ann.stem.split('_')[0]}",
                split=split,
                boxes=boxes,
                labels=labels,
                ignore=ignore,
            )
        )
    return frames
