"""Unified 6-class schema shared by every dataset, model and metric."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASSES: tuple[str, ...] = ("motorbike", "car", "bus", "truck", "bicycle", "pedestrian")
CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
CANONICAL_WIDTH = 1920


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 4), np.float32)


def _empty_labels() -> np.ndarray:
    return np.zeros((0,), np.int64)


def _as_boxes(a: object, image_id: str, what: str) -> np.ndarray:
    b = np.asarray(a, np.float32)
    if b.size == 0:
        return np.zeros((0, 4), np.float32)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"{image_id}: {what} must have shape (N, 4), got {b.shape}")
    if not np.isfinite(b).all() or (b[:, 2:] < b[:, :2]).any():
        raise ValueError(f"{image_id}: {what} must be finite xyxy with x2 >= x1 and y2 >= y1")
    return b


@dataclass(eq=False)
class FrameAnn:
    """One labelled image. Boxes are xyxy pixels at native resolution."""

    image_id: str
    path: str
    width: int
    height: int
    source: str
    scene_id: str
    split: str
    boxes: np.ndarray = field(default_factory=_empty_boxes)
    labels: np.ndarray = field(default_factory=_empty_labels)
    ignore: np.ndarray = field(default_factory=_empty_boxes)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.image_id}: image size must be positive")
        self.boxes = _as_boxes(self.boxes, self.image_id, "boxes")
        self.ignore = _as_boxes(self.ignore, self.image_id, "ignore")
        raw = np.asarray(self.labels).reshape(-1)
        if raw.size and not np.all(np.mod(raw, 1) == 0):
            raise ValueError(f"{self.image_id}: labels must be integers")
        self.labels = raw.astype(np.int64)
        if len(self.boxes) != len(self.labels):
            raise ValueError(
                f"{self.image_id}: {len(self.boxes)} boxes vs {len(self.labels)} labels"
            )
        if len(self.labels) and (self.labels.min() < 0 or self.labels.max() >= len(CLASSES)):
            raise ValueError(
                f"{self.image_id}: label out of range {self.labels.min()}..{self.labels.max()}"
            )


def clip_boxes(
    boxes: np.ndarray, width: int, height: int, min_size: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Clip xyxy boxes to the image and drop boxes thinner than `min_size` px."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    keep = ((b[:, 2] - b[:, 0]) >= min_size) & ((b[:, 3] - b[:, 1]) >= min_size)
    return b[keep], keep
