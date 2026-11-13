"""Detector interface. Every backend (.pt, .onnx, .engine) yields Detections."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy pixels, native resolution
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) our class ids

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


class Detector(Protocol):
    name: str
    cache_tag: str  # changes whenever anything that shapes the predictions changes

    def predict_paths(self, paths: Sequence[str]) -> Iterator[Detections]:
        """Yield one Detections per path, in input order."""
        ...
