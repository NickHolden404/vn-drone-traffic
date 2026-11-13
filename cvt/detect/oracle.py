"""Ground truth as detections. Harness sanity check: must score mAP = 1.0."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

from cvt.detect.base import Detections


class OracleDetector:
    name = "oracle"
    cache_tag = "oracle"

    def __init__(self, images: pd.DataFrame, boxes: pd.DataFrame):
        self._id_of = dict(zip(images.path, images.image_id, strict=True))
        obj = boxes[~boxes.is_ignore.astype(bool)]
        self._by_id = {k: g for k, g in obj.groupby("image_id")}

    def predict_paths(self, paths: Sequence[str]) -> Iterator[Detections]:
        for p in paths:
            g = self._by_id.get(self._id_of[p])
            if g is None:
                yield Detections.empty()
                continue
            yield Detections(
                g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                np.ones(len(g), np.float32),
                g.label.to_numpy(np.int64),
            )
