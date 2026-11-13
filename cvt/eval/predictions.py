"""Run a detector over a set of images, caching predictions as parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cvt.detect.base import Detector
from cvt.utils import write_parquet_atomic

PRED_COLS = ["image_id", "x1", "y1", "x2", "y2", "score", "label"]


def predict_frames(detector: Detector, images: pd.DataFrame, cache_file: Path | None) -> pd.DataFrame:
    """Run `detector` over `images` (in order) or load the cached parquet. The cache is written
    atomically and only after every image produced a result."""
    if cache_file is not None and Path(cache_file).exists():
        return pd.read_parquet(cache_file)
    ids, parts = [], []
    dets = detector.predict_paths(images.path.tolist())
    for iid, det in zip(images.image_id, dets, strict=True):
        if len(det.boxes):
            ids.append(np.full(len(det.boxes), iid, dtype=object))
            parts.append(np.column_stack([det.boxes, det.scores, det.labels]).astype(np.float64))
    if parts:
        arr = np.concatenate(parts)
        df = pd.DataFrame({
            "image_id": np.concatenate(ids), "x1": arr[:, 0], "y1": arr[:, 1], "x2": arr[:, 2],
            "y2": arr[:, 3], "score": arr[:, 4], "label": arr[:, 5].astype(np.int64),
        })
    else:
        df = pd.DataFrame(columns=PRED_COLS)
    if cache_file is not None:
        write_parquet_atomic(df, cache_file)
    return df
