"""Pseudo-labels from the members' cached predictions: WBF, per-class thresholds, disagreement."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.schema import CLASSES
from cvt.detect.base import Detections
from cvt.detect.composite import _dets, _top
from cvt.detect.fusion import cross_class_suppress, wbf
from cvt.eval.predictions import PRED_COLS
from cvt.pseudo.disagreement import disagreement
from cvt.utils import write_parquet_atomic

# fixed prefix of the per-frame stats; one count column per member follows (n_boxes_m0, ...)
FRAME_COLS = ["image_id", "n_boxes", "disagreement"]


def member_count_cols(n_members: int) -> list[str]:
    return [f"n_boxes_m{i}" for i in range(n_members)]


def _thr_array(thresholds: Mapping[int, float]) -> np.ndarray:
    return np.array([thresholds[c] for c in range(len(CLASSES))], np.float32)


def _keep(d: Detections, thr: np.ndarray) -> Detections:
    k = d.scores >= thr[d.labels]
    return Detections(d.boxes[k], d.scores[k], d.labels[k])


def pseudo_label_frames(
    member_dts: Sequence[pd.DataFrame], images: pd.DataFrame, thresholds: Mapping[int, float],
    iou_thr: float = 0.55, skip_thr: float = 0.0, max_det: int = 1000,
    member_thresholds: Sequence[Mapping[int, float]] | None = None,
    cross_class_iou: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(labels in PRED_COLS, per-frame stats) for `images`, in order. The stats are FRAME_COLS
    followed by one box count per member (`member_count_cols`).

    Fused boxes are kept when score >= thresholds[class], then the top-`max_det` per frame
    (like the detectors). Disagreement is the mean pairwise D between members after per-class
    thresholds: member i's own `member_thresholds[i]` when given (members' scores are
    calibrated differently), else the fused `thresholds`; the member counts use the same.
    With `cross_class_iou`, kept fused boxes then go through `cross_class_suppress`."""
    thr = _thr_array(thresholds)
    if member_thresholds is None:
        member_thr = [thr] * len(member_dts)
    else:
        if len(member_thresholds) != len(member_dts):
            raise ValueError(f"member_thresholds must be one per member: got "
                             f"{len(member_thresholds)} for {len(member_dts)} members")
        member_thr = [_thr_array(t) for t in member_thresholds]
    groups = [{k: g for k, g in dt.groupby("image_id")} for dt in member_dts]
    label_rows, frame_rows = [], []
    for iid in images.image_id:
        raw = [_dets(g.get(iid)) for g in groups]
        members = [_keep(d, t) for d, t in zip(raw, member_thr, strict=True)]
        fused = _keep(wbf(raw, iou_thr, skip_thr), thr)
        if cross_class_iou is not None and len(fused.boxes):
            k = cross_class_suppress(fused.boxes, fused.scores, fused.labels, cross_class_iou)
            fused = Detections(fused.boxes[k], fused.scores[k], fused.labels[k])
        fused = _top(fused, max_det)
        ds = [disagreement(x, y) for x, y in combinations(members, 2)]
        frame_rows.append((iid, len(fused.boxes), float(np.mean(ds)) if ds else 0.0,
                           *(len(m.boxes) for m in members)))
        for b, s, c in zip(fused.boxes, fused.scores, fused.labels, strict=True):
            label_rows.append((iid, *map(float, b), float(s), int(c)))
    cols = FRAME_COLS + member_count_cols(len(member_dts))
    return pd.DataFrame(label_rows, columns=PRED_COLS), pd.DataFrame(frame_rows, columns=cols)


def save_pseudo(labels: pd.DataFrame, frames: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    """labels, frames, then meta.json last, each atomically: an existing meta.json is removed
    first, so a directory without one is an interrupted save."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta_path.unlink(missing_ok=True)
    write_parquet_atomic(labels, out_dir / "labels.parquet")
    write_parquet_atomic(frames, out_dir / "frames.parquet")
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, meta_path)


def load_pseudo(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    return pd.read_parquet(out_dir / "labels.parquet"), pd.read_parquet(out_dir / "frames.parquet")
