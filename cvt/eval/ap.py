"""COCO-exact AP with per-image weights, so cluster bootstrap needs no re-matching.

pycocotools' `evaluate()` does the greedy matching once; we keep its per-detection TP/FP
flags and re-run its `accumulate()` arithmetic with weights (bootstrap multiplicities).
With all weights = 1 this reproduces pycocotools to floating-point precision (see tests).
Prediction scores should be kept at full float precision: pycocotools breaks score ties by image
order, so rounded scores make AP depend on that order.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cvt.data.schema import CLASSES

AREA_NAMES: tuple[str, ...] = ("all", "tiny", "small", "medium+")
AREA_RNG: dict[str, list[float]] = {
    "all": [0, 1e10],
    "tiny": [0, 16**2],
    "small": [16**2, 32**2],
    "medium+": [32**2, 1e10],
}
IOU_THRS = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
REC_THRS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
MAX_DET = 1000


@dataclass
class MatchTable:
    """Detections of one (class, area) cell, sorted by descending score."""

    img_idx: np.ndarray  # (D,) image position of each detection
    tp: np.ndarray  # (T, D) bool
    fp: np.ndarray  # (T, D) bool
    npig: np.ndarray  # (I,) non-ignored GT count per image
    scores: np.ndarray | None = None  # (D,) detection scores, descending


def configure(E: COCOeval, img_ids: Sequence[int], class_ids: Sequence[int], max_det: int = MAX_DET) -> None:
    p = E.params
    p.imgIds = list(img_ids)
    p.catIds = [c + 1 for c in sorted(class_ids)]
    p.areaRng = [AREA_RNG[a] for a in AREA_NAMES]
    p.areaRngLbl = list(AREA_NAMES)
    p.maxDets = [max_det]
    p.iouThrs = IOU_THRS.copy()
    p.recThrs = REC_THRS.copy()


def build_tables(
    coco_gt: COCO, coco_dt: COCO, img_ids: Sequence[int], class_ids: Sequence[int],
    max_det: int = MAX_DET,
) -> dict[tuple[int, str], MatchTable]:
    class_ids = sorted(class_ids)
    if list(img_ids) != sorted(img_ids):
        raise ValueError("img_ids must be ascending (pycocotools sorts them internally)")
    E = COCOeval(coco_gt, coco_dt, "bbox")
    configure(E, img_ids, class_ids, max_det)
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
    n_img, n_area, n_iou = len(E.params.imgIds), len(AREA_NAMES), len(IOU_THRS)

    tables: dict[tuple[int, str], MatchTable] = {}
    for k, c in enumerate(class_ids):
        for a, area in enumerate(AREA_NAMES):
            npig = np.zeros(n_img)
            idx, scores, dtm, dtig = [], [], [], []
            for i in range(n_img):
                e = E.evalImgs[k * n_area * n_img + a * n_img + i]
                if e is None:
                    continue
                npig[i] = np.count_nonzero(np.logical_not(e["gtIgnore"]))
                s = np.asarray(e["dtScores"][:max_det], float)
                if len(s) == 0:
                    continue
                idx.append(np.full(len(s), i))
                scores.append(s)
                dtm.append(e["dtMatches"][:, : len(s)])
                dtig.append(e["dtIgnore"][:, : len(s)])
            if scores:
                s_all = np.concatenate(scores)
                order = np.argsort(-s_all, kind="mergesort")
                m = np.concatenate(dtm, axis=1)[:, order]
                ig = np.concatenate(dtig, axis=1)[:, order].astype(bool)
                tables[(c, area)] = MatchTable(
                    np.concatenate(idx)[order], (m != 0) & ~ig, (m == 0) & ~ig, npig, s_all[order]
                )
            else:
                empty = np.zeros((n_iou, 0), bool)
                tables[(c, area)] = MatchTable(np.zeros(0, int), empty, empty, npig, np.zeros(0))
    return tables


def ap_from_table(t: MatchTable, weights: np.ndarray | None = None) -> np.ndarray:
    """(T,) AP per IoU threshold; NaN when the (weighted) cell has no GT."""
    w = np.ones(len(t.npig)) if weights is None else np.asarray(weights, float)
    npig = float((t.npig * w).sum())
    n_iou = t.tp.shape[0]
    if npig == 0:
        return np.full(n_iou, np.nan)
    n_det = t.tp.shape[1]
    if n_det == 0:
        return np.zeros(n_iou)
    dw = w[t.img_idx]
    tp_sum = np.cumsum(t.tp * dw, axis=1)
    fp_sum = np.cumsum(t.fp * dw, axis=1)
    rc = tp_sum / npig
    pr = tp_sum / (tp_sum + fp_sum + np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
    out = np.zeros(n_iou)
    for ti in range(n_iou):
        inds = np.searchsorted(rc[ti], REC_THRS, side="left")
        valid = inds < n_det
        q = np.zeros(len(REC_THRS))
        q[valid] = pr[ti, inds[valid]]
        out[ti] = q.mean()
    return out


def eligible_classes(
    tables: dict[tuple[int, str], MatchTable],
    class_ids: Sequence[int],
    scene_of_image: np.ndarray,
    min_scenes: int = 4,
) -> list[int]:
    """Classes averaged into mAP: GT present in at least min(min_scenes, #scenes) scenes
    (pre-declared rule: >= min(4, #scenes)).

    Fixing this set from the full sample keeps the averaged class set identical across
    bootstrap resamples; rarer classes still get a per-class AP point estimate.
    `scene_of_image` must be aligned with the image order used to build `tables`.
    """
    scenes = np.asarray(scene_of_image)
    need = max(min(min_scenes, len(np.unique(scenes))), 1)
    out = []
    for c in sorted(class_ids):
        has_gt = tables[(c, "all")].npig > 0
        if len(np.unique(scenes[has_gt])) >= need:
            out.append(c)
    return out


def summarize(
    tables: dict[tuple[int, str], MatchTable],
    class_ids: Sequence[int],
    weights: np.ndarray | None = None,
    areas: Sequence[str] = AREA_NAMES,
    map_classes: Sequence[int] | None = None,
    strict: bool = False,
) -> dict[str, float]:
    """Per-class AP (area 'all') for every class in `class_ids` (NaN when it has no GT here),
    and mAP50 / mAP50_95 per area (suffix @area) over `map_classes` (default `class_ids`).

    strict=False skips map classes without GT (point estimates). strict=True makes mAP NaN when
    any map class lacks GT, so bootstrap resamples always average the same class set.
    """
    map_classes = sorted(class_ids if map_classes is None else map_classes)
    every = sorted(set(class_ids) | set(map_classes))
    out: dict[str, float] = {}
    for area in areas:
        ap = {c: ap_from_table(tables[(c, area)], weights) for c in every}
        suffix = "" if area == "all" else f"@{area}"
        for key, per_class in (
            ("mAP50", [ap[c][0] for c in map_classes]),
            ("mAP50_95", [ap[c].mean() for c in map_classes]),
        ):
            vals = np.array(per_class, float)
            if vals.size == 0 or np.isnan(vals).all() or (strict and np.isnan(vals).any()):
                out[key + suffix] = float("nan")
            else:
                out[key + suffix] = float(np.nanmean(vals))
        if area == "all":
            for c in sorted(class_ids):
                out[f"AP50/{CLASSES[c]}"] = float(ap[c][0])
                out[f"AP50_95/{CLASSES[c]}"] = float(ap[c].mean())
    return out


def per_class_thresholds(
    tables: dict[tuple[int, str], MatchTable],
    class_ids: Sequence[int],
    iou_index: int = 0,
    default: float = 0.5,
) -> dict[int, float]:
    """Score threshold per class maximising F1 at IOU_THRS[iou_index] (area 'all').

    Ignored detections count as neither TP nor FP, as in COCO matching. Classes without
    ground truth or detections fall back to `default`; classes with ground truth but no true
    positive at any cut get `math.inf` (emit nothing for that class).
    """
    out: dict[int, float] = {}
    for c in sorted(class_ids):
        t = tables[(c, "all")]
        npig = float(t.npig.sum())
        if npig == 0 or t.tp.shape[1] == 0 or t.scores is None:
            out[c] = default
            continue
        tp = np.cumsum(t.tp[iou_index])
        fp = np.cumsum(t.fp[iou_index])
        precision = tp / np.maximum(tp + fp, 1e-9)
        recall = tp / npig
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        out[c] = float(t.scores[int(np.argmax(f1))]) if f1.max() > 0 else math.inf
    return out
