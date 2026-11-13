"""Box IoU, class-wise NMS and weighted boxes fusion (Solovyev et al. 2021) in numpy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cvt.detect.base import Detections


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) IoU of xyxy boxes."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = iw.clip(0) * ih.clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float) -> np.ndarray:
    """Class-wise greedy NMS; returns kept indices in descending score order."""
    boxes, scores, labels = np.asarray(boxes, float), np.asarray(scores), np.asarray(labels)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(boxes), bool)
    keep = []
    for pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        rest = order[pos + 1 :]
        rest = rest[~suppressed[rest] & (labels[rest] == labels[i])]
        if len(rest):
            suppressed[rest[box_iou(boxes[i : i + 1], boxes[rest])[0] > iou_thr]] = True
    return np.array(keep, dtype=np.int64)


def wbf(dets: Sequence[Detections], iou_thr: float = 0.55, skip_thr: float = 0.0) -> Detections:
    """Weighted boxes fusion with equal model weights and conf_type 'avg'.

    Per class, boxes (highest score first) join the fused box they overlap most if IoU > iou_thr,
    else start a new cluster. A fused box is the score-weighted mean of its members; its score is
    the mean member score times the number of distinct models in the cluster over #models
    (counting models, not boxes, so a box found by one of two teachers keeps half its score even
    if that teacher produced two overlapping boxes).
    """
    n_models = len(dets)
    if n_models == 0:
        return Detections.empty()
    boxes = np.concatenate([d.boxes for d in dets]).astype(float).reshape(-1, 4)
    model = np.concatenate([np.full(len(d.scores), m, np.int64) for m, d in enumerate(dets)])
    scores = np.concatenate([d.scores for d in dets]).astype(float)
    labels = np.concatenate([d.labels for d in dets]).astype(np.int64)
    k = scores >= skip_thr
    boxes, scores, labels, model = boxes[k], scores[k], labels[k], model[k]
    out_b, out_s, out_l = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        fused = np.zeros((0, 4))
        members: list[list[int]] = []
        for i in idx:
            if len(fused):
                ious = box_iou(boxes[i : i + 1], fused)[0]
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    members[j].append(i)
                    m = np.array(members[j])
                    w = scores[m]
                    fused[j] = (boxes[m] * w[:, None]).sum(0) / w.sum()
                    continue
            members.append([i])
            fused = np.vstack([fused, boxes[i]])
        for j, mem in enumerate(members):
            out_b.append(fused[j])
            out_s.append(scores[mem].mean() * min(n_models, len(set(model[mem].tolist()))) / n_models)
            out_l.append(c)
    return Detections(np.array(out_b, np.float32).reshape(-1, 4),
                      np.array(out_s, np.float32), np.array(out_l, np.int64))


def cross_class_suppress(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                         iou_thr: float) -> np.ndarray:
    """Keep mask: a box is dropped when a higher-scored kept box of a *different* class overlaps it
    with IoU >= iou_thr (one object predicted as two classes, e.g. a scooter as motorbike and
    bicycle). Same-class overlaps are left to NMS/WBF."""
    keep = np.zeros(len(boxes), bool)
    kept: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if kept:
            k = np.asarray(kept)
            if ((box_iou(boxes[i : i + 1], boxes[k])[0] >= iou_thr) & (labels[k] != labels[i])).any():
                continue
        keep[i] = True
        kept.append(int(i))
    return keep
