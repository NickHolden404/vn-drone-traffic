"""Per-frame disagreement between two teachers: the pseudo-label quality signal
and the active-learning acquisition score."""

from __future__ import annotations

import numpy as np

from cvt.detect.base import Detections
from cvt.detect.fusion import box_iou


def disagreement(a: Detections, b: Detections, iou_thr: float = 0.5) -> float:
    """D in [0, 1]: greedy class-agnostic matching at IoU >= iou_thr (highest IoU first);
    D = (unmatched + sum(1 - IoU) over matches + class flips among matches) / |union|,
    |union| = |a| + |b| - matches. Both empty -> 0; exactly one empty -> 1."""
    na, nb = len(a.boxes), len(b.boxes)
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return 1.0
    iou = box_iou(a.boxes, b.boxes)
    pairs = np.argwhere(iou >= iou_thr)
    order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_a, used_b, matches, cost = set(), set(), 0, 0.0
    for i, j in pairs[order]:
        if i in used_a or j in used_b:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        matches += 1
        cost += (1.0 - iou[i, j]) + float(a.labels[i] != b.labels[j])
    union = na + nb - matches
    return float(min(1.0, (union - matches + cost) / union))
