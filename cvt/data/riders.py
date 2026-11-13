"""Fold rider boxes into the vehicle they sit on (UIT-Drone7 convention)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def merge_riders(
    boxes: np.ndarray,
    labels: np.ndarray,
    rider_label: int,
    vehicle_labels: Sequence[int],
    min_inside: float = 0.5,
    scores: np.ndarray | None = None,
    min_vehicle_score: float = 0.0,
    feet_min_inside: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Remove riders sitting on a rideable vehicle box.

    A rider is merged when >= `min_inside` of its area lies inside the vehicle, or when
    >= `feet_min_inside` does and its feet (bottom-centre) are on the vehicle -- the upright
    riders that oblique drone views show above a small bike box (verified on VisDrone train:
    94.6% of the 25-50% band have their feet on the bike; mosaics in reports/).

    Each such rider is assigned to the vehicle covering most of it, and that vehicle box
    grows to the union. With `scores` (predictions), only vehicles scoring at least
    `min_vehicle_score` may absorb a rider, and the merged vehicle keeps the higher score.
    Returns (boxes, labels, keep, scores) where `keep` masks the inputs and `scores` is
    None when none were given.
    """
    if rider_label in vehicle_labels:
        raise ValueError("rider_label must not also be a vehicle label")
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    labels = np.asarray(labels).reshape(-1)
    if len(boxes) != len(labels):
        raise ValueError(f"{len(boxes)} boxes vs {len(labels)} labels")
    sc = None if scores is None else np.asarray(scores, np.float32).reshape(-1).copy()
    if sc is not None and len(sc) != len(boxes):
        raise ValueError(f"{len(sc)} scores vs {len(boxes)} boxes")

    keep = np.ones(len(boxes), bool)
    riders = np.flatnonzero(labels == rider_label)
    vehicles = np.flatnonzero(np.isin(labels, list(vehicle_labels)))
    if sc is not None:
        vehicles = vehicles[sc[vehicles] >= min_vehicle_score]
    if len(riders) == 0 or len(vehicles) == 0:
        return boxes, labels.copy(), keep, sc

    r, v = boxes[riders], boxes[vehicles]
    iw = np.minimum(r[:, None, 2], v[None, :, 2]) - np.maximum(r[:, None, 0], v[None, :, 0])
    ih = np.minimum(r[:, None, 3], v[None, :, 3]) - np.maximum(r[:, None, 1], v[None, :, 1])
    inter = iw.clip(0) * ih.clip(0)
    r_area = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    frac = inter / np.maximum(r_area[:, None], 1e-9)
    best = frac.argmax(axis=1)
    best_frac = frac[np.arange(len(riders)), best]
    vb = v[best]
    feet_x, feet_y = (r[:, 0] + r[:, 2]) / 2, r[:, 3]
    feet_on = (feet_x >= vb[:, 0]) & (feet_x <= vb[:, 2]) & (feet_y >= vb[:, 1]) & (feet_y <= vb[:, 3])
    merge = (best_frac >= min_inside) | ((best_frac >= feet_min_inside) & feet_on)

    for ri, vi, m in zip(riders, vehicles[best], merge, strict=True):
        if m:
            keep[ri] = False
            b, rb = boxes[vi], boxes[ri]
            boxes[vi] = [min(b[0], rb[0]), min(b[1], rb[1]), max(b[2], rb[2]), max(b[3], rb[3])]
            if sc is not None:
                sc[vi] = max(sc[vi], sc[ri])
    return boxes[keep], labels[keep], keep, (None if sc is None else sc[keep])
