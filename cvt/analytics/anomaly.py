"""Scoring per-frame anomaly scores against the UIT-ADrone frame-level labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(labels_dir: Path, clip: str, alias: dict) -> np.ndarray:
    name = alias.get(clip, clip)
    path = Path(labels_dir) / f"{name}.npy"
    labels = np.load(path, allow_pickle=False)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path.name}: labels must be a 1-D array of 0/1")
    return labels.astype(np.int8)


def align(scores: np.ndarray, labels: np.ndarray, step: int, clip: str):
    sub = scores[::step]
    if abs(len(sub) - len(labels)) > 1:
        raise ValueError(f"{clip}: {len(sub)} scored frames vs {len(labels)} labels")
    n = min(len(sub), len(labels))
    return sub[:n], labels[:n]


def auc(scores, labels) -> float:
    """ROC AUC from average ranks; NaN when one class is absent."""
    scores = np.asarray(scores, float)
    pos = np.asarray(labels) == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pooled_auc(per_clip: list, n_boot: int, seed: int) -> tuple[float, float, float]:
    """AUC over the concatenated clips, with a 95% interval from resampling whole clips."""

    def pooled(idx):
        return auc(
            np.concatenate([per_clip[i][0] for i in idx]),
            np.concatenate([per_clip[i][1] for i in idx]),
        )

    n = len(per_clip)
    value = pooled(range(n)) if n else float("nan")
    if n < 2:
        return value, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [pooled(rng.integers(0, n, n)) for _ in range(n_boot)]
    boots = [b for b in boots if not np.isnan(b)]
    if not boots:
        return value, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return value, float(lo), float(hi)
