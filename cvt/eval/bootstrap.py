"""Cluster (scene-level) bootstrap. Test sets are consecutive video frames, so whole clips
are resampled; per-image resampling would understate uncertainty."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray | None], dict[str, float]]


def scene_weights(codes: np.ndarray, n_scenes: int, rng: np.random.Generator) -> np.ndarray:
    """Per-image multiplicity after drawing `n_scenes` scenes with replacement."""
    draw = rng.integers(0, n_scenes, n_scenes)
    return np.bincount(draw, minlength=n_scenes)[codes].astype(float)


def _resamples(scene_of_image: np.ndarray, n_boot: int, seed: int):
    _, codes = np.unique(np.asarray(scene_of_image), return_inverse=True)
    n_scenes = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield scene_weights(codes, n_scenes, rng)


def _interval(values: list[float], alpha: float, min_valid: float) -> tuple[float, float]:
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    if v.size == 0 or ok.mean() < min_valid:
        return float("nan"), float("nan")
    return float(np.percentile(v[ok], 100 * alpha / 2)), float(np.percentile(v[ok], 100 * (1 - alpha / 2)))


def bootstrap_ci(
    metric_fn: MetricFn, scene_of_image: np.ndarray, n_boot: int = 1000, seed: int = 0,
    alpha: float = 0.05, min_valid: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile CI per metric. `scene_of_image` must be aligned with the per-image weights the
    metric function expects (same order as the images used to build the tables). A CI is NaN
    when fewer than `min_valid` of the resamples give a finite value."""
    samples: dict[str, list[float]] = defaultdict(list)
    for w in _resamples(scene_of_image, n_boot, seed):
        for k, v in metric_fn(w).items():
            samples[k].append(v)
    return {k: _interval(v, alpha, min_valid) for k, v in samples.items()}


def paired_bootstrap(
    fn_a: MetricFn, fn_b: MetricFn, scene_of_image: np.ndarray, n_boot: int = 1000,
    seed: int = 0, alpha: float = 0.05, min_valid: float = 0.95,
) -> dict[str, tuple[float, float, float]]:
    """(delta a-b, lo, hi) per shared metric, using the same resamples for both models.

    `scene_of_image` must be aligned with the per-image weights the metric functions expect.
    Metrics missing from a resample count as NaN; the CI is NaN when fewer than `min_valid`
    of the resamples are finite."""
    nan = float("nan")
    pa, pb = fn_a(None), fn_b(None)
    keys = sorted(set(pa) & set(pb))
    diffs: dict[str, list[float]] = defaultdict(list)
    for w in _resamples(scene_of_image, n_boot, seed):
        a, b = fn_a(w), fn_b(w)
        for k in keys:
            diffs[k].append(a.get(k, nan) - b.get(k, nan))
    return {k: (pa[k] - pb[k], *_interval(diffs[k], alpha, min_valid)) for k in keys}
