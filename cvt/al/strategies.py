"""Frame rankings of the pool (no labels used). Budget b takes the first b frames of a ranking,
so budgets are nested."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _unique(ids: Sequence[str]) -> list[str]:
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image ids in a ranking input")
    return ids


def random_ranking(ids: Sequence[str], seed: int) -> list[str]:
    """Seeded shuffle of the sorted ids (independent of the input order)."""
    ids = sorted(_unique(ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    return [ids[k] for k in order]


def disagreement_ranking(scores: pd.DataFrame) -> list[str]:
    """`image_id, disagreement` -> ids by disagreement descending, ties by image_id."""
    ids = _unique(scores.image_id)
    d = scores.disagreement.to_numpy(float)
    if not np.isfinite(d).all():
        raise ValueError("disagreement scores must be finite")
    return [i for _, i in sorted(zip(-d, ids, strict=True))]


def _far_enough(taken: list[int], frame: int, gap: int) -> bool:
    k = bisect_left(taken, frame)
    return all(abs(frame - taken[j]) >= gap for j in (k - 1, k) if 0 <= j < len(taken))


def variety_ranking(
    scores: pd.DataFrame,
    clip_of: Mapping[str, str],
    frame_of: Mapping[str, int],
    gaps: Sequence[int] = (15, 5, 0),
) -> list[str]:
    """Round-robin over the clips (sorted); each turn takes the clip's highest-disagreement frame
    at least `gap` frames from every frame already taken from that clip. When no clip has such a
    frame, the round-robin continues with the next gap; the last gap must be 0 so the ranking
    covers every frame."""
    gaps = [int(g) for g in gaps]
    if not gaps or gaps[-1] != 0 or any(a <= b for a, b in zip(gaps, gaps[1:], strict=False)):
        raise ValueError(f"gaps must be strictly decreasing and end with 0, got {gaps}")
    ranked = disagreement_ranking(scores)
    missing = [i for i in ranked if i not in clip_of or i not in frame_of]
    if missing:
        raise ValueError(f"{len(missing)} ids have no clip or frame number: {missing[:5]}")
    remaining: dict[str, list[str]] = {}
    for i in ranked:
        remaining.setdefault(clip_of[i], []).append(i)
    clips = sorted(remaining)
    taken: dict[str, list[int]] = {c: [] for c in clips}
    out: list[str] = []
    turn = 0
    for gap in gaps:
        # frames too close for this gap stay too close (taken frames only grow): skip them
        queue = {c: list(reversed(remaining[c])) for c in clips}
        misses = 0
        while misses < len(clips):
            c = clips[turn % len(clips)]
            turn += 1
            q = queue[c]
            while q and not _far_enough(taken[c], int(frame_of[q[-1]]), gap):
                q.pop()
            if not q:
                misses += 1
                continue
            misses = 0
            i = q.pop()
            out.append(i)
            insort(taken[c], int(frame_of[i]))
        done = set(out)
        remaining = {c: [i for i in remaining[c] if i not in done] for c in clips}
    if len(out) != len(ranked):
        raise AssertionError("variety ranking lost frames")
    return out


def budget_picks(ranking: Sequence[str], budget: int) -> list[str]:
    """The first `budget` frames of `ranking`."""
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 < budget <= len(ranking):
        raise ValueError(f"budget must be an int in [1, {len(ranking)}], got {budget!r}")
    return list(ranking[:budget])
