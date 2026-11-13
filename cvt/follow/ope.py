"""Lock-test scoring: one-pass-evaluation (OPE) metrics for a single locked target.

Pure numpy/pandas. Only frames where the target's GT box exists are scored. A NaN prediction
counts as IoU 0 and infinite centre error. A wrong-target frame is a scored frame where the
prediction has IoU >= 0.5 with a GT box of another track; a switch is the start of a run of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvt.data.schema import CLASS_ID
from cvt.eval.bootstrap import bootstrap_ci

VEHICLES: tuple[str, ...] = ("car", "truck", "bus", "motorbike")
_BOX = ["x1", "y1", "x2", "y2"]
_IOU_T = np.linspace(0.0, 1.0, 21)
_NORM_T = np.linspace(0.0, 0.5, 51)
_ID_COLS = {"seq_id", "track_id", "n_frames", "label", "first", "last"}


def lock_targets(
    gt: pd.DataFrame, fps: float, min_s: float = 5.0, classes=VEHICLES
) -> pd.DataFrame:
    """One row per GT vehicle track spanning at least `min_s` seconds (last - first + 1 frames).

    A track is keyed by (seq_id, track_id) over its vehicle rows only: a track whose label changes
    (car -> truck) is one target with its most frequent vehicle label (ties -> smallest label id)
    and the span of all its vehicle rows."""
    cols = ["seq_id", "track_id", "label", "first", "last"]
    key = ["seq_id", "track_id"]
    ids = {CLASS_ID[c] for c in classes}
    g = gt[gt["label"].isin(ids)]
    if g.empty:
        return pd.DataFrame(columns=cols)
    span = g.groupby(key, as_index=False).agg(first=("frame", "min"), last=("frame", "max"))
    votes = g.groupby([*key, "label"], as_index=False).size()
    label = (votes.sort_values([*key, "size", "label"], ascending=[True, True, False, True])
             .drop_duplicates(key)[[*key, "label"]])
    out = span.merge(label, on=key, how="left", validate="one_to_one")
    keep = (out["last"] - out["first"] + 1) >= min_s * fps
    return out[keep].sort_values(["seq_id", "track_id"]).reset_index(drop=True)[cols]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of (n,4) arrays; NaN boxes give 0."""
    iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = inter / union
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def ope_scores(pred: pd.DataFrame, gt_track: pd.DataFrame, others: pd.DataFrame) -> dict[str, float]:
    """Score one locked target's predicted boxes against its GT track.

    `pred`: frame, x1..y2 (NaN = lost). `gt_track`: the target's GT rows. `others`: the other GT
    tracks in the same frames. Returns success_auc, precision_20, norm_precision_auc, lock_kept,
    wrong_frames, wrong_switches and n_frames over frames where the target's GT exists.
    """
    gt = gt_track[["frame", *_BOX]].sort_values("frame").reset_index(drop=True)
    n = len(gt)
    p = pred[["frame", *_BOX]].drop_duplicates("frame").set_index("frame")
    pb = p.reindex(gt["frame"].to_numpy()).to_numpy(float)
    gb = gt[_BOX].to_numpy(float)
    iou = _iou(pb, gb)
    pc = np.stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2], 1)
    gc = np.stack([(gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2], 1)
    d = pc - gc
    err = np.where(np.isnan(d).any(1), np.inf, np.hypot(d[:, 0], d[:, 1]))
    w = np.maximum(gb[:, 2] - gb[:, 0], 1e-9)
    h = np.maximum(gb[:, 3] - gb[:, 1], 1e-9)
    nerr = np.where(np.isnan(d).any(1), np.inf, np.hypot(d[:, 0] / w, d[:, 1] / h))

    wrong = np.zeros(n, bool)
    if len(others) and n:
        o = others[["frame", *_BOX]].rename(columns={c: f"o_{c}" for c in _BOX})
        m = pd.DataFrame({"frame": gt["frame"].to_numpy(), **{c: pb[:, i] for i, c in enumerate(_BOX)}})
        m["idx"] = np.arange(n)
        m = m.merge(o, on="frame")
        if len(m):
            oi = _iou(m[_BOX].to_numpy(float), m[[f"o_{c}" for c in _BOX]].to_numpy(float))
            wrong[np.unique(m["idx"].to_numpy()[oi >= 0.5])] = True
    starts = wrong & ~np.concatenate([[False], wrong[:-1]]) if n else wrong

    def frac(x):
        return float(np.mean(x)) if n else float("nan")

    return {
        "success_auc": float(np.mean([frac(iou > t) for t in _IOU_T])) if n else float("nan"),
        "precision_20": frac(err <= 20.0),
        "norm_precision_auc": float(np.mean([frac(nerr <= t) for t in _NORM_T])) if n else float("nan"),
        "lock_kept": frac(iou > 0.5),
        "wrong_frames": float(wrong.sum()),
        "wrong_switches": float(starts.sum()),
        "n_frames": float(n),
    }


def aggregate(
    per_target: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """(mean, lo, hi) per metric, 95% cluster-bootstrap CI over sequences.

    Rate metrics are frame-weighted means; `wrong_frame_rate` is sum(wrong_frames)/sum(n_frames)
    and `wrong_switches_per_target` the plain mean of wrong_switches over targets. A NaN in any
    per-target metric raises ValueError naming the target (it would otherwise poison the means)."""
    df = per_target.reset_index(drop=True)
    counts = {"wrong_frames", "wrong_switches"}
    cols = [
        c for c in df.columns
        if c not in _ID_COLS | counts and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in [*cols, "n_frames", *sorted(counts)]:
        bad = np.flatnonzero(pd.to_numeric(df[c], errors="coerce").isna().to_numpy())
        if len(bad):
            r = df.iloc[bad[0]]
            raise ValueError(f"per-target metric {c} is NaN for target seq {r['seq_id']} track "
                             f"{r.get('track_id', '?')} ({len(bad)} target(s) affected)")
    vals = df[cols].to_numpy(float)
    nf = df["n_frames"].to_numpy(float)
    wf = df["wrong_frames"].to_numpy(float)
    sw = df["wrong_switches"].to_numpy(float)
    seq = df["seq_id"].to_numpy()
    nan = float("nan")

    def metric(w):
        m = np.ones(len(df)) if w is None else w
        ww = nf * m
        tot = ww.sum()
        out = {c: (float((vals[:, i] * ww).sum() / tot) if tot > 0 else nan) for i, c in enumerate(cols)}
        out["wrong_frame_rate"] = float((wf * m).sum() / tot) if tot > 0 else nan
        out["wrong_switches_per_target"] = float((sw * m).sum() / m.sum()) if m.sum() > 0 else nan
        return out

    point = metric(None)
    ci = bootstrap_ci(metric, seq, n_boot=n_boot, seed=seed)
    return {c: (point[c], *ci[c]) for c in point}
