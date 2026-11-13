"""MOT metrics (HOTA, CLEAR, Identity) as additive per-sequence statistics.

Per-sequence computation follows TrackEval 1.3.0 (`hota.py`, `clear.py`, `identity.py`) exactly;
`SeqStats` holds only plainly summable fields, so TrackEval's `combine_sequences` is a field-wise
sum (`ass_sum` etc. already carry the TP weighting) and a cluster bootstrap over sequences is
`sum(s.scaled(k))`. Use one class per call; class-combined stats are also a sum.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MOT_COLS = ["frame", "id", "x1", "y1", "x2", "y2"]
ALPHAS: np.ndarray = np.arange(0.05, 0.951, 0.05)  # == TrackEval's np.arange(0.05, 0.99, 0.05)
THRESHOLD = 0.5  # CLEAR and Identity IoU threshold
EPS = np.finfo("float").eps

_ARRAY_FIELDS = ("hota_tp", "hota_fn", "hota_fp", "ass_sum", "assre_sum", "asspr_sum", "loca_sum")


@dataclass
class SeqStats:
    """Summable statistics of one or more sequences. Arrays are per alpha (len(ALPHAS));
    counts are ints for a single sequence and may become floats after `scaled`."""

    hota_tp: np.ndarray
    hota_fn: np.ndarray
    hota_fp: np.ndarray
    ass_sum: np.ndarray  # sum over matched pairs of count * AssIoU (= AssA * TP)
    assre_sum: np.ndarray
    asspr_sum: np.ndarray
    loca_sum: np.ndarray  # sum of IoU over matches (= LocA * TP)
    clr_tp: float
    clr_fn: float
    clr_fp: float
    idsw: float
    idtp: float
    idfn: float
    idfp: float
    n_gt: float

    @classmethod
    def zero(cls) -> SeqStats:
        z = np.zeros(len(ALPHAS))
        return cls(*(z.copy() for _ in _ARRAY_FIELDS), 0, 0, 0, 0, 0, 0, 0, 0)

    def __add__(self, other: SeqStats) -> SeqStats:
        return SeqStats(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    def scaled(self, k: float) -> SeqStats:
        return SeqStats(*(getattr(self, f.name) * k for f in fields(self)))


def _check(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in MOT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if df.duplicated(["frame", "id"]).any():
        dup = df.loc[df.duplicated(["frame", "id"], keep=False), ["frame", "id"]].iloc[0]
        raise ValueError(
            f"{name}: id {dup['id']} appears more than once in frame {dup['frame']}"
        )


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TrackEval `_calculate_box_ious` for x0y0x1y1 boxes."""
    lo = np.minimum(a[:, None, :], b[None, :, :])
    hi = np.maximum(a[:, None, :], b[None, :, :])
    inter = np.maximum(lo[..., 2] - hi[..., 0], 0) * np.maximum(lo[..., 3] - hi[..., 1], 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    inter[area_a <= 0 + EPS, :] = 0
    inter[:, area_b <= 0 + EPS] = 0
    inter[union <= 0 + EPS] = 0
    union[union <= 0 + EPS] = 1
    return inter / union


def _split(
    df: pd.DataFrame, frames: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Contiguous ids (np.unique order, as TrackEval) and per-frame (ids, boxes) in input order."""
    uniq, inv = np.unique(df["id"].to_numpy(), return_inverse=True)
    order = np.argsort(df["frame"].to_numpy(), kind="stable")
    f = df["frame"].to_numpy()[order]
    ids, boxes = inv[order], df[MOT_COLS[2:]].to_numpy(float)[order]
    starts = np.searchsorted(f, frames, "left")
    ends = np.searchsorted(f, frames, "right")
    return (uniq, [ids[s:e] for s, e in zip(starts, ends, strict=True)],
            [boxes[s:e] for s, e in zip(starts, ends, strict=True)])


def sequence_stats(gt: pd.DataFrame, tr: pd.DataFrame) -> SeqStats:
    """Stats of one sequence and one class; `gt`/`tr` have MOT_COLS, ids unique per frame."""
    _check(gt, "gt")
    _check(tr, "tracker")
    frames = np.union1d(gt["frame"].to_numpy(), tr["frame"].to_numpy())
    g_uniq, g_ids, g_box = _split(gt, frames)
    t_uniq, t_ids, t_box = _split(tr, frames)
    sims = [_box_iou(a, b) for a, b in zip(g_box, t_box, strict=True)]
    n_gt, n_tr = len(gt), len(tr)
    hota = _hota(g_ids, t_ids, sims, len(g_uniq), len(t_uniq), n_gt, n_tr)
    clr_tp, clr_fn, clr_fp, idsw = _clear(g_ids, t_ids, sims, len(g_uniq))
    idtp, idfn, idfp = _identity(g_ids, t_ids, sims, len(g_uniq), len(t_uniq), n_gt, n_tr)
    return SeqStats(*hota, clr_tp, clr_fn, clr_fp, idsw, idtp, idfn, idfp, n_gt)


def _hota(g_ids, t_ids, sims, n_gid, n_tid, n_gt, n_tr) -> tuple[np.ndarray, ...]:
    """TrackEval HOTA.eval_sequence, returning summable fields."""
    na = len(ALPHAS)
    zeros = np.zeros(na)
    if n_tr == 0 or n_gt == 0:
        return (zeros.copy(), np.full(na, float(n_gt)), np.full(na, float(n_tr)),
                zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())
    potential = np.zeros((n_gid, n_tid))
    g_count = np.zeros((n_gid, 1))
    t_count = np.zeros((1, n_tid))
    for gi, ti, sim in zip(g_ids, t_ids, sims, strict=True):
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 0 + EPS
        sim_iou[mask] = sim[mask] / denom[mask]
        potential[gi[:, None], ti[None, :]] += sim_iou
        g_count[gi] += 1
        t_count[0, ti] += 1
    align = potential / (g_count + t_count - potential)

    mg, mt, ms = [], [], []  # matched gt id, tracker id, IoU (Hungarian matches, before alpha)
    for gi, ti, sim in zip(g_ids, t_ids, sims, strict=True):
        if len(gi) == 0 or len(ti) == 0:
            continue
        rows, cols = linear_sum_assignment(-(align[gi[:, None], ti[None, :]] * sim))
        mg.append(gi[rows])
        mt.append(ti[cols])
        ms.append(sim[rows, cols])
    mg, mt, ms = (np.concatenate(x) if x else np.zeros(0) for x in (mg, mt, ms))

    tp, ass, assre, asspr, loca = (np.zeros(na) for _ in range(5))
    gc, tc = g_count[:, 0], t_count[0]
    for a, alpha in enumerate(ALPHAS):
        ok = ms >= alpha - EPS
        tp[a] = ok.sum()
        loca[a] = ms[ok].sum()
        pair, cnt = np.unique(mg[ok].astype(np.int64) * n_tid + mt[ok].astype(np.int64),
                              return_counts=True)
        g, t, cnt = pair // n_tid, pair % n_tid, cnt.astype(float)
        ass[a] = np.sum(cnt * cnt / np.maximum(1, gc[g] + tc[t] - cnt))
        assre[a] = np.sum(cnt * cnt / np.maximum(1, gc[g]))
        asspr[a] = np.sum(cnt * cnt / np.maximum(1, tc[t]))
    return tp, n_gt - tp, n_tr - tp, ass, assre, asspr, loca


def _clear(g_ids, t_ids, sims, n_gid) -> tuple[int, int, int, int]:
    """TrackEval CLEAR.eval_sequence (TP, FN, FP, IDSW)."""
    tp = fn = fp = idsw = 0
    prev_tr = np.full(n_gid, np.nan)  # last tracker id ever matched to each gt id
    prev_step = np.full(n_gid, np.nan)  # tracker id matched at the previous matching step
    for gi, ti, sim in zip(g_ids, t_ids, sims, strict=True):
        if len(gi) == 0:
            fp += len(ti)
            continue
        if len(ti) == 0:
            fn += len(gi)
            continue
        score = 1000.0 * (ti[None, :] == prev_step[gi[:, None]]) + sim
        score[sim < THRESHOLD - EPS] = 0
        rows, cols = linear_sum_assignment(-score)
        ok = score[rows, cols] > 0 + EPS
        mg, mt = gi[rows[ok]], ti[cols[ok]]
        prev = prev_tr[mg]
        idsw += int(np.sum(~np.isnan(prev) & (mt != prev)))
        prev_tr[mg] = mt
        prev_step[:] = np.nan
        prev_step[mg] = mt
        tp += len(mg)
        fn += len(gi) - len(mg)
        fp += len(ti) - len(mg)
    return tp, fn, fp, idsw


def _identity(g_ids, t_ids, sims, n_gid, n_tid, n_gt, n_tr) -> tuple[int, int, int]:
    """TrackEval Identity.eval_sequence (IDTP, IDFN, IDFP)."""
    if n_tr == 0 or n_gt == 0:
        return 0, n_gt, n_tr
    potential = np.zeros((n_gid, n_tid))
    g_count = np.zeros(n_gid)
    t_count = np.zeros(n_tid)
    for gi, ti, sim in zip(g_ids, t_ids, sims, strict=True):
        r, c = np.nonzero(sim >= THRESHOLD)
        potential[gi[r], ti[c]] += 1
        g_count[gi] += 1
        t_count[ti] += 1
    n = n_gid + n_tid
    fp_mat, fn_mat = np.zeros((n, n)), np.zeros((n, n))
    fp_mat[n_gid:, :n_tid] = 1e10
    fn_mat[:n_gid, n_tid:] = 1e10
    fn_mat[:n_gid, :n_tid] = g_count[:, None]
    fn_mat[np.arange(n_gid), n_tid + np.arange(n_gid)] = g_count
    fp_mat[:n_gid, :n_tid] = t_count[None, :]
    fp_mat[n_gid + np.arange(n_tid), np.arange(n_tid)] = t_count
    fn_mat[:n_gid, :n_tid] -= potential
    fp_mat[:n_gid, :n_tid] -= potential
    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    idfn = int(fn_mat[rows, cols].sum())
    idfp = int(fp_mat[rows, cols].sum())
    return int(g_count.sum()) - idfn, idfn, idfp


def summarize(s: SeqStats) -> dict[str, float]:
    """TrackEval final fields; HOTA/DetA/AssA/LocA are means over ALPHAS. Ratio metrics are NaN
    when there are neither gt nor tracker boxes (e.g. a class absent from a resample)."""
    n_tr = float(s.hota_tp[0] + s.hota_fp[0])
    counts = {"IDsw": float(s.idsw), "FP": float(s.clr_fp), "FN": float(s.clr_fn)}
    if s.n_gt == 0 and n_tr == 0:
        nan = float("nan")
        return {"HOTA": nan, "DetA": nan, "AssA": nan, "LocA": nan, "MOTA": nan, "IDF1": nan,
                **counts}
    tp = s.hota_tp
    det_a = tp / np.maximum(1, tp + s.hota_fn + s.hota_fp)
    ass_a = s.ass_sum / np.maximum(1, tp)
    loc_a = np.maximum(1e-10, s.loca_sum) / np.maximum(1e-10, tp)
    mota = (s.clr_tp - s.clr_fp - s.idsw) / max(1, s.clr_tp + s.clr_fn)
    idf1 = s.idtp / max(1, s.idtp + 0.5 * s.idfp + 0.5 * s.idfn)
    return {
        "HOTA": float(np.sqrt(det_a * ass_a).mean()), "DetA": float(det_a.mean()),
        "AssA": float(ass_a.mean()), "LocA": float(loc_a.mean()),
        "MOTA": float(mota), "IDF1": float(idf1), **counts,
    }
