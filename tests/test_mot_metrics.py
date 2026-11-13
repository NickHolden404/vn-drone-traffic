import math

import numpy as np
import pandas as pd
import pytest

from cvt.eval.mot import ALPHAS, MOT_COLS, SeqStats, sequence_stats, summarize

W, H = 20.0, 15.0


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=MOT_COLS)


def _box(frame, tid, x, y, w=W, h=H):
    return (frame, tid, x, y, x + w, y + h)


def _track(tid, frames, x0=10.0, y0=10.0, dx=3.0):
    return [_box(f, tid, x0 + dx * f, y0) for f in frames]


def _random_seq(seed: int, n_frames: int = 40, n_obj: int = 5):
    """Well-separated moving objects with jitter, misses, id switches/swaps and far-away FPs."""
    rng = np.random.default_rng(seed)
    spans = [sorted(rng.integers(1, n_frames + 1, 2)) for _ in range(n_obj)]
    tid = {o: 100 + o for o in range(n_obj)}
    swap_at = int(rng.integers(5, n_frames - 5))
    gt, tr = [], []
    for f in range(1, n_frames + 1):
        if f == swap_at:
            tid[0], tid[1] = tid[1], tid[0]
        for o, (s, e) in enumerate(spans):
            if not s <= f <= e:
                continue
            x, y = 10.0 + 100.0 * o + 2.0 * f, 20.0 + 1.5 * f
            gt.append(_box(f, o + 1, x, y))
            if rng.random() < 0.15:
                continue
            if rng.random() < 0.05:
                tid[o] = int(rng.integers(1000, 100000))
            j = rng.normal(0, 2.0, 4)
            tr.append((f, tid[o], x + j[0], y + j[1], x + W + j[2], y + H + j[3]))
        if rng.random() < 0.2:
            x, y = 800.0 + 100.0 * rng.random(), 20.0 + 200.0 * rng.random()
            tr.append(_box(f, 500000 + f, x, y))
    return _df(gt), _df(tr)


def test_alphas_match_trackeval():
    assert np.allclose(ALPHAS, np.arange(0.05, 0.99, 0.05)) and len(ALPHAS) == 19


def test_perfect_tracker_scores_one():
    gt = _df(_track(1, range(1, 11)) + _track(2, range(1, 11), y0=200.0))
    tr = gt.assign(id=gt["id"].map({1: 7, 2: 3}))
    m = summarize(sequence_stats(gt, tr))
    assert m["HOTA"] == pytest.approx(1) and m["MOTA"] == pytest.approx(1)
    assert m["IDF1"] == pytest.approx(1) and m["LocA"] == pytest.approx(1)
    assert m["IDsw"] == 0 and m["FP"] == 0 and m["FN"] == 0


def test_id_switch_halfway():
    gt = _df(_track(1, range(1, 11)))
    tr = _df(_track(5, range(1, 6)) + _track(6, range(6, 11)))
    m = summarize(sequence_stats(gt, tr))
    assert m["MOTA"] == pytest.approx(0.9) and m["IDsw"] == 1
    assert m["IDF1"] == pytest.approx(0.5) and m["AssA"] == pytest.approx(0.5)
    assert m["DetA"] == pytest.approx(1)
    assert m["HOTA"] == pytest.approx(0.5**0.5)


def test_missed_frames_and_false_positive():
    gt = _df(_track(1, range(1, 11)))
    tr = _df(_track(1, [1, 2, 3, 4, 5, 8, 9, 10]) + [_box(4, 9, 500.0, 300.0)])
    m = summarize(sequence_stats(gt, tr))
    assert m["FN"] == 2 and m["FP"] == 1 and m["MOTA"] == pytest.approx(1 - 3 / 10)
    assert m["DetA"] == pytest.approx(8 / 11)
    assert m["IDsw"] == 0


def test_localization_lowers_detA_only_above_iou():
    gt = _df(_track(1, range(1, 5)))
    tr = gt.assign(x1=gt["x1"] + 4.0, x2=gt["x2"] + 4.0)  # IoU = 16*15 / (24*15) = 2/3
    s = sequence_stats(gt, tr)
    expected_tp = np.where(ALPHAS <= 2 / 3, 4, 0)
    assert np.array_equal(s.hota_tp, expected_tp)
    m = summarize(s)
    hit = ALPHAS <= 2 / 3
    assert m["DetA"] == pytest.approx(hit.mean()) and m["MOTA"] == pytest.approx(1)
    # TrackEval: LocA is 1 at alphas without any match, then averaged over all alphas
    assert m["LocA"] == pytest.approx(np.where(hit, 2 / 3, 1.0).mean())


def test_empty_tracker_and_empty_gt_frames():
    gt = _df(_track(1, [1, 2, 3, 7, 8, 9]))
    m = summarize(sequence_stats(gt, _df([])))
    assert m["DetA"] == 0 and m["HOTA"] == 0 and m["MOTA"] == 0 and m["IDF1"] == 0
    assert m["FN"] == 6 and m["FP"] == 0
    m = summarize(sequence_stats(gt, gt.copy()))  # frames 4-6 have neither gt nor tracker boxes
    assert m["HOTA"] == pytest.approx(1) and m["MOTA"] == pytest.approx(1)


def test_tracker_only_sequence():
    tr = _df(_track(3, [1, 2]))
    m = summarize(sequence_stats(_df([]), tr))
    assert m["FP"] == 2 and m["HOTA"] == 0 and m["MOTA"] == -2 and m["IDF1"] == 0


def test_duplicate_id_in_frame_fails():
    gt = _df(_track(1, [1]) + _track(1, [1], y0=100.0))
    with pytest.raises(ValueError, match="more than once"):
        sequence_stats(gt, gt.iloc[:1])


def test_missing_columns_fail():
    with pytest.raises(ValueError, match="missing columns"):
        sequence_stats(_df([]).drop(columns="x2"), _df([]))


def _offset(df: pd.DataFrame, frames: int, ids: int) -> pd.DataFrame:
    return df.assign(frame=df["frame"] + frames, id=df["id"] + ids)


def test_stats_add_equals_concatenated_sequences():
    (ga, ta), (gb, tb) = _random_seq(1), _random_seq(2)
    pooled = sequence_stats(
        pd.concat([ga, _offset(gb, 1000, 10**6)]), pd.concat([ta, _offset(tb, 1000, 10**6)])
    )
    summed = sequence_stats(ga, ta) + sequence_stats(gb, tb)
    for f in ("hota_tp", "hota_fn", "hota_fp", "ass_sum", "assre_sum", "asspr_sum", "loca_sum"):
        assert np.allclose(getattr(pooled, f), getattr(summed, f), rtol=1e-12, atol=1e-12), f
    mp, ms = summarize(pooled), summarize(summed)
    assert mp.keys() == ms.keys()
    for k in mp:
        assert ms[k] == pytest.approx(mp[k], rel=1e-12, abs=1e-12), k


def test_scaled_matches_repeated_add():
    g, t = _random_seq(3)
    s = sequence_stats(g, t)
    a, b = s.scaled(3), s + s + s
    for f in SeqStats.__dataclass_fields__:
        assert np.allclose(getattr(a, f), getattr(b, f)), f
    m1, m2 = summarize(s), summarize(s.scaled(2))
    for k in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1"):
        assert m2[k] == pytest.approx(m1[k]), k
    assert m2["FP"] == 2 * m1["FP"] and m2["IDsw"] == 2 * m1["IDsw"]


def test_combined_with_absent_class():
    g, t = _random_seq(4)
    s = sequence_stats(g, t)
    assert summarize(SeqStats.zero() + s) == pytest.approx(summarize(s))
    z = summarize(SeqStats.zero())
    for k in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1"):
        assert math.isnan(z[k]), k
    assert z["FP"] == 0 and z["FN"] == 0 and z["IDsw"] == 0
    assert math.isnan(summarize(sequence_stats(_df([]), _df([])))["HOTA"])


def _crowded_seq(seed: int, n_frames: int = 40, n_obj: int = 6):
    """Objects spaced closer than their width (neighbours overlap), jittered tracker boxes
    (integer jitter on odd seeds: exact IoU ties), misses, id swaps/switches, near and far FPs."""
    rng = np.random.default_rng(seed)
    spans = [sorted(rng.integers(1, n_frames + 1, 2)) for _ in range(n_obj)]
    tid = {o: 100 + o for o in range(n_obj)}
    swaps = {int(f): (int(a), int(a + 1)) for f, a in zip(
        rng.integers(5, n_frames - 5, 2), rng.integers(0, n_obj - 1, 2), strict=True)}
    gt, tr = [], []
    for f in range(1, n_frames + 1):
        if f in swaps:
            a, b = swaps[f]
            tid[a], tid[b] = tid[b], tid[a]
        for o, (s, e) in enumerate(spans):
            if not s <= f <= e:
                continue
            x, y = 10.0 + 12.0 * o + 1.5 * f, 20.0 + 5.0 * o - 0.5 * f
            gt.append(_box(f, o + 1, x, y))
            if rng.random() < 0.15:
                continue
            if rng.random() < 0.05:
                tid[o] = int(rng.integers(1000, 100000))
            j = rng.normal(0, 3.0, 4)
            j = np.round(j) if seed % 2 else j
            tr.append((f, tid[o], x + j[0], y + j[1], x + W + j[2], y + H + j[3]))
        if rng.random() < 0.3:  # FP overlapping the objects
            x, y = 10.0 + 80.0 * rng.random(), 15.0 + 25.0 * rng.random()
            tr.append(_box(f, 500000 + f, x, y))
        if rng.random() < 0.1:  # far-away FP
            tr.append(_box(f, 600000 + f, 800.0, 20.0 + 200.0 * rng.random()))
    return _df(gt), _df(tr)


def _te_data(gt: pd.DataFrame, tr: pd.DataFrame) -> dict:
    """TrackEval's per-sequence `data` dict: timesteps = all frames from 1 to the last one,
    ids remapped to 0..n-1, similarity = TrackEval's own `_calculate_box_ious` (x0y0x1y1)."""
    from trackeval.datasets._base_dataset import _BaseDataset

    n_t = int(max(gt["frame"].max(), tr["frame"].max()))
    g_map = {v: i for i, v in enumerate(np.unique(gt["id"]))}
    t_map = {v: i for i, v in enumerate(np.unique(tr["id"]))}
    gids, tids, sims = [], [], []
    for f in range(1, n_t + 1):
        g, t = gt[gt["frame"] == f], tr[tr["frame"] == f]
        gids.append(g["id"].map(g_map).to_numpy(int))
        tids.append(t["id"].map(t_map).to_numpy(int))
        sims.append(_BaseDataset._calculate_box_ious(
            g[MOT_COLS[2:]].to_numpy(float), t[MOT_COLS[2:]].to_numpy(float), "x0y0x1y1"))
    return {
        "num_timesteps": n_t, "gt_ids": gids, "tracker_ids": tids, "similarity_scores": sims,
        "num_gt_dets": len(gt), "num_tracker_dets": len(tr),
        "num_gt_ids": len(g_map), "num_tracker_ids": len(t_map),
    }


def _te_eval(gt: pd.DataFrame, tr: pd.DataFrame, metrics) -> dict:
    data = _te_data(gt, tr)
    return {type(m).__name__: m.eval_sequence(data) for m in metrics}


def _te_summary(res: dict) -> dict:
    return {
        "HOTA": float(np.mean(res["HOTA"]["HOTA"])), "DetA": float(np.mean(res["HOTA"]["DetA"])),
        "AssA": float(np.mean(res["HOTA"]["AssA"])), "LocA": float(np.mean(res["HOTA"]["LocA"])),
        "MOTA": float(res["CLEAR"]["MOTA"]), "IDF1": float(res["Identity"]["IDF1"]),
        "IDsw": float(res["CLEAR"]["IDSW"]), "FP": float(res["CLEAR"]["CLR_FP"]),
        "FN": float(res["CLEAR"]["CLR_FN"]),
    }


def _te_metrics():
    from trackeval.metrics import CLEAR, HOTA, Identity

    cfg = {"PRINT_CONFIG": False}
    return [HOTA(), CLEAR(cfg), Identity(cfg)]


def _assert_close(ours: dict, ref: dict, label: str) -> None:
    assert ours.keys() == ref.keys()
    for k in ours:
        assert ours[k] == pytest.approx(ref[k], rel=0, abs=1e-9), f"{label}: {k}"


@pytest.mark.parametrize("seed", range(5))
def test_parity_with_trackeval(seed):
    """HOTA/CLEAR/Identity of one sequence and of two combined sequences vs TrackEval 1.3.0."""
    pytest.importorskip("trackeval")
    metrics = _te_metrics()
    seqs = [_crowded_seq(10 + seed), _crowded_seq(50 + seed), _random_seq(90 + seed)]
    ours_s = [sequence_stats(g, t) for g, t in seqs]
    refs = [_te_eval(g, t, metrics) for g, t in seqs]
    for i, (o, r) in enumerate(zip(ours_s, refs, strict=True)):
        _assert_close(summarize(o), _te_summary(r), f"seq {i}")
    assert (ours_s[0].clr_fp, ours_s[0].idsw) != (0, 0)  # the sequences do exercise FPs/switches

    for a, b in ((0, 1), (1, 2)):
        comb = {m.get_name(): m.combine_sequences({"a": refs[a][m.get_name()],
                                                   "b": refs[b][m.get_name()]})
                for m in metrics}
        _assert_close(summarize(ours_s[a] + ours_s[b]), _te_summary(comb), f"combined {a}+{b}")
