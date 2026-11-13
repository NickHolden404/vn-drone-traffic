import numpy as np
import pandas as pd
import pytest

from cvt.analytics.anomaly import align, auc, load_labels, pooled_auc
from cvt.analytics.events import (
    event_table,
    flow_field,
    frame_scores,
    stopped_score,
    wrong_way_score,
)

FPS = 30.0

def _kin(tid, xs, ys, label=1, f0=0):
    n = len(xs)
    vx, vy = np.gradient(np.asarray(xs, float)) * FPS, np.gradient(np.asarray(ys, float)) * FPS
    return pd.DataFrame({"track_id": tid, "frame": range(f0, f0 + n), "label": label,
                         "t_s": np.arange(f0, f0 + n) / FPS, "xs_m": xs, "ys_m": ys,
                         "speed_kmh": np.hypot(vx, vy) * 3.6, "heading_rad": np.arctan2(vy, vx)})

def test_stopped_score_needs_duration_and_saturates():
    n = 3000                                   # 100 s: drive 10 s, stand 80 s, drive 10 s
    x = np.r_[np.linspace(0, 100, 300), np.full(2400, 100.0), np.linspace(100, 200, 300)]
    k = _kin(1, x, np.zeros(n))
    s = stopped_score(k, FPS, 3.0, 5.0, 60.0)
    assert s[:300].max() == 0 and s[-250:].max() == 0
    assert s[300 + 3 * 30] == 0                                 # 3 s into a long stop (causal)
    assert s[300 + 5 * 30 - 1] == 0 and s[300 + 5 * 30] == pytest.approx(5 / 60)
    assert s[300 + 30 * 30] == pytest.approx(0.5)               # 30 s into the stop
    assert s[300 + 70 * 30] == 1.0
    brief = _kin(2, np.r_[np.linspace(0, 50, 150), np.full(90, 50.0), np.linspace(50, 99, 150)],
                 np.zeros(390))
    assert stopped_score(brief, FPS, 3.0, 5.0, 60.0).max() == 0   # a 3 s stop
    ped = _kin(3, np.full(600, 5.0), np.zeros(600), label=5)      # pedestrians are not scored
    assert stopped_score(ped, FPS, 3.0, 5.0, 60.0).max() == 0

def _stop(segments, fps=FPS):
    """One car (track 1): (n_rows, speed_kmh) segments; speed None leaves `n_rows` frames out."""
    frames, speeds, f = [], [], 0
    for n, v in segments:
        if v is not None:
            frames.append(np.arange(f, f + n))
            speeds.append(np.full(n, v, float))
        f += n
    frame, speed = np.concatenate(frames), np.concatenate(speeds)
    return pd.DataFrame({"track_id": 1, "frame": frame, "label": 1, "t_s": frame / fps,
                         "speed_kmh": speed})


def test_short_tracker_gaps_do_not_break_a_stop():
    # 10 s driving, then a 60 s stop (frames 300..2100) with 3 missing frames and 5 NaN-speed rows
    k = _stop([(300, 30.0), (600, 0.0), (3, None), (600, 0.0), (5, np.nan), (593, 0.0),
               (300, 30.0)])
    s = stopped_score(k, FPS, 3.0, 5.0, 60.0, gap_s=1.0)
    last = int(np.flatnonzero(k.frame.to_numpy() == 2100)[0])
    assert s[last] == 1.0 and s[last + 1] == 0
    ev = event_table(k, s, np.zeros(len(k)), FPS, gap_s=1.0, v_stop_kmh=3.0)
    assert len(ev) == 1
    assert ev.t_start[0] == pytest.approx(10.0) and ev.t_end[0] == pytest.approx(2101 / FPS)
    assert ev.duration_s[0] == pytest.approx(1801 / FPS)
    # without bridging the same stop breaks into three runs and never saturates
    s0 = stopped_score(k, FPS, 3.0, 5.0, 60.0, gap_s=0.0)
    assert s0.max() < 0.5
    assert len(event_table(k, s0, np.zeros(len(k)), FPS, gap_s=0.0, v_stop_kmh=3.0)) == 3


def test_driving_between_two_stops_splits_them():
    k = _stop([(300, 0.0), (60, 30.0), (300, 0.0)])            # 10 s, 2 s driving, 10 s
    s = stopped_score(k, FPS, 3.0, 5.0, 60.0, gap_s=1.0)
    ev = event_table(k, s, np.zeros(len(k)), FPS, gap_s=1.0, v_stop_kmh=3.0)
    assert len(ev) == 2 and ev.t_start.tolist() == pytest.approx([0.0, 12.0])
    assert s[150] == pytest.approx(5.0 / 60.0) and s[360 + 149] == 0   # elapsed restarts


def test_wrong_way_run_bridges_a_gap():
    k = _stop([(20, 30.0), (10, None), (20, 30.0)])
    k = k.assign(xs_m=0.0, ys_m=0.0, heading_rad=0.0)
    field = {"x0": -1.0, "y0": -1.0, "cell_m": 2.0, "ux": np.array([[-1.0]]),
             "uy": np.array([[0.0]]), "ok": np.array([[True]])}
    # each piece is 20 frames (0.67 s); bridged they span 50 frames = 1.67 s >= t_wrong_s
    assert wrong_way_score(k, field, FPS, 5.0, 120.0, 1.0, gap_s=1.0).min() == 1.0
    assert wrong_way_score(k, field, FPS, 5.0, 120.0, 1.0, gap_s=0.0).max() == 0


def test_rows_against_the_condition_are_not_bridged():
    field = {"x0": -1.0, "y0": -1.0, "cell_m": 2.0, "ux": np.array([[-1.0]]),
             "uy": np.array([[0.0]]), "ok": np.array([[True]])}
    # frames 0 and 30 against the flow (heading 0), frames 1-29 present and moving with it
    k = _stop([(31, 30.0)]).assign(xs_m=0.0, ys_m=0.0, heading_rad=np.pi)
    k.loc[[0, 30], "heading_rad"] = 0.0
    w = wrong_way_score(k, field, FPS, 5.0, 120.0, 1.0, gap_s=1.0)
    assert w.max() == 0
    ev = event_table(k, np.zeros(len(k)), w, FPS, gap_s=1.0, v_stop_kmh=3.0)
    assert (ev.kind != "wrong_way").all()
    # a crawler at 4 km/h that dips to 2.5 km/h on one frame per second for 10 s
    c = _stop([(300, 4.0)])
    c.loc[c.frame % 30 == 0, "speed_kmh"] = 2.5
    s = stopped_score(c, FPS, 3.0, 5.0, 60.0, gap_s=1.0)
    assert s.max() == 0
    ev = event_table(c, s, np.zeros(len(c)), FPS, gap_s=1.0, v_stop_kmh=3.0)
    assert (ev.kind != "stopped").all()


def _one_way(n_tracks=25):
    """Eastbound traffic along y = 10 m, staggered in time; 20 m/s."""
    return [_kin(i, np.linspace(0, 100, 150), np.full(150, 10.0), f0=10 * i)
            for i in range(n_tracks)]

def test_wrong_way_against_the_learned_flow():
    wrong = _kin(99, np.linspace(100, 0, 150), np.full(150, 10.0), f0=400)
    kin = pd.concat([*_one_way(), wrong], ignore_index=True)
    field = flow_field(kin, 2.0, 5.0, 20, 0.6)
    assert field["ok"].any()
    w = wrong_way_score(kin, field, FPS, 5.0, 120.0, 1.0)
    assert w[(kin.track_id == 99).to_numpy()].mean() > 0.8
    assert w[(kin.track_id != 99).to_numpy()].max() == 0

def test_no_flow_where_traffic_is_sparse_or_two_way():
    few = pd.concat(_one_way(5), ignore_index=True)
    assert not flow_field(few, 2.0, 5.0, 20, 0.6)["ok"].any()
    both = pd.concat([*_one_way(25), *[_kin(100 + i, np.linspace(100, 0, 150),
                                           np.full(150, 10.0), f0=10 * i) for i in range(25)]],
                     ignore_index=True)
    f = flow_field(both, 2.0, 5.0, 20, 0.6)
    assert not f["ok"].any()                    # two opposing streams: no dominant direction
    assert wrong_way_score(both, f, FPS, 5.0, 120.0, 1.0).max() == 0

def test_frame_scores_and_event_table():
    k = pd.concat([_kin(1, np.zeros(10), np.zeros(10)), _kin(2, np.ones(5), np.ones(5), f0=3)],
                  ignore_index=True)
    stopped = np.r_[np.full(10, 0.2), np.zeros(5)]
    wrong = np.r_[np.zeros(10), [0, 1, 1, 1, 0]]
    fs = frame_scores(k, np.maximum(stopped, wrong), 12)
    assert fs.tolist() == [0.2] * 4 + [1.0] * 3 + [0.2] * 3 + [0.0, 0.0]
    ev = event_table(k, stopped, wrong, FPS, v_stop_kmh=3.0)
    assert set(ev.kind) == {"stopped", "wrong_way"}
    ww = ev[ev.kind == "wrong_way"].iloc[0]
    assert ww.track_id == 2 and ww.duration_s == pytest.approx(3 / FPS)

def test_auc_and_alignment(tmp_path):
    assert auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == pytest.approx(0.5)
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([1, 1])))
    s = np.arange(11402, dtype=float)
    a, b = align(s, np.ones(2281, np.int8), 5, "c")     # ceil(11402 / 5) = 2281
    assert len(a) == len(b) == 2281 and a[1] == 5.0
    a, b = align(s, np.ones(2280, np.int8), 5, "c")     # one short
    assert len(a) == len(b) == 2280
    with pytest.raises(ValueError, match="clipX"):
        align(s, np.ones(2200, np.int8), 5, "clipX")
    np.save(tmp_path / "long_name.npy", np.array([0, 1, 1], np.int8))
    assert load_labels(tmp_path, "short", {"short": "long_name"}).tolist() == [0, 1, 1]
    np.save(tmp_path / "bad.npy", np.array([0, 2], np.int8))
    with pytest.raises(ValueError, match="bad"):
        load_labels(tmp_path, "bad", {})

def test_pooled_auc_over_clips():
    rng = np.random.default_rng(0)
    clips = []
    for k in range(8):
        y = (rng.random(400) < 0.4).astype(np.int8) if k else np.ones(400, np.int8)
        clips.append((y + rng.normal(0, 0.8, 400), y))       # one clip is all-abnormal
    v, lo, hi = pooled_auc(clips, 200, 0)
    assert 0.6 < lo < v < hi < 1.0
    assert np.isnan(pooled_auc(clips[:1], 200, 0)[0])        # a single one-class clip
    v1, lo1, _ = pooled_auc(clips[1:2], 200, 0)
    assert v1 > 0.6 and np.isnan(lo1)                        # one clip: no CI
