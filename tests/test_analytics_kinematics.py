import numpy as np
import pandas as pd
import pytest

from cvt.analytics.kinematics import ground_tracks, kinematics, noise_floor


def _track(tid, xs, ys, label=1):
    n = len(xs)
    return pd.DataFrame(
        {
            "seq_id": "c",
            "frame": range(n),
            "track_id": tid,
            "label": label,
            "x1": np.array(xs) - 5.0,
            "y1": np.array(ys) - 5.0,
            "x2": np.array(xs) + 5.0,
            "y2": np.array(ys) + 5.0,
        }
    )


def _identity(n):
    return np.tile(np.eye(3), (n, 1, 1)), np.ones(n, bool)


def test_constant_speed_and_heading():
    n, fps, mpp = 90, 30.0, 0.05  # 10 px/frame * 0.05 m/px * 30 fps = 15 m/s = 54 km/h
    H, valid = _identity(n)
    kin = kinematics(
        ground_tracks(_track(1, 10.0 * np.arange(n), np.full(n, 50.0)), H, valid, range(n), mpp),
        fps,
        0.5,
    )
    mid = kin.iloc[20:70]
    assert mid.speed_kmh.to_numpy() == pytest.approx(54.0, abs=0.5)
    assert mid.heading_rad.to_numpy() == pytest.approx(0.0, abs=0.02)
    assert mid.accel_ms2.abs().max() < 0.2
    assert kin.x_m.iloc[10] == pytest.approx(5.0)


def test_homography_is_applied():
    n = 30
    H = np.tile(np.diag([2.0, 2.0, 1.0]), (n, 1, 1))  # reference pixels = 2 x frame pixels
    g = ground_tracks(
        _track(1, np.full(n, 10.0), np.full(n, 20.0)), H, np.ones(n, bool), range(n), 0.1
    )
    assert g.x_m.iloc[0] == pytest.approx(2.0) and g.y_m.iloc[0] == pytest.approx(4.0)


def test_invalid_frames_and_short_tracks_have_no_speed():
    n, fps = 60, 30.0
    H, valid = _identity(n)
    valid[20:30] = False
    kin = kinematics(
        ground_tracks(_track(1, 5.0 * np.arange(n), np.zeros(n)), H, valid, range(n), 0.05),
        fps,
        0.5,
    )
    bad = kin.iloc[20:30]
    for c in ["xs_m", "ys_m", "speed_kmh", "heading_rad", "accel_ms2"]:
        assert bad[c].isna().all(), c
    assert kin.speed_kmh.iloc[40:50].notna().all()
    short = kinematics(
        ground_tracks(_track(2, [0.0, 5.0, 10.0], [0.0] * 3), *_identity(3), range(3), 0.05),
        fps,
        0.5,
    )
    assert short.speed_kmh.isna().all()


def test_noise_floor_uses_parked_tracks_only():
    n, fps = 300, 30.0
    rng = np.random.default_rng(0)
    parked = _track(1, 100 + rng.normal(0, 0.5, n), 100 + rng.normal(0, 0.5, n))
    mover = _track(2, 3.0 * np.arange(n), np.full(n, 200.0))
    creeper = _track(4, 100 + np.arange(n) * (100.0 / n), np.full(n, 300.0))  # ~5 m in total
    brief = _track(3, np.full(30, 400.0), np.full(30, 400.0))  # still, but spans 10 % only
    H, valid = _identity(n)
    kin = kinematics(
        ground_tracks(pd.concat([parked, mover, creeper, brief]), H, valid, range(n), 0.05),
        fps,
        0.5,
    )
    nf = noise_floor(kin, n, 0.9, 2.0)
    assert nf["n_parked"] == 1 and 0.0 <= nf["median_kmh"] < 3.0 and nf["p95_kmh"] < 6.0
    assert np.isnan(noise_floor(kin[kin.track_id == 2], n, 0.9, 2.0)["median_kmh"])


def test_noise_floor_ignores_pedestrians():
    n, fps = 300, 30.0
    rng = np.random.default_rng(0)
    parked = _track(1, 100 + rng.normal(0, 0.5, n), 100 + rng.normal(0, 0.5, n))
    # a pedestrian pacing within 1 m of one spot at up to 6 km/h for the whole clip
    t = np.arange(n) / fps
    ped = _track(5, 100 + 20 * np.sin(t / 0.6), np.full(n, 300.0), label=5)
    H, valid = _identity(n)
    kin = kinematics(ground_tracks(pd.concat([parked, ped]), H, valid, range(n), 0.05), fps, 0.5)
    assert 5.5 < kin[kin.track_id == 5].speed_kmh.max() < 7.0
    assert noise_floor(kin, n, 0.9, 2.0)["n_parked"] == 1
    assert np.isnan(noise_floor(kin[kin.track_id == 5], n, 0.9, 2.0)["median_kmh"])


def test_speed_is_unbiased_at_run_ends():
    n, fps, mpp = 90, 30.0, 0.05
    H, valid = _identity(n)
    kin = kinematics(
        ground_tracks(_track(1, 10.0 * np.arange(n), np.full(n, 50.0)), H, valid, range(n), mpp),
        fps,
        0.5,
    )
    assert kin.speed_kmh.to_numpy() == pytest.approx(54.0, abs=0.5)
    assert kin.accel_ms2.abs().max() < 0.5
    m = 15  # exactly the window length at 30 fps, 0.5 s
    short = kinematics(
        ground_tracks(
            _track(2, 10.0 * np.arange(m), np.full(m, 50.0)), *_identity(m), range(m), mpp
        ),
        fps,
        0.5,
    )
    assert short.speed_kmh.iloc[m // 2] == pytest.approx(54.0, abs=0.5)


def test_frame_gap_splits_the_run():
    fps, mpp = 30.0, 0.05
    tr = _track(1, 10.0 * np.arange(70), np.full(70, 50.0))
    tr = tr[(tr.frame < 30) | (tr.frame >= 40)]
    H, valid = _identity(70)
    kin = kinematics(ground_tracks(tr, H, valid, range(70), mpp), fps, 0.5)
    halves = [
        kinematics(ground_tracks(h, H, valid, range(70), mpp), fps, 0.5)
        for h in (tr[tr.frame < 30], tr[tr.frame >= 40])
    ]
    assert kin.speed_kmh.notna().all()
    for c in ["xs_m", "speed_kmh", "accel_ms2"]:
        assert kin[c].to_numpy() == pytest.approx(
            np.concatenate([h[c].to_numpy() for h in halves]), nan_ok=True
        )
