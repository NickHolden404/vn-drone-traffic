import numpy as np
import pandas as pd

from cvt.follow.gimbal import (
    GimbalCommand,
    GimbalParams,
    GimbalState,
    follow,
    gimbal_metrics,
    render_crop,
    step,
)

DT = 1 / 30
P = GimbalParams(kp=2.0, kd=0.2, target_frac=0.25)


def box(cx, cy, h, aspect=1.0):
    w = h * aspect
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run(state, target, params, seconds):
    out = []
    for _ in range(int(seconds / DT)):
        state, cmd, clamped = step(state, target, params, DT)
        out.append((state, cmd, clamped))
    return out


def test_static_target_converges_and_stays_smooth():
    target = box(0.7, 0.5, 0.1)
    traj = run(GimbalState(0.5, 0.5, 2.5), target, P, 2.0)
    assert abs(0.7 - traj[-1][0].cx) < P.deadband
    pans = np.array([c.pan_rate for _, c, _ in traj])
    tilts = np.array([c.tilt_rate for _, c, _ in traj])
    assert np.abs(pans).max() <= P.max_rate + 1e-9
    assert np.abs(tilts).max() <= P.max_rate + 1e-9
    accel = np.abs(np.diff(np.concatenate([[0.0], pans]))) / DT
    assert accel.max() <= P.max_accel + 1e-6


def test_zoom_holds_target_frac():
    target = box(0.5, 0.5, 0.1)
    traj = run(GimbalState(0.5, 0.5, 1.0), target, P, 8.0)
    crop_h = 1.0 / traj[-1][0].zoom
    assert abs(crop_h * P.target_frac - 0.1) / 0.1 < 0.05


def test_clamp_at_frame_edge_anti_windup():
    corner = box(0.96, 0.96, 0.125)
    traj = run(GimbalState(0.5, 0.5, 2.0), corner, P, 4.0)
    assert any(c for _, _, c in traj)
    assert traj[-1][2]
    assert traj[-1][0].cx <= 1 - 0.5 / traj[-1][0].zoom + 1e-9
    state = traj[-1][0]
    back = run(state, box(0.5, 0.5, 0.125), P, 4.0)
    assert min(s.cx for s, _, _ in back) > 0.5 - 0.05
    assert min(s.cy for s, _, _ in back) > 0.5 - 0.05


def test_lost_target_holds_still():
    s = GimbalState(0.4, 0.6, 2.0)
    s2, cmd, clamped = step(s, None, P, DT)
    assert s2 == s
    assert (cmd.pan_rate, cmd.tilt_rate, cmd.zoom_rate) == (0.0, 0.0, 0.0)
    assert clamped is False


def test_to_mavlink_units():
    m = GimbalCommand(0.5, 0.0, 0.1).to_mavlink(84.0, 16 / 9)
    assert abs(m["yaw_rate"] - 42.0) < 1e-9
    assert m["zoom_rate"] == 0.1
    up = GimbalCommand(0.0, 0.2, 0.0).to_mavlink(84.0, 16 / 9)
    assert up["pitch_rate"] < 0


def test_render_crop_size():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    assert render_crop(frame, GimbalState(0.5, 0.5, 2.0)).shape == (1080, 1920, 3)


def test_follow_and_metrics():
    n = 90
    x1, y1, x2, y2 = (np.array(v) * s for v, s in zip(box(0.6, 0.5, 0.1, 9 / 16), (1920, 1080, 1920, 1080), strict=True))
    boxes = pd.DataFrame({"frame": range(n), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    boxes.loc[40:44, ["x1", "y1", "x2", "y2"]] = np.nan
    df = follow(boxes, 1920, 1080, 30.0, P)
    assert len(df) == n
    assert {"frame", "t", "cx", "cy", "zoom", "pan_rate", "tilt_rate", "zoom_rate", "clamped"} <= set(df.columns)
    assert (df.loc[40:44, ["pan_rate", "tilt_rate", "zoom_rate"]] == 0).all().all()
    m = gimbal_metrics(df, boxes.dropna(), 1920, 1080)
    assert set(m) == {"centre_err", "inside", "smooth", "clamped_frac"}
    assert m["inside"] > 0.9
    assert 0 <= m["clamped_frac"] <= 1
    assert m["centre_err"] < 0.5
