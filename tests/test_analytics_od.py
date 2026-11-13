import numpy as np
import pandas as pd
import pytest

from cvt.analytics.od import (
    arm_name,
    assign_arm,
    endpoints,
    find_arms,
    od_summary,
    od_table,
    perimeter_point,
    perimeter_pos,
)

W, H = 1000, 500  # perimeter 3000: top 0..1/3, right 1/3..1/2, bottom 1/2..5/6, left 5/6..1


def test_perimeter_pos():
    p = perimeter_pos(
        np.array([500.0, 999.0, 500.0, 1.0]), np.array([1.0, 250.0, 499.0, 250.0]), W, H
    )
    assert p == pytest.approx([500 / 3000, 1250 / 3000, 2000 / 3000, 2750 / 3000], abs=2e-3)


def _line(tid, p0, p1, f0, n=40, label=1):
    xs, ys = np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)
    return pd.DataFrame(
        {
            "track_id": tid,
            "frame": range(f0, f0 + n),
            "label": label,
            "x1": xs - 8,
            "y1": ys - 8,
            "x2": xs + 8,
            "y2": ys + 8,
        }
    )


def _scene():
    """West -> east through traffic (6 cars), north -> east turns (5 motorbikes), plus one track
    present from frame 0 and one that dies mid-frame."""
    t = [_line(i, (10, 250), (990, 250), 10 + 3 * i) for i in range(6)]
    t += [_line(10 + i, (500, 10), (990, 250), 12 + 3 * i, label=0) for i in range(5)]
    t += [_line(50, (300, 250), (990, 250), 0), _line(51, (10, 250), (500, 250), 20)]
    return pd.concat(t, ignore_index=True)


def test_endpoints_complete_and_incomplete():
    ep = endpoints(_scene(), (W, H), 0, 200, 0.06).set_index("track_id")
    assert ep.complete.sum() == 11
    assert not ep.loc[50, "complete"] and np.isnan(ep.loc[50, "entry"])  # there at the start
    assert not ep.loc[51, "complete"] and np.isnan(ep.loc[51, "exit"])  # lost mid-frame


def test_short_border_to_border_stub_is_not_complete():
    stub = _line(7, (10, 250), (990, 250), 5, n=20)          # 20 frames, west to east
    assert endpoints(stub, (W, H), 0, 200, 0.06).complete.all()
    ep = endpoints(stub, (W, H), 0, 200, 0.06, min_frames=30)
    assert not ep.complete.any() and ep[["entry", "exit"]].isna().all(axis=None)
    assert endpoints(stub, (W, H), 0, 200, 0.06, min_frames=20).complete.all()


def test_endpoints_per_frame_sizes():
    sizes = pd.DataFrame({"frame": range(300), "width": W, "height": H})
    a = endpoints(_scene(), sizes, 0, 200, 0.06)
    b = endpoints(_scene(), (W, H), 0, 200, 0.06)
    pd.testing.assert_frame_equal(a, b)


def test_arms_and_od():
    ep = endpoints(_scene(), (W, H), 0, 200, 0.06)
    arms = find_arms(np.r_[ep.entry, ep.exit], 0.03, 0.06, 5)
    assert len(arms) == 3  # north, east, west
    od = od_table(ep, arms, 0.06)
    n, e, w = (
        int(assign_arm(np.array([p]), arms, 0.06)[0])
        for p in (500 / 3000, 1250 / 3000, 2750 / 3000)
    )
    assert od.loc[(1, w, e), "n"] == 6 and od.loc[(0, n, e), "n"] == 5 and od.n.sum() == 11
    s = od_summary(ep, arms, 0.06)
    assert s["tracks"] == 13 and s["complete"] == 11 and s["arms"] == 3
    assert s["assigned"] == 11 and s["complete_frac"] == pytest.approx(11 / 13)
    assert [arm_name(i) for i in range(3)] == ["A", "B", "C"]


def test_arm_density_wraps_around_the_corner():
    pos = np.r_[np.full(4, 0.995), np.full(4, 0.005)]  # one arm across the 0/1 seam
    arms = find_arms(pos, 0.03, 0.06, 5)
    assert len(arms) == 1 and min(arms[0], 1 - arms[0]) < 0.01


def test_no_arms_gives_empty_tables():
    ep = endpoints(_line(1, (300, 250), (600, 250), 5), (W, H), 0, 200, 0.06)
    arms = find_arms(np.r_[ep.entry, ep.exit], 0.03, 0.06, 5)
    assert len(arms) == 0 and od_table(ep, arms, 0.06).empty
    assert od_summary(ep, arms, 0.06)["complete"] == 0
    assert (assign_arm(np.array([0.3, np.nan]), arms, 0.06) == -1).all()


def test_perimeter_point_inverts_perimeter_pos():
    pos = np.array([0.0, 1 / 6, 0.4, 2 / 3, 0.9, 0.9995])
    x, y = perimeter_point(pos, W, H)
    on_border = (x == 0) | (x == W) | (y == 0) | (y == H)
    assert on_border.all() and (x >= 0).all() and (x <= W).all() and (y >= 0).all()
    assert (x[1], y[1]) == pytest.approx((500, 0)) and (x[2], y[2]) == pytest.approx((1000, 200))
    assert (x[3], y[3]) == pytest.approx((500, 500)) and (x[4], y[4]) == pytest.approx((0, 300))
    d = np.abs(perimeter_pos(x, y, W, H) - pos)
    assert np.minimum(d, 1 - d).max() < 1e-9
    xs, ys = perimeter_point(0.25, W, H)
    assert (xs[0], ys[0]) == pytest.approx((750, 0))
