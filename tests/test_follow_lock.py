import numpy as np
import pandas as pd
import pytest

from cvt.data.schema import CLASS_ID
from cvt.follow import lock as lock_mod
from cvt.follow.lock import LockParams, hist_similarity, hsv_hist, lock_params, run_lock

CAR = CLASS_ID["car"]
FPS = 10.0
FRAMES = list(range(0, 81))


def _rows(tid, frames, x, y=100.0, size=40.0, label=CAR):
    """One track: `x` is a callable frame -> left edge (or a constant)."""
    xs = [x(f) if callable(x) else float(x) for f in frames]
    return pd.DataFrame({
        "seq_id": "s", "frame": list(frames), "track_id": tid,
        "x1": xs, "y1": y, "x2": [v + size for v in xs], "y2": y + size,
        "score": 0.9, "label": label,
    })


def _tracks(*parts):
    return pd.concat(parts, ignore_index=True)


def _params(**kw):
    base = dict(coast_s=2.0, reid_thr=0.7, gate_growth=0.5, use_mil=False, reacquire=True,
                appearance="none")
    base.update(kw)
    return LockParams(**base)


def _box(df, f):
    r = df[df.frame == f].iloc[0]
    return r[["x1", "y1", "x2", "y2"]].to_numpy(float)


def _no_feats(frame, boxes):
    raise AssertionError("feats must not be called")


def _run(tracks, params, warps=None, read_frame=None, feats=_no_feats, init_box=None):
    if init_box is None:
        init_box = _box(tracks[tracks.track_id == 1], 0)
    return run_lock(tracks, feats, warps, FRAMES, 0, np.asarray(init_box, float), params, FPS,
                    read_frame)


class _FakeMil:
    def __init__(self, log):
        self.log = log
        self.box = None

    def init(self, img, box):
        self.log.append(("init", tuple(int(v) for v in box)))
        self.box = tuple(box)

    def update(self, img):
        self.log.append(("update", None))
        return True, self.box


@pytest.fixture
def fake_mil(monkeypatch):
    log = []
    monkeypatch.setattr(lock_mod, "make_mil", lambda: _FakeMil(log))
    return log


def _reader(calls):
    def read(frame):
        calls.append(frame)
        return np.zeros((1080, 1920, 3), np.uint8)
    return read


def test_variants_map_to_params():
    assert lock_params("id_only", 1.0, 0.7, 0.5) == LockParams(1.0, 0.7, 0.5, False, False, "none")
    assert lock_params("features", 2.0, 0.5, 1.0).appearance == "features"
    assert lock_params("hsv", 2.0, 0.5, 1.0).use_mil
    assert not lock_params("mil", 2.0, 0.5, 1.0).reacquire


def test_id_only_follows_track_and_inherits_swap():
    # object A moves right; at frame 50 the tracker gives A's ID 1 to object B (at y=400)
    a1 = _rows(1, range(0, 50), lambda f: 100 + 2 * f)
    b1 = _rows(1, range(50, 81), lambda f: 600.0, y=400.0)
    a2 = _rows(2, range(50, 81), lambda f: 100 + 2 * f)
    t = _tracks(a1, b1, a2)
    out = _run(t, _params(use_mil=False, reacquire=False))
    assert out.frame.tolist() == FRAMES
    assert (out.source == "track").all() and (out.track_id == 1).all()
    r60 = out[out.frame == 60].iloc[0]
    assert r60.x1 == 600.0 and r60.y1 == 400.0  # followed B: the swap is inherited


def test_reacquire_new_id_inside_gate():
    path = lambda f: 100 + 2 * f  # noqa: E731
    t = _tracks(
        _rows(1, range(0, 41), path),
        _rows(2, range(45, 81), path),
        _rows(3, range(43, 81), 700.0),  # new but far outside the gate
        _rows(4, range(0, 81), lambda f: path(f) + 20, y=130.0),  # in the gate, but coexisted
    )
    out = _run(t, _params())
    src = out.set_index("frame").source
    assert (src.loc[41:44] == "lost").all()
    assert out.loc[out.frame.between(41, 44), "x1"].isna().all()
    r = out.set_index("frame").loc[45:]
    assert (r.source == "track").all() and (r.track_id == 2).all()
    assert r.loc[45, "x1"] == path(45)


def test_reacquire_off_keeps_lost():
    path = lambda f: 100 + 2 * f  # noqa: E731
    t = _tracks(_rows(1, range(0, 41), path), _rows(2, range(45, 81), path))
    out = _run(t, _params(reacquire=False))
    assert (out[out.frame > 40].source == "lost").all()


def _warp_case():
    # the object is static in the world at x=500; the camera is still until frame 40, then pans
    # so the image moves +30 px/frame. ID 1 ends at 40; ID 2 is the same object from 45.
    x_img = lambda f: 500.0 + 30.0 * max(0, f - 40)  # noqa: E731
    t = _tracks(_rows(1, range(0, 41), x_img), _rows(2, range(45, 81), x_img))
    warps = np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]]), (len(FRAMES), 1, 1))
    warps[41:, 0, 2] = 30.0
    return t, warps


def test_gate_uses_camera_warp():
    t, warps = _warp_case()
    with_w = _run(t, _params(), warps=warps).set_index("frame")
    assert with_w.loc[45, "source"] == "track" and with_w.loc[45, "track_id"] == 2
    without = _run(t, _params(), warps=None).set_index("frame")
    assert (without.loc[41:, "source"] == "lost").all()


def test_coast_then_lost(fake_mil):
    t = _tracks(_rows(1, range(0, 41), lambda f: 100 + 2 * f))
    out = _run(t, _params(coast_s=1.0, use_mil=False)).set_index("frame")
    assert (out.loc[:40, "source"] == "track").all()
    assert (out.loc[41:, "source"] == "lost").all()
    assert out.loc[41:, ["x1", "y1", "x2", "y2"]].isna().all().all()
    assert (out.loc[41:, "track_id"] == -1).all()
    # with MIL: held for coast_s * fps frames, then dropped and lost
    calls = []
    out = _run(t, _params(coast_s=1.0, use_mil=True), read_frame=_reader(calls)).set_index("frame")
    assert (out.loc[41:50, "source"] == "mil").all()
    assert out.loc[41:50, "x1"].notna().all()
    assert (out.loc[51:, "source"] == "lost").all()
    assert out.loc[51:, "x1"].isna().all()
    # frames are read only during the MIL episode: the last locked frame (init) and 41..50
    assert calls == list(range(40, 51))
    assert fake_mil[0] == ("init", (180, 100, 40, 40))


def test_target_id_returns_after_coast():
    t = _tracks(_rows(1, range(0, 41), 100.0), _rows(1, range(70, 81), 100.0))
    out = _run(t, _params(coast_s=1.0, reacquire=False)).set_index("frame")
    assert (out.loc[51:69, "source"] == "lost").all()
    assert (out.loc[70:, "source"] == "track").all()


def test_init_without_overlap_starts_mil(fake_mil):
    calls = []
    other = _rows(1, range(0, 81), 1500.0)  # far from the initial box
    truck = _rows(7, range(3, 81), 505.0, y=500.0, label=CLASS_ID["truck"])
    t = _tracks(other, truck)
    init = np.array([500.0, 500.0, 540.0, 540.0])
    out = _run(t, _params(use_mil=True, coast_s=2.0), read_frame=_reader(calls),
               init_box=init).set_index("frame")
    assert (out.loc[0:2, "source"] == "mil").all()
    assert out.loc[0, ["x1", "y1", "x2", "y2"]].tolist() == init.tolist()
    # the class is unknown, so the truck is adopted
    assert (out.loc[3:, "source"] == "track").all() and (out.loc[3:, "track_id"] == 7).all()
    assert calls == [0, 1, 2]
    assert fake_mil[0] == ("init", (500, 500, 40, 40))


def test_init_nearest_centre_within_diagonal():
    t = _tracks(_rows(1, range(0, 81), 140.0), _rows(2, range(0, 81), 400.0))
    init = np.array([100.0, 100.0, 135.0, 140.0])  # no overlap with track 1, 40 px from it
    out = _run(t, _params(use_mil=False, reacquire=False), init_box=init)
    assert (out.track_id == 1).all()


def _feature_case(sim_a, sim_b):
    path = lambda f: 100 + 2 * f  # noqa: E731
    t = _tracks(
        _rows(1, range(0, 41), path),
        _rows(2, range(45, 81), lambda f: path(f) + 5),  # nearer the gate centre
        _rows(3, range(45, 81), lambda f: path(f) - 15),
    )
    e = {1: [1.0, 0.0], 2: [sim_a, np.sqrt(1 - sim_a ** 2)], 3: [sim_b, np.sqrt(1 - sim_b ** 2)]}
    by_x = {(int(r.frame), round(r.x1, 3)): int(r.track_id) for r in t.itertuples()}

    def feats(frame, boxes):
        return np.array([e[by_x[(frame, round(float(b[0]), 3))]] for b in boxes], np.float16)

    return t, feats


def test_two_candidates_feature_tiebreak_and_veto():
    t, feats = _feature_case(0.6, 0.9)
    p = _params(appearance="features", reid_thr=0.7)
    out = run_lock(t, feats, None, FRAMES, 0, _box(t, 0), p, FPS, None).set_index("frame")
    assert out.loc[45, "track_id"] == 3 and out.loc[45, "source"] == "track"
    # motion only would take the nearer one
    out = run_lock(t, _no_feats, None, FRAMES, 0, _box(t, 0), _params(), FPS, None)
    assert out.set_index("frame").loc[45, "track_id"] == 2
    # both below reid_thr: vetoed
    t, feats = _feature_case(0.6, 0.5)
    out = run_lock(t, feats, None, FRAMES, 0, _box(t, 0), p, FPS, None)
    assert (out[out.frame > 40].source == "lost").all()


def _colour_reader(calls, colours):
    """Frames painted with each track's colour (BGR) at its box."""
    def read(frame):
        calls.append(frame)
        img = np.zeros((400, 600, 3), np.uint8)
        for box, bgr in colours(frame):
            x1, y1, x2, y2 = (int(v) for v in box)
            img[y1:y2, x1:x2] = bgr
        return img
    return read


def test_hsv_histogram_similarity():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :50] = (0, 0, 255)
    img[:, 50:] = (255, 0, 0)
    red = hsv_hist(img, np.array([0, 0, 50, 100.0]))
    blue = hsv_hist(img, np.array([50, 0, 100, 100.0]))
    assert red.shape == (16, 4) and abs(red.sum() - 1) < 1e-6
    assert hist_similarity(red, red) == pytest.approx(1.0)
    assert hist_similarity(red, blue) < 0.1
    assert hsv_hist(img, np.array([200, 200, 300, 300.0])) is None


def test_hsv_variant_picks_same_colour():
    path = lambda f: 100 + 2 * f  # noqa: E731
    t = _tracks(
        _rows(1, range(0, 41), path),
        _rows(2, range(45, 81), lambda f: path(f) + 5),  # nearer, but blue
        _rows(3, range(45, 81), path, y=160.0),  # 60 px off, still in the gate; red like the target
    )
    red, blue = (0, 0, 255), (255, 0, 0)

    def colours(frame):
        g = t[t.frame == frame]
        return [(r[["x1", "y1", "x2", "y2"]].to_numpy(float), blue if r.track_id == 2 else red)
                for _, r in g.iterrows()]

    calls = []
    p = _params(appearance="hsv", reid_thr=0.7)
    out = run_lock(t, _no_feats, None, FRAMES, 0, _box(t, 0), p, FPS,
                   _colour_reader(calls, colours)).set_index("frame")
    assert out.loc[45, "track_id"] == 3
    assert set(range(0, 41)) <= set(calls)  # running mean over the locked frames


def test_real_opencv_mil_follows_moving_box():
    def img(f):
        a = np.full((240, 320, 3), 40, np.uint8)
        x = 100 + 3 * f
        a[100:140, x:x + 40] = (0, 200, 255)
        a[110:120, x + 10:x + 20] = (255, 255, 255)
        return a

    import cv2

    cv2.setRNGSeed(0)  # MIL samples patches at random; fixed seed, same run every time
    t = _rows(1, range(0, 11), lambda f: 100 + 3 * f)
    out = run_lock(t, _no_feats, None, list(range(25)), 0, _box(t, 0),
                   _params(use_mil=True, reacquire=False), FPS, img).set_index("frame")
    m = out.loc[11:]
    assert (m.source == "mil").all()
    err = (m.x1 - (100 + 3 * m.index.to_numpy())).abs()
    # MIL drifts a little and recovers (worst frame up to ~28 px over 200 seeds); the box must
    # keep overlapping the 40-px target and follow it on average
    assert err.max() < 40 and err.mean() < 25


def test_reacquired_id_starts_a_fresh_velocity_history():
    # ID 1 moves +10 px/frame to frame 40; ID 2 (a static object inside the gate) is adopted at
    # 42 and seen only at 42-43; ID 3 then appears at ID 2's place. ID 1's velocity must not
    # carry over to ID 2: with a mixed history the gate would drift ~6.7 px/frame and miss ID 3.
    t = _tracks(
        _rows(1, range(0, 41), lambda f: 100 + 10 * f),
        _rows(2, range(42, 44), 500.0),
        _rows(3, range(70, 81), 500.0),
    )
    out = _run(t, _params(coast_s=10.0, use_mil=False)).set_index("frame")
    assert out.loc[42, "track_id"] == 2 and out.loc[43, "track_id"] == 2
    assert (out.loc[44:69, "source"] == "lost").all()
    assert (out.loc[70:, "source"] == "track").all() and (out.loc[70:, "track_id"] == 3).all()


def test_lock_clears_history_only_when_the_id_changes():
    s = lock_mod._Lock(_params(), FPS, FRAMES, None, _no_feats, None)
    box = lambda x: np.array([x, 100.0, x + 40, 140.0])  # noqa: E731
    for i in range(3):
        s.lock(i, 1, box(100 + 10 * i), CAR, np.array([1]))
    assert len(s.hist) == 3
    s.lock(4, 2, box(500), CAR, np.array([2]))
    assert [h[0] for h in s.hist] == [4]
    s.lock(5, 2, box(500), CAR, np.array([2]))
    assert [h[0] for h in s.hist] == [4, 5]
