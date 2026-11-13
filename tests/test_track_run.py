from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cvt.track.run import TRACK_COLS, gmc_warps, load_tracker_cfg, track_sequence

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "trackers"
TRACKERS = ["bytetrack", "botsort"]
DET_COLS = ["frame", "x1", "y1", "x2", "y2", "score", "label"]


def _cfg(name: str) -> dict:
    return load_tracker_cfg(CFG_DIR / f"{name}.yaml")


def _det(frame, cx, cy, w=40.0, h=30.0, score=0.9, label=1):
    return (frame, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, label)


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DET_COLS)


def _warps(n: int) -> np.ndarray:
    return np.tile(np.eye(2, 3), (n, 1, 1))


def _run(dets, frames, name="bytetrack", **kw):
    warps = kw.pop("warps", _warps(len(frames)) if name == "botsort" else None)
    return track_sequence(dets, frames, _cfg(name), seq_id="s", warps=warps, **kw)


@pytest.mark.parametrize("name", TRACKERS)
def test_two_objects_keep_ids(name):
    rng = np.random.default_rng(0)
    frames = list(range(1, 31))
    rows = []
    for f in frames:
        for cx0, cy0, vx, vy, miss in [(100, 100, 3, 1, 15), (300, 200, -2, 0.5, None)]:
            if f == miss:
                continue
            nx, ny = rng.uniform(-1, 1, 2)
            rows.append(_det(f, cx0 + vx * f + nx, cy0 + vy * f + ny))
    out = _run(_df(rows), frames, name)
    assert list(out.columns) == TRACK_COLS
    assert out["track_id"].nunique() == 2
    per_track = out.groupby("track_id")["frame"].nunique()
    assert (per_track >= 28).all()
    assert (out["seq_id"] == "s").all() and (out["camera_id"] == "uav0").all()
    assert not out["locked"].any()


def test_output_fields_and_timestamps():
    frames = list(range(10, 20))
    rows = [_det(f, 100 + 2 * f, 100, score=0.5 + 0.01 * f) for f in frames]
    out = _run(_df(rows), frames, fps=10.0, camera_id="camX")
    assert len(out) > 0
    assert (out["camera_id"] == "camX").all()
    np.testing.assert_allclose(out["timestamp_s"], (out["frame"] - 10) / 10.0)
    by_frame = _df(rows).set_index("frame")["score"]
    np.testing.assert_allclose(out["score"], by_frame.loc[out["frame"]].to_numpy(), atol=1e-6)
    assert out["label"].dtype.kind == "i" and out["track_id"].dtype.kind == "i"
    assert out["locked"].dtype == bool
    assert (out["x2"] > out["x1"]).all()


def test_missing_frames_keep_frame_numbers():
    frames = [1, 2, 5]
    rows = [_det(f, 100 + 3 * f, 100) for f in frames]
    out = _run(_df(rows), frames)
    assert len(out) > 0
    assert set(out["frame"]) <= {1, 2, 5}


def test_empty_frames_advance_tracker_and_empty_input():
    frames = list(range(1, 8))
    rows = [_det(f, 100, 100) for f in (1, 2, 3, 6, 7)]  # frames 4, 5 empty
    out = _run(_df(rows), frames)
    assert out["track_id"].nunique() == 1
    assert set(out["frame"]) <= {1, 2, 3, 6, 7}
    empty = _run(_df([]), frames)
    assert len(empty) == 0 and list(empty.columns) == TRACK_COLS


def test_dets_outside_frames_raise():
    with pytest.raises(ValueError, match="not in frames"):
        _run(_df([_det(9, 100, 100)]), [1, 2])


def test_frames_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(_df([]), [1, 3, 2])


def test_botsort_uses_injected_warps(monkeypatch):
    import ultralytics.trackers.byte_tracker as bt
    from ultralytics.trackers.utils.gmc import GMC

    def _boom(*a, **k):
        raise AssertionError("an image was read")

    monkeypatch.setattr(cv2, "imread", _boom)
    monkeypatch.setattr(GMC, "apply", _boom)
    seen = []
    real = bt.multi_gmc
    monkeypatch.setattr(bt, "multi_gmc", lambda t, H: (seen.append(np.array(H)), real(t, H)))

    n, pan = 20, 8.0
    frames = list(range(1, n + 1))
    warps = _warps(n)
    warps[1:, 0, 2] = -pan  # camera pans right: static objects move left in the image
    rows = []
    for i, f in enumerate(frames):
        for wx, wy in [(200.0, 150.0), (400.0, 300.0)]:
            rows.append(_det(f, wx - pan * i, wy, w=24.0, h=24.0))
    out = track_sequence(_df(rows), frames, _cfg("botsort"), seq_id="s", warps=warps)
    assert out["track_id"].nunique() == 2
    assert (out.groupby("track_id")["frame"].nunique() == n).all()
    assert seen and all(np.allclose(h, warps[1]) or np.allclose(h, np.eye(2, 3)) for h in seen)
    assert any(np.allclose(h[:, 2], [-pan, 0.0]) for h in seen)


def test_botsort_requires_warps_unless_gmc_none():
    rows = [_det(f, 100, 100) for f in (1, 2, 3)]
    with pytest.raises(ValueError, match="warps"):
        track_sequence(_df(rows), [1, 2, 3], _cfg("botsort"), seq_id="s")
    cfg = _cfg("botsort") | {"gmc_method": "none"}
    out = track_sequence(_df(rows), [1, 2, 3], cfg, seq_id="s")
    assert out["track_id"].nunique() == 1


def test_warps_shape_checked():
    rows = [_det(f, 100, 100) for f in (1, 2, 3)]
    with pytest.raises(ValueError, match="shape"):
        track_sequence(_df(rows), [1, 2, 3], _cfg("botsort"), seq_id="s", warps=_warps(2))


def _single_object(labels, scores):
    frames = list(range(1, len(labels) + 1))
    rows = [
        _det(f, 100 + 2 * f, 100, score=s, label=lab)
        for f, lab, s in zip(frames, labels, scores, strict=True)
    ]
    return _run(_df(rows), frames)


def test_majority_label_and_tie_break():
    out = _single_object([1] * 6 + [3] * 4, [0.5] * 6 + [0.9] * 4)  # majority wins over score
    assert out["track_id"].nunique() == 1 and set(out["label"]) == {1}
    # tie on count: highest summed score wins, regardless of order
    out = _single_object([1, 3] * 5, [0.5, 0.9] * 5)
    assert set(out["label"]) == {3}
    out = _single_object([3, 1] * 5, [0.5, 0.9] * 5)
    assert set(out["label"]) == {1}


def test_class_filter_before_tracking():
    frames = list(range(1, 16))
    rows = []
    for f in frames:
        rows.append(_det(f, 100 + 2 * f, 100, label=1, score=0.6))
        rows.append(_det(f, 100 + 2 * f, 100, label=5, score=0.95))  # same box, higher score
        rows.append(_det(f, 300, 200 - f, label=3, score=0.8))
    dets = _df(rows)
    out = _run(dets, frames, classes=[1, 3])
    assert set(out["label"]) == {1, 3}
    assert out["track_id"].nunique() == 2
    ref = _run(dets[dets["label"].isin([1, 3])], frames)
    pd.testing.assert_frame_equal(out, ref)


def test_unknown_cfg_key_raises(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text((CFG_DIR / "bytetrack.yaml").read_text() + "bogus_key: 1\n")
    with pytest.raises(ValueError, match="bogus_key"):
        load_tracker_cfg(p)


def test_cfg_validation(tmp_path):
    base = (CFG_DIR / "botsort.yaml").read_text()
    for bad, match in [
        (base.replace("with_reid: false", "with_reid: true"), "with_reid"),
        (base.replace("track_buffer: 30\n", ""), "track_buffer"),
        (base.replace("name: botsort\n", ""), "name"),
        (base.replace("tracker_type: botsort", "tracker_type: nope"), "tracker_type"),
    ]:
        p = tmp_path / "t.yaml"
        p.write_text(bad)
        with pytest.raises(ValueError, match=match):
            load_tracker_cfg(p)


def test_shipped_cfgs_load():
    for name in TRACKERS:
        cfg = _cfg(name)
        assert cfg["name"] == name and cfg["frame_rate"] == 30
        assert cfg["tracker_type"] == name


def _write_pan_frames(tmp_path, n=4, step=6):
    rng = np.random.default_rng(1)
    big = cv2.GaussianBlur(rng.integers(0, 256, (240, 400, 3), dtype=np.uint8), (0, 0), 2)
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i}.png"
        cv2.imwrite(str(p), big[20:220, step * i : step * i + 300])
        paths.append(str(p))
    return paths


def test_gmc_warps_on_synthetic_pan(tmp_path):
    paths = _write_pan_frames(tmp_path)
    w = gmc_warps(paths)
    assert w.shape == (4, 2, 3)
    np.testing.assert_allclose(w[0], np.eye(2, 3))
    np.testing.assert_allclose(w[1:, 0, 2], -6.0, atol=1.5)  # content moves left as window pans right
    assert np.abs(w[1:, 1, 2]).max() < 1.5


def test_gmc_warps_edge_cases(tmp_path):
    assert gmc_warps([]).shape == (0, 2, 3)
    with pytest.raises(FileNotFoundError, match="nope"):
        gmc_warps([str(tmp_path / "nope.png")])
