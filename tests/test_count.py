import math

import pandas as pd
import pytest

from cvt.track.count import (
    Line,
    border_margin,
    count_errors,
    count_table,
    crossings,
    first_crossings,
    load_scene,
    near_border,
)

LINE = Line("l", (0.0, 0.0), (100.0, 0.0))  # d = +x; cross(d, v) = 100 * v.y, so y > 0 is "pos"


def _pts(rows, track_id=1, label=1):
    """rows: (frame, x, y[, valid]) -> a points frame for one track."""
    recs = []
    for r in rows:
        valid = r[3] if len(r) > 3 else True
        recs.append(
            {"track_id": track_id, "frame": r[0], "x": r[1], "y": r[2], "label": label,
             "valid": valid}
        )
    return pd.DataFrame(recs)


def test_single_crossing_each_direction():
    down = crossings(_pts([(0, 50, -10), (1, 50, -4), (2, 50, 4), (3, 50, 10)]), LINE)
    assert down[["track_id", "label", "frame", "direction"]].values.tolist() == [[1, 1, 2, "pos"]]
    up = crossings(_pts([(0, 50, 10), (1, 50, 4), (2, 50, -4), (3, 50, -10)]), LINE)
    assert up["direction"].tolist() == ["neg"]
    assert up["frame"].tolist() == [2]


def test_jitter_on_line_counts_once():
    ys = [-10, -5, 1, -1, 1, -1, 0.5, 1, 5, 10]
    ev = crossings(_pts([(i, 50, y) for i, y in enumerate(ys)]), LINE)
    assert len(ev) == 1
    assert ev["direction"].tolist() == ["pos"]
    assert ev["frame"].tolist() == [8]  # first non-zero side on the far side
    # jitter that returns to the original side is no crossing
    back = crossings(_pts([(i, 50, y) for i, y in enumerate([-10, -5, 1, -1, 1, -5, -10])]), LINE)
    assert back.empty


def test_first_crossings_keep_each_tracks_first_event():
    # rule 2: track 1 crosses, comes back and crosses again (3 events), track 2 once
    a = _pts([(0, 50, -10), (1, 50, 10), (2, 50, -10), (3, 50, 10)], track_id=1)
    b = _pts([(5, 20, 10), (6, 20, -10)], track_id=2, label=2)
    ev = crossings(pd.concat([b, a]), LINE)
    assert len(ev) == 4
    first = first_crossings(ev)
    assert first[["track_id", "frame", "direction"]].values.tolist() == [[1, 1, "pos"],
                                                                         [2, 6, "neg"]]
    assert list(first.columns) == list(ev.columns)
    assert first_crossings(ev.iloc[:0]).empty


def test_near_border_margin():
    # m = max(2, 0.01 * min(W, H)): 2 px on a 128 x 96 image, 10.8 px on a 1920 x 1080 one
    assert border_margin(128, 96) == pytest.approx(2.0)
    assert border_margin(1920, 1080) == pytest.approx(10.8)
    x1, y1, x2, y2 = (pd.Series(v, dtype=float) for v in (
        [2.0, 2.5, 50, 50, 50, 50], [50, 50, 2.0, 50, 50, 50],
        [60, 60, 60, 126.0, 125.5, 60], [60, 60, 60, 60, 60, 94.0]))
    got = near_border(x1, y1, x2, y2, 128, 96)
    assert got.tolist() == [True, False, True, True, False, True]
    # per-row image sizes
    assert near_border(x1, y1, x2, y2, [128] * 6, [96] * 6).tolist() == got.tolist()


def test_crossing_outside_segment_ignored():
    assert crossings(_pts([(0, 150, -10), (1, 150, 10)]), LINE).empty
    assert crossings(_pts([(0, -50, -10), (1, -50, 10)]), LINE).empty
    # diagonal path that crosses the infinite line beyond p2
    assert crossings(_pts([(0, 90, -10), (1, 190, 10)]), LINE).empty
    # ... and one that crosses inside the segment
    assert len(crossings(_pts([(0, 40, -10), (1, 60, 10)]), LINE)) == 1


def test_invalid_frames_skipped():
    rows = [(0, 50, -10, True), (1, 50, 30, False), (2, 50, 40, False), (3, 50, 10, True)]
    ev = crossings(_pts(rows), LINE)
    assert ev["frame"].tolist() == [3]  # side change measured from frame 0 to frame 3
    # an invalid point on the far side alone creates no event
    assert crossings(_pts([(0, 50, -10, True), (1, 50, 10, False)]), LINE).empty


def test_unsorted_input_and_multiple_tracks():
    a = _pts([(3, 50, 10), (0, 50, -10), (2, 50, 4), (1, 50, -4)], track_id=1, label=1)
    b = _pts([(0, 20, 10), (1, 20, -10)], track_id=2, label=2)
    ev = crossings(pd.concat([a, b]), LINE).sort_values("track_id")
    assert ev["track_id"].tolist() == [1, 2]
    assert ev["label"].tolist() == [1, 2]
    assert ev["direction"].tolist() == ["pos", "neg"]


def test_empty_points_and_degenerate_line():
    empty = pd.DataFrame(columns=["track_id", "frame", "x", "y", "label", "valid"])
    out = crossings(empty, LINE)
    assert out.empty
    assert list(out.columns) == ["track_id", "label", "frame", "direction"]
    with pytest.raises(ValueError, match="distinct"):
        crossings(_pts([(0, 0, 0)]), Line("z", (1.0, 1.0), (1.0, 1.0)))


def test_count_table():
    ev = pd.DataFrame(
        {"track_id": [1, 2, 3], "label": [1, 1, 2], "frame": [5, 6, 7],
         "direction": ["pos", "pos", "neg"]}
    )
    tab = count_table(ev)
    assert tab.index.names == ["label", "direction"]
    assert tab["n"].to_dict() == {(1, "pos"): 2, (2, "neg"): 1}
    assert count_table(ev.iloc[:0]).empty


def _table(cells):
    idx = pd.MultiIndex.from_tuples(list(cells), names=["label", "direction"])
    return pd.DataFrame({"n": list(cells.values())}, index=idx)


def test_count_errors_union_of_cells():
    gt = _table({(1, "pos"): 3})
    pred = _table({(1, "pos"): 2, (2, "neg"): 1})
    r = count_errors(gt, pred)
    assert r["count_MAE"] == pytest.approx(1.0)
    assert r["count_pct_err"] == pytest.approx(2 / 3)
    assert r["gt_total"] == 3
    assert r["pred_total"] == 3


def test_count_errors_empty_cases():
    empty = count_table(pd.DataFrame(columns=["track_id", "label", "frame", "direction"]))
    r = count_errors(empty, empty)
    assert math.isnan(r["count_MAE"]) and math.isnan(r["count_pct_err"])
    assert r["gt_total"] == 0 and r["pred_total"] == 0
    r = count_errors(empty, _table({(1, "pos"): 2}))
    assert r["count_MAE"] == pytest.approx(2.0)
    assert math.isnan(r["count_pct_err"])


def test_load_scene(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "seq_id: a\nset_name: T\nref_frame: 0\nlines:\n- {name: l1, p1: [0, 1], p2: [5, 6]}\n",
        encoding="utf-8",
    )
    sc = load_scene(p)
    assert sc["seq_id"] == "a" and sc["ref_frame"] == 0
    assert sc["lines"] == [Line("l1", (0.0, 1.0), (5.0, 6.0))]


@pytest.mark.parametrize(
    "text",
    [
        "set_name: T\nref_frame: 0\nlines: []\n",  # missing seq_id
        "seq_id: a\nset_name: T\nref_frame: 0\nlines:\n- {name: l, p1: [1, 1], p2: [1, 1]}\n",
        "seq_id: a\nset_name: T\nref_frame: 0\nlines:\n- {name: l, p1: [1, 1, 3], p2: [2, 2]}\n",
        "seq_id: a\nset_name: T\nref_frame: 0\nlines:\n- {p1: [1, 1], p2: [2, 2]}\n",
        "seq_id: a\nset_name: T\nref_frame: 0\nlines: bad\n",
    ],
)
def test_load_scene_rejects_bad_files(tmp_path, text):
    p = tmp_path / "bad_scene.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bad_scene.yaml"):
        load_scene(p)
