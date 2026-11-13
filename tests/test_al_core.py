import ast
import csv
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt.al.oracle import Oracle, OracleError, Revealed, oracle_from_manifest
from cvt.al.strategies import (
    budget_picks,
    disagreement_ranking,
    random_ranking,
    variety_ranking,
)
from cvt.data.export_yolo import ExportError, export_labels, export_revealed, manifest_labels
from cvt.data.manifest import (
    REVEALED_SET,
    SET_ROLES,
    ManifestError,
    build_manifest,
    save_manifest,
    validate_manifest,
)
from tests.helpers import make_uitdrone7, make_visdrone

AL_DIR = Path(__file__).resolve().parents[1] / "cvt" / "al"


# --------------------------------------------------------------------------- rankings


def test_rankings_are_permutations_and_budgets_nest():
    ids = [f"clip{c}_{f}" for c in range(4) for f in range(300)]
    r0 = random_ranking(ids, 0)
    assert sorted(r0) == sorted(ids) and len(r0) == len(ids)
    assert r0 == random_ranking(list(reversed(ids)), 0)  # input order does not matter
    assert r0 != random_ranking(ids, 1) and random_ranking(ids, 1) != random_ranking(ids, 2)
    scores = pd.DataFrame({"image_id": ids, "disagreement": np.random.default_rng(3).random(len(ids))})
    clip_of = {i: i.split("_")[0] for i in ids}
    frame_of = {i: int(i.split("_")[1]) for i in ids}
    for r in (r0, disagreement_ranking(scores), variety_ranking(scores, clip_of, frame_of)):
        assert sorted(r) == sorted(ids)
        picks = [budget_picks(r, b) for b in (100, 200, 400, 800)]
        for small, big in zip(picks, picks[1:], strict=False):
            assert big[: len(small)] == small and len(big) == 2 * len(small)
    assert budget_picks(r0, len(ids)) == r0
    for bad in (0, -1, len(ids) + 1):
        with pytest.raises(ValueError):
            budget_picks(r0, bad)
    with pytest.raises(ValueError, match="duplicate"):
        random_ranking([*ids, ids[0]], 0)


def test_disagreement_ranking_ties():
    scores = pd.DataFrame({"image_id": ["d", "b", "a", "c", "e"],
                           "disagreement": [0.5, 0.9, 0.5, 0.5, 0.1]})
    assert disagreement_ranking(scores) == ["b", "a", "c", "d", "e"]
    with pytest.raises(ValueError, match="duplicate"):
        disagreement_ranking(pd.DataFrame({"image_id": ["a", "a"], "disagreement": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="finite"):
        disagreement_ranking(pd.DataFrame({"image_id": ["a", "b"], "disagreement": [1.0, np.nan]}))


def _three_clips(d_of):
    ids = [f"{c}_{f:02d}" for c in "abc" for f in range(40)]
    clip_of = {i: i.split("_")[0] for i in ids}
    frame_of = {i: int(i.split("_")[1]) for i in ids}
    scores = pd.DataFrame({"image_id": ids, "disagreement": [d_of(i) for i in ids]})
    return scores, clip_of, frame_of


def _min_dist(taken: list[int], f: int) -> float:
    if not taken:
        return float("inf")
    k = bisect_left(taken, f)
    return min(abs(f - taken[j]) for j in (k - 1, k) if 0 <= j < len(taken))


def test_variety_passes_and_gaps():
    # D falls with the frame number: pass 1 takes 0, 15, 30 of each clip, pass 2 (gap 5) 5, 10,
    # 20, 25, 35, then gap 0 takes the rest in frame order; clips round-robin a, b, c throughout
    scores, clip_of, frame_of = _three_clips(lambda i: -int(i.split("_")[1]))
    r = variety_ranking(scores, clip_of, frame_of)
    rr = lambda frames: [f"{c}_{f:02d}" for f in frames for c in "abc"]  # noqa: E731
    first = [f for f in range(40) if f not in (0, 15, 30, 5, 10, 20, 25, 35)]
    assert r == rr([0, 15, 30]) + rr([5, 10, 20, 25, 35]) + rr(first)

    # random D: properties of the three passes
    rng = np.random.default_rng(7)
    d = dict(zip(scores.image_id, rng.random(len(scores)), strict=True))
    scores, clip_of, frame_of = _three_clips(lambda i: d[i])
    r = variety_ranking(scores, clip_of, frame_of)
    assert sorted(r) == sorted(scores.image_id)
    by_d = sorted(d, key=lambda i: (-d[i], i))
    taken: dict[str, list[int]] = {c: [] for c in "abc"}
    gap_class, clips = [], []
    for i in r:
        c, f = clip_of[i], frame_of[i]
        dist = _min_dist(taken[c], f)
        g = 15 if dist >= 15 else 5 if dist >= 5 else 0
        # each turn takes the clip's highest-D frame that satisfies the pass gap
        free = [j for j in by_d if clip_of[j] == c and j not in r[: len(clips)]]
        best = next(j for j in free if _min_dist(taken[c], frame_of[j]) >= g)
        assert i == best
        gap_class.append(g)
        clips.append(c)
        taken[c] = sorted([*taken[c], f])
    # pass 2 only after pass 1, pass 3 only after pass 2
    assert gap_class == sorted(gap_class, reverse=True)
    n1 = gap_class.count(15)
    assert 6 <= n1 <= 9  # 40 frames hold 2 or 3 frames >= 15 apart
    # pass 1 ends only when no clip has a frame >= 15 from all its taken frames
    t1: dict[str, list[int]] = {c: [] for c in "abc"}
    for i in r[:n1]:
        t1[clip_of[i]].append(frame_of[i])
    for i in r[n1:]:
        assert _min_dist(sorted(t1[clip_of[i]]), frame_of[i]) < 15
    # pass 1 picks are >= 15 apart within a clip and go round-robin from clip a
    for c in "abc":
        fs = sorted(t1[c])
        assert all(b - a >= 15 for a, b in zip(fs, fs[1:], strict=False))
    assert clips[0] == "a"
    order = "abc"
    for k in range(n1 - 1):
        a, b = order.index(clips[k]), order.index(clips[k + 1])
        skipped = [order[(a + s) % 3] for s in range(1, (b - a) % 3 or 3)]
        # a clip is skipped only once it has nothing left at gap 15
        for c in skipped:
            done = r[: k + 1]
            left = [j for j in scores.image_id if clip_of[j] == c and j not in done]
            assert all(_min_dist(sorted(frame_of[j] for j in done if clip_of[j] == c),
                                 frame_of[j]) < 15 for j in left)


def test_variety_round_robin_pointer_continues_across_passes():
    # unequal clips, gaps (10, 0), D falls with the frame number. Pass 1 takes a0 b0 c0 b20 (a and
    # c have nothing >= 10 from frame 0 left; the last pick is in b), so pass 2 starts at the clip
    # after b: c5, a5, b5. A pointer reset to the first clip would give a5, b5, c5.
    frames = {"a": [0, 5], "b": [0, 5, 20], "c": [0, 5]}
    ids = [f"{c}_{f:02d}" for c, fs in frames.items() for f in fs]
    scores = pd.DataFrame({"image_id": ids, "disagreement": [-int(i.split("_")[1]) for i in ids]})
    clip_of = {i: i.split("_")[0] for i in ids}
    frame_of = {i: int(i.split("_")[1]) for i in ids}
    r = variety_ranking(scores, clip_of, frame_of, gaps=(10, 0))
    assert r == ["a_00", "b_00", "c_00", "b_20", "c_05", "a_05", "b_05"]


def test_variety_rejects_bad_input():
    scores, clip_of, frame_of = _three_clips(lambda i: 0.0)
    with pytest.raises(ValueError, match="gap"):
        variety_ranking(scores, clip_of, frame_of, gaps=(15, 5))
    del clip_of["a_00"]
    with pytest.raises(ValueError, match="clip"):
        variety_ranking(scores, clip_of, frame_of)


# --------------------------------------------------------------------------- oracle


def _manifest_dir(tmp_path: Path) -> Path:
    make_visdrone(tmp_path / "vd" / "train", n_images=2, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=1, seed0=100)
    make_uitdrone7(tmp_path / "uit", n_train=4)
    cfg = {"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd/train"), "val_dir": str(tmp_path / "vd/val"),
                         "rider_merge": True, "tune_frac": 0.4},
        "uitdrone7": {"images_dir": str(tmp_path / "uit/images"),
                      "labels_dir": str(tmp_path / "uit/annotations")},
    }}
    images, boxes = build_manifest(cfg, exclusions=set())
    out = tmp_path / "manifest"
    save_manifest(images, boxes, out)
    return out


def _log_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_oracle_reveals_only_requested_and_logs(tmp_path: Path):
    log = tmp_path / "reports" / "al_reveals.csv"
    oracle = oracle_from_manifest(_manifest_dir(tmp_path), log_path=log)
    assert isinstance(oracle, Oracle)
    ids = oracle.pool_ids()
    assert ids == [f"uitdrone7/train/CLIPtrain_{k}" for k in (1, 2, 3, 4)]
    frames = oracle.pool_frames()
    assert list(frames.columns) == ["image_id", "clip", "frame", "path"]  # no labels
    assert set(frames["clip"]) == {"CLIPtrain"} and sorted(frames["frame"]) == [1, 2, 3, 4]
    assert not log.exists()  # nothing logged before a reveal

    rev = oracle.reveal([ids[2], ids[0]], strategy="random-s0", budget=2)
    assert isinstance(rev, Revealed)
    assert list(rev.images.image_id) == [ids[2], ids[0]]
    assert set(rev.boxes.image_id) == {ids[0], ids[2]}
    assert len(rev.boxes) == 6  # 3 objects per synthetic UIT frame
    rows = _log_rows(log)
    assert rows[0] == ["timestamp", "strategy", "budget", "image_id"]
    assert [r[1:] for r in rows[1:]] == [["random-s0", "2", ids[2]], ["random-s0", "2", ids[0]]]
    assert all(r[0] for r in rows[1:])

    oracle.reveal([ids[1]], strategy="disagreement", budget=100)
    rows = _log_rows(log)
    assert len(rows) == 4 and rows[3][1:] == ["disagreement", "100", ids[1]]  # appended

    test_id = "uitdrone7/test/CLIPtest_1"
    for bad in ([test_id], [ids[0], "nope"], [], [ids[0], ids[0]]):
        with pytest.raises(OracleError):
            oracle.reveal(bad, strategy="random-s0", budget=4)
    with pytest.raises(OracleError):
        oracle.reveal(ids[:3], strategy="random-s0", budget=2)
    with pytest.raises(OracleError):
        oracle.reveal(ids[:1], strategy="", budget=2)
    assert len(_log_rows(log)) == 4  # refused reveals log nothing


def test_oracle_refuses_foreign_log_header(tmp_path: Path):
    log = tmp_path / "reports" / "al_reveals.csv"
    oracle = oracle_from_manifest(_manifest_dir(tmp_path), log_path=log)
    ids = oracle.pool_ids()
    log.parent.mkdir(parents=True)
    foreign = "when,who,n,what\nx,y,1,z\n"
    log.write_text(foreign, encoding="utf-8", newline="\n")
    with pytest.raises(OracleError, match="not a reveal log"):
        oracle.reveal(ids[:1], strategy="random-s0", budget=1)
    assert log.read_text(encoding="utf-8") == foreign  # untouched
    log.write_text("", encoding="utf-8")  # an empty file is a fresh log
    oracle.reveal(ids[:1], strategy="random-s0", budget=1)
    assert _log_rows(log)[0] == ["timestamp", "strategy", "budget", "image_id"]


def test_oracle_rejects_unparseable_pool_ids(tmp_path: Path):
    from cvt.al import oracle as mod

    images = pd.DataFrame({"image_id": ["uitdrone7/train/CLIP_x"], "path": ["p.jpg"], "width": [64],
                           "height": [48], "source": ["uitdrone7"], "scene_id": ["uitdrone7/CLIP"],
                           "split": ["train"], "set_name": ["UD-pool"]})
    boxes = pd.DataFrame(columns=["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"])
    with pytest.raises(OracleError, match="frame number"):
        mod.Oracle(images, boxes, tmp_path / "log.csv")


def test_pool_export_requires_oracle_reveal(tmp_path: Path):
    mdir = _manifest_dir(tmp_path)
    images = pd.read_parquet(mdir / "images.parquet")
    boxes = pd.read_parquet(mdir / "boxes.parquet")
    pool = images[images.set_name == "UD-pool"]
    labels, ignore = manifest_labels(pool, boxes)
    with pytest.raises(ExportError, match="refusing"):
        export_labels(pool, labels, ignore, "pool", tmp_path / "yolo")

    oracle = oracle_from_manifest(mdir, log_path=tmp_path / "al_reveals.csv")
    picked = oracle.pool_ids()[:2]
    lst = export_revealed(oracle.reveal(picked, strategy="random-s0", budget=2), "al", tmp_path / "yolo")
    paths = lst.read_text().split()
    assert len(paths) == 2 and all(Path(p).exists() for p in paths)
    for p in paths:
        lab = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        assert len(lab.read_text().splitlines()) == 3

    # a Revealed built around test frames is still refused
    test = images[images.set_name == "T-VN"]
    with pytest.raises(ExportError, match="pool"):
        export_revealed(Revealed(test, boxes[boxes.image_id.isin(test.image_id)]), "x", tmp_path / "y")
    assert SET_ROLES[REVEALED_SET] == "train"
    with pytest.raises(ManifestError, match=REVEALED_SET):
        validate_manifest(images.assign(set_name=REVEALED_SET), boxes)


def test_only_oracle_imports_manifest_or_names_label_sources():
    files = sorted(AL_DIR.glob("*.py"))
    assert {f.name for f in files} >= {"__init__.py", "oracle.py", "strategies.py", "prepare.py"}
    forbidden = ("boxes.parquet", "labels.parquet", "manifest_labels", "load_manifest", "load_pseudo")
    for f in files:
        if f.name != "oracle.py":
            text = f.read_text(encoding="utf-8")
            assert not [s for s in forbidden if s in text], f.name
    importers = set()
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                mods = [base, *(f"{base}.{a.name}" for a in node.names)]
            else:
                continue
            if any(m == "cvt.data.manifest" or m.startswith("cvt.data.manifest.") for m in mods):
                importers.add(f.name)
    assert importers == {"oracle.py"}
