from pathlib import Path

import pandas as pd
import pytest

from cvt.data.manifest import (
    ManifestError,
    build_manifest,
    frames_from_manifest,
    load_manifest,
    save_manifest,
    tune_scenes,
    validate_manifest,
)
from tests.helpers import make_uitdrone7, make_visdrone


def _cfg(tmp_path: Path) -> dict:
    make_visdrone(tmp_path / "vd" / "train", n_images=6, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=2, seed0=100)
    make_uitdrone7(tmp_path / "uit")
    return {
        "sources": {
            "visdrone_det": {
                "train_dir": str(tmp_path / "vd" / "train"),
                "val_dir": str(tmp_path / "vd" / "val"),
                "rider_merge": True,
                "tune_frac": 0.5,
            },
            "uitdrone7": {
                "images_dir": str(tmp_path / "uit" / "images"),
                "labels_dir": str(tmp_path / "uit" / "annotations"),
            },
        }
    }


def test_build_assigns_sets(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    counts = images.set_name.value_counts().to_dict()
    assert counts["T-VD"] == 2 and counts["UD-pool"] == 3 and counts["T-VN"] == 3
    assert counts.get("VD-train", 0) + counts.get("VD-tune", 0) == 6
    assert set(boxes.columns) == {"image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"}
    assert boxes.is_ignore.sum() == 2 * 8  # two ignore regions per VisDrone image


def test_exclusions_drop_train_side_only(tmp_path: Path):
    cfg = _cfg(tmp_path)
    images, _ = build_manifest(cfg, exclusions=set())
    tr = images[images.set_name.isin(["VD-train", "VD-tune"])].image_id.iloc[0]
    te = images[images.set_name == "T-VN"].image_id.iloc[0]
    images2, boxes2 = build_manifest(cfg, exclusions={tr, te})
    assert tr not in set(images2.image_id) and tr not in set(boxes2.image_id)
    assert te in set(images2.image_id)  # tests are never silently shrunk


def test_tune_scenes_hits_image_fraction_with_skewed_scene_sizes():
    sizes = {f"visdrone/{i:07d}": (150 if i < 35 else 8) for i in range(208)}
    picked = tune_scenes(sizes, 0.10)
    assert picked == tune_scenes(dict(reversed(list(sizes.items()))), 0.10)  # order-independent
    frac = sum(sizes[s] for s in picked) / sum(sizes.values())
    assert 0.10 <= frac < 0.10 + 150 / sum(sizes.values())


def test_visdrone_train_siblings_of_val_sequences_are_dropped(tmp_path: Path):
    cfg = _cfg(tmp_path)
    make_visdrone(tmp_path / "vd" / "val", n_images=1, seed0=1)  # val now shares sequence 0000001
    images, _ = build_manifest(cfg, exclusions=set())
    train_like = images[images.set_name.isin(["VD-train", "VD-tune"])]
    assert len(train_like) == 5
    assert "visdrone/0000001" not in set(train_like.scene_id)
    assert "visdrone/0000001" in set(images[images.set_name == "T-VD"].scene_id)


def test_validate_rejects_out_of_bounds_and_shared_scenes(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    bad = boxes.copy()
    bad.loc[bad.index[0], "x2"] = 10_000
    with pytest.raises(ManifestError, match="malformed"):
        validate_manifest(images, bad)
    shared = images.copy()
    pool = shared.set_name == "UD-pool"
    shared.loc[shared.index[pool][0], "scene_id"] = shared[shared.set_name == "T-VN"].scene_id.iloc[0]
    with pytest.raises(ManifestError, match="scenes shared"):
        validate_manifest(shared, boxes)


def test_save_load_roundtrip_and_hash_ignores_paths(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    h1 = save_manifest(images, boxes, tmp_path / "m")
    im2, bx2, h2 = load_manifest(tmp_path / "m")
    assert h1 == h2
    pd.testing.assert_frame_equal(images, im2)
    moved = images.assign(path=images.path.str.replace("uit", "elsewhere"))
    assert save_manifest(moved, boxes, tmp_path / "m2") == h1


def test_frames_from_manifest_roundtrip(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    frames = frames_from_manifest(images, boxes, "T-VD")
    assert len(frames) == 2
    assert all(len(f.ignore) == 2 for f in frames)


def test_empty_or_misconfigured_sources_raise(tmp_path: Path):
    cfg = _cfg(tmp_path)
    bad = {**cfg["sources"]["visdrone_det"], "val_dir": str(tmp_path / "nope")}
    with pytest.raises(ManifestError, match="VisDrone val"):
        build_manifest({"sources": {"visdrone_det": bad}}, exclusions=set())
    same = {**cfg["sources"]["visdrone_det"], "val_dir": cfg["sources"]["visdrone_det"]["train_dir"]}
    with pytest.raises(ManifestError, match="after dropping val siblings"):
        build_manifest({"sources": {"visdrone_det": same}}, exclusions=set())
    for frac in (0, 1, 10):
        with pytest.raises(ManifestError, match="tune_frac"):
            build_manifest({"sources": {"visdrone_det": {**cfg["sources"]["visdrone_det"], "tune_frac": frac}}}, exclusions=set())
    with pytest.raises(ManifestError, match="rider_merge"):
        build_manifest({"sources": {"visdrone_det": {**cfg["sources"]["visdrone_det"], "rider_merge": "false"}}}, exclusions=set())


def test_validate_rejects_nan_boxes(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    bad = boxes.copy()
    bad.loc[bad.index[0], "y2"] = float("nan")
    with pytest.raises(ManifestError, match="malformed"):
        validate_manifest(images, bad)


def test_hash_ignores_column_order_and_float_width(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    from cvt.data.manifest import manifest_hash

    shuffled = boxes[list(reversed(boxes.columns))].astype({"x1": "float32", "label": "int32"})
    # float32 changes values only if coordinates are not exactly representable; fixtures are
    assert manifest_hash(images[list(reversed(images.columns))], shuffled) == manifest_hash(images, boxes)


def test_frames_from_manifest_rejects_unknown_set(tmp_path: Path):
    images, boxes = build_manifest(_cfg(tmp_path), exclusions=set())
    with pytest.raises(ManifestError, match="unknown set"):
        frames_from_manifest(images, boxes, "T-XX")


def test_expected_counts_preflight(tmp_path: Path):
    cfg = _cfg(tmp_path)
    ok = {**cfg, "expected": {"T-VD": [2, 2], "T-VN": [3, 1]}}
    images, _ = build_manifest(ok, exclusions=set())
    assert (images.set_name == "T-VD").sum() == 2
    bad = {**cfg, "expected": {"T-VD": [2, 2], "T-VN": [5, 1], "UD-pool": [3, 7]}}
    with pytest.raises(ManifestError) as e:
        build_manifest(bad, exclusions=set())
    msg = str(e.value)
    assert "T-VN: got (3, 1), expected (5, 1)" in msg
    assert "UD-pool: got (3, 1), expected (3, 7)" in msg
    assert "T-VD" not in msg


def test_expected_counts_checked_before_exclusions(tmp_path: Path):
    cfg = _cfg(tmp_path)
    images, _ = build_manifest(cfg, exclusions=set())
    pool_id = images[images.set_name == "UD-pool"].image_id.iloc[0]
    images2, _ = build_manifest({**cfg, "expected": {"UD-pool": [3, 1]}}, exclusions={pool_id})
    assert (images2.set_name == "UD-pool").sum() == 2


def test_build_reports_unmatched_exclusions(tmp_path: Path):
    cfg = _cfg(tmp_path)
    images, _ = build_manifest(cfg, exclusions=set())
    tr = images[images.set_name == "UD-pool"].image_id.iloc[0]
    report: dict = {}
    build_manifest(cfg, exclusions={tr, "nope/a", "nope/b"}, report=report)
    assert report["unmatched_exclusions"] == ["nope/a", "nope/b"]
