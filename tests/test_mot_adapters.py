from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt.data.adapters.uavdt import SEQ_ATTR_COLS, load_uavdt
from cvt.data.adapters.visdrone_mot import MOT_GT_COLS, load_visdrone_mot
from cvt.data.manifest import (
    SET_ROLES,
    ManifestError,
    build_manifest,
    load_manifest,
    load_mot,
    manifest_hash,
    save_manifest,
)
from cvt.data.schema import CLASS_ID
from tests.helpers import (
    make_uavdt,
    make_uitdrone7,
    make_visdrone,
    make_visdrone_mot,
    uavdt_attributes,
)

SEQ = "uav0000013_00000_v"


def test_visdrone_mot_parses_ids_ignore_and_riders(tmp_path: Path):
    root = make_visdrone_mot(tmp_path / "mot")
    frames, gt = load_visdrone_mot(root, "val", rider_merge=True)
    assert [f.image_id for f in frames] == [
        f"visdrone_mot/val/{SEQ}/0000001", f"visdrone_mot/val/{SEQ}/0000002"]
    assert {f.scene_id for f in frames} == {f"visdrone_mot/{SEQ}"}
    assert {f.source for f in frames} == {"visdrone_mot"}
    f1 = frames[0]
    np.testing.assert_array_equal(f1.labels, [CLASS_ID["motorbike"], CLASS_ID["pedestrian"]])
    np.testing.assert_allclose(f1.boxes[0], [10, 6, 30, 24])  # motor grown to include the rider
    np.testing.assert_allclose(sorted(f1.ignore.tolist()), [[0, 40, 10, 48], [30, 30, 40, 40]])
    assert list(gt.columns) == MOT_GT_COLS
    # rider (track 2), category 0 (track 4) and score-0 (track 5) rows are not in the GT table
    assert gt[["frame", "track_id", "label"]].values.tolist() == [
        [1, 1, CLASS_ID["motorbike"]], [1, 3, CLASS_ID["pedestrian"]],
        [2, 1, CLASS_ID["motorbike"]], [2, 3, CLASS_ID["pedestrian"]]]
    assert set(gt.seq_id) == {SEQ}
    assert gt.image_id.tolist()[0] == f"visdrone_mot/val/{SEQ}/0000001"
    np.testing.assert_allclose(gt[["x1", "y1", "x2", "y2"]].iloc[0], [10, 6, 30, 24])

    _, gt2 = load_visdrone_mot(root, "val", rider_merge=False)
    assert 2 in set(gt2.track_id)  # without merging the rider keeps its row


def test_frame_index_from_file_not_position(tmp_path: Path):
    lines = {1: ["1,1,10,10,20,14,1,4,0,0"], 2: ["2,1,11,10,20,14,1,4,0,0"],
             5: ["5,1,15,10,20,14,1,4,0,0"]}
    root = make_visdrone_mot(tmp_path / "mot", frames=lines)
    frames, gt = load_visdrone_mot(root, "test", rider_merge=True)
    assert [f.image_id.rsplit("/", 1)[1] for f in frames] == ["0000001", "0000002", "0000005"]
    assert gt.frame.tolist() == [1, 2, 5]
    assert gt.frame.dtype == np.int64 and gt.track_id.dtype == np.int64


def test_visdrone_mot_empty_frame_and_missing_image(tmp_path: Path):
    root = make_visdrone_mot(tmp_path / "mot")
    (root / "annotations" / f"{SEQ}.txt").write_text("1,1,10,10,20,14,1,4,0,0\n")
    frames, gt = load_visdrone_mot(root, "val", rider_merge=True)
    assert len(frames) == 2 and len(frames[1].boxes) == 0 and gt.frame.tolist() == [1]
    (root / "annotations" / f"{SEQ}.txt").write_text("9,1,10,10,20,14,1,4,0,0\n")
    with pytest.raises(FileNotFoundError, match="0000009"):
        load_visdrone_mot(root, "val", rider_merge=True)


def test_visdrone_mot_target_id_0_is_a_real_track(tmp_path: Path):
    # real files: ids start at 0 and id 0 is an ordinary object (score 1, category 1/2/4/6)
    lines = {1: ["1,0,10,10,20,14,1,4,0,1", "1,1,40,20,6,12,1,1,1,2"],
             2: ["2,0,11,10,20,14,1,4,0,1"]}
    root = make_visdrone_mot(tmp_path / "mot", frames=lines)
    _, gt = load_visdrone_mot(root, "val", rider_merge=True)
    assert gt[["frame", "track_id", "label"]].values.tolist() == [
        [1, 0, CLASS_ID["car"]], [1, 1, CLASS_ID["pedestrian"]], [2, 0, CLASS_ID["car"]]]


def _uavdt(images_dir: Path, gt_dir: Path, seqs=("M0101",), attributes=None):
    attributes = uavdt_attributes(seqs) if attributes is None else attributes
    return load_uavdt(images_dir, gt_dir, list(seqs), attributes, "test")


def test_uavdt_mot_gt_joins_whole_for_classes_and_ignores_unevaluated(tmp_path: Path):
    images_dir, gt_dir = make_uavdt(tmp_path / "u", seqs=("M0101", "M0202"))
    frames, gt, attrs = _uavdt(images_dir, gt_dir, seqs=("M0202",))  # only the listed sequences
    assert [f.image_id for f in frames] == ["uavdt/M0202/000001", "uavdt/M0202/000002"]
    assert {f.scene_id for f in frames} == {"uavdt/M0202"} and frames[0].source == "uavdt"
    assert {f.split for f in frames} == {"test"}
    np.testing.assert_array_equal(
        frames[0].labels, [CLASS_ID["car"], CLASS_ID["truck"]])  # 1 car, 2 truck
    np.testing.assert_array_equal(frames[1].labels, [CLASS_ID["car"], CLASS_ID["bus"]])  # 3 bus
    np.testing.assert_allclose(frames[0].boxes[1], [40, 20, 52, 32])
    # frame 1: _gt_ignore row + the _gt_whole-only row (clipped); frame 2: _gt_ignore row +
    # the score-0 _gt row
    np.testing.assert_allclose(sorted(frames[0].ignore.tolist()), [[0, 40, 10, 48],
                                                                   [50, 0, 64, 10]])
    np.testing.assert_allclose(sorted(frames[1].ignore.tolist()), [[0, 40, 10, 48],
                                                                   [30, 2, 38, 8]])
    assert list(gt.columns) == MOT_GT_COLS
    assert gt[["frame", "track_id", "label"]].values.tolist() == [
        [1, 1, 1], [1, 2, 3], [2, 1, 1], [2, 3, 2]]
    assert set(gt.seq_id) == {"M0202"}
    # detection GT (FrameAnn.boxes) is the same set of boxes as the MOT GT
    for f in frames:
        g = gt[gt.image_id == f.image_id]
        np.testing.assert_allclose(g[["x1", "y1", "x2", "y2"]].to_numpy(), f.boxes)
        np.testing.assert_array_equal(g.label.to_numpy(), f.labels)
    assert list(attrs.columns) == SEQ_ATTR_COLS
    (row,) = attrs.to_dict("records")
    assert row == {"seq_id": "M0202", "daylight": True, "night": False, "fog": False,
                   "low_alt": False, "medium_alt": True, "high_alt": False,
                   "front_view": True, "side_view": False, "bird_view": False}
    assert all(attrs[c].dtype == bool for c in SEQ_ATTR_COLS[1:])


def test_uavdt_join_errors(tmp_path: Path):
    images_dir, gt_dir = make_uavdt(tmp_path / "u")
    gt_file = gt_dir / "M0101_gt.txt"
    gt_file.write_text("1,1,10,10,20,14,1,1,-1\n1,9,10,10,20,14,1,1,-1\n")
    with pytest.raises(ValueError, match=r"M0101_gt\.txt:2: .*frame 1 track 9.*M0101_gt_whole"):
        _uavdt(images_dir, gt_dir)
    gt_file.write_text("1,1,10,10,20,14,1,1,-1\n1,2,41,20,12,12,1,1,-1\n")
    with pytest.raises(ValueError, match=r"M0101_gt\.txt:2: .*box differs"):
        _uavdt(images_dir, gt_dir)
    gt_file.write_text("1,1,10,10,20,14,2,1,-1\n")
    with pytest.raises(ValueError, match=r"M0101_gt\.txt:1: score must be 0 or 1"):
        _uavdt(images_dir, gt_dir)
    gt_file.write_text("1,1,10,10,20,14,1,1\n")
    with pytest.raises(ValueError, match=r"M0101_gt\.txt:1: expected 9"):
        _uavdt(images_dir, gt_dir)
    gt_file.write_text("1,1,10,10,20,14,1,1,-1\n1,1,11,10,20,14,1,1,-1\n")
    with pytest.raises(ValueError, match=r"M0101_gt\.txt:2: track 1 appears twice in frame 1"):
        _uavdt(images_dir, gt_dir)
    gt_file.write_text("1,1,10,10,20,14,1,1,-1\n9,1,10,10,20,14,1,1,-1\n")
    (gt_dir / "M0101_gt_whole.txt").write_text("1,1,10,10,20,14,1,1,1\n9,1,10,10,20,14,1,1,1\n")
    with pytest.raises(FileNotFoundError, match="img000009"):
        _uavdt(images_dir, gt_dir)
    gt_file.unlink()
    with pytest.raises(FileNotFoundError, match="M0101_gt.txt"):
        _uavdt(images_dir, gt_dir)


def test_uavdt_sequence_and_attribute_errors(tmp_path: Path):
    images_dir, gt_dir = make_uavdt(tmp_path / "u", seqs=("M0101", "M0202"))
    attrs = uavdt_attributes(("M0101",))
    with pytest.raises(ValueError, match="unknown UAVDT attribute.*dusk"):
        _uavdt(images_dir, gt_dir, attributes={**attrs, "dusk": []})
    del attrs["bird_view"]
    with pytest.raises(ValueError, match="missing UAVDT attribute.*bird_view"):
        _uavdt(images_dir, gt_dir, attributes=attrs)
    attrs = uavdt_attributes(("M0101", "M0202"))
    with pytest.raises(ValueError, match="daylight.*M0202.*not in the sequence list"):
        _uavdt(images_dir, gt_dir, attributes=attrs)
    with pytest.raises(ValueError, match=r"listed twice: \['M0101'\]"):
        _uavdt(images_dir, gt_dir, seqs=("M0101", "M0101"))
    with pytest.raises(ValueError, match="no UAVDT sequences"):
        _uavdt(images_dir, gt_dir, seqs=())
    with pytest.raises(FileNotFoundError, match="M0303"):
        _uavdt(images_dir, gt_dir, seqs=("M0303",))


def test_unknown_category_raises_with_line(tmp_path: Path):
    root = make_visdrone_mot(tmp_path / "mot")
    ann = root / "annotations" / f"{SEQ}.txt"
    ann.write_text("1,1,10,10,20,14,1,4,0,0\n2,1,10,10,20,14,1,42,0,0\n")
    with pytest.raises(ValueError, match=rf"{SEQ}\.txt:2: unknown VisDrone category 42"):
        load_visdrone_mot(root, "val", rider_merge=True)
    ann.write_text("1,1,10,10,20,14,1,4,0\n")
    with pytest.raises(ValueError, match=rf"{SEQ}\.txt:1: expected 10 comma-separated fields"):
        load_visdrone_mot(root, "val", rider_merge=True)

    images_dir, gt_dir = make_uavdt(tmp_path / "u")
    (gt_dir / "M0101_gt.txt").write_text("1,1,10,10,20,14,1,1,-1\n")
    whole = gt_dir / "M0101_gt_whole.txt"
    whole.write_text("1,1,10,10,20,14,1,1,1\n1,2,40,20,12,12,1,1,4\n")
    with pytest.raises(ValueError, match=r"M0101_gt_whole.txt:2: unknown UAVDT category 4"):
        _uavdt(images_dir, gt_dir)
    whole.write_text("1,1,10,10,20,14,1,1\n")
    with pytest.raises(ValueError, match=r"M0101_gt_whole.txt:1: expected 9"):
        _uavdt(images_dir, gt_dir)
    whole.write_text("1,1,10,10,20,14,1,1,1\n")
    (gt_dir / "M0101_gt_ignore.txt").write_text("1,0,0,40,10,8\n")  # real files have 9 fields
    with pytest.raises(ValueError, match=r"M0101_gt_ignore.txt:1: expected 9"):
        _uavdt(images_dir, gt_dir)


def test_visdrone_mot_duplicate_rows(tmp_path: Path):
    lines = {1: ["1,1,10,10,20,14,1,4,0,0", "1,3,40,20,6,12,1,1,0,0", "1,1,10,10,20,14,1,4,0,0",
                 "1,0,0,40,10,8,0,0,0,0", "1,0,30,30,10,10,0,0,0,0"],  # ignore rows may share ids
             2: ["2,1,11,10,20,14,1,4,0,0"]}
    root = make_visdrone_mot(tmp_path / "mot", frames=lines)
    frames, gt = load_visdrone_mot(root, "val", rider_merge=True)
    assert gt[["frame", "track_id"]].values.tolist() == [[1, 1], [1, 3], [2, 1]]  # order kept
    assert len(frames[0].boxes) == 2 and len(frames[0].ignore) == 2
    ann = root / "annotations" / f"{SEQ}.txt"
    ann.write_text("1,1,10,10,20,14,1,4,0,0\n2,1,11,10,20,14,1,4,0,0\n1,1,12,10,20,14,1,4,0,0\n")
    with pytest.raises(ValueError, match=rf"{SEQ}\.txt:3: track 1 appears twice in frame 1"):
        load_visdrone_mot(root, "val", rider_merge=True)


def test_uavdt_duplicate_rows(tmp_path: Path):
    images_dir, gt_dir = make_uavdt(tmp_path / "u")
    (gt_dir / "M0101_gt.txt").write_text(
        "1,1,10,10,20,14,1,1,-1\n1,2,40,20,12,12,1,1,-1\n1,1,10,10,20,14,1,1,-1\n"
        "2,1,11,10,20,14,1,1,-1\n")
    whole = gt_dir / "M0101_gt_whole.txt"
    whole.write_text("1,1,10,10,20,14,1,1,1\n1,2,40,20,12,12,1,1,2\n1,1,10,10,20,14,1,1,1\n"
                     "2,1,11,10,20,14,1,1,1\n")
    frames, gt, _ = _uavdt(images_dir, gt_dir)
    assert gt[["frame", "track_id"]].values.tolist() == [[1, 1], [1, 2], [2, 1]]
    assert len(frames[0].ignore) == 1  # the duplicate whole row is not an extra ignore region
    whole.write_text("1,1,10,10,20,14,1,1,1\n1,2,40,20,12,12,1,1,2\n1,2,41,20,12,12,1,1,2\n")
    with pytest.raises(ValueError, match=r"M0101_gt_whole\.txt:3: track 2 appears twice in frame 1"):
        _uavdt(images_dir, gt_dir)


def _mot_cfg(tmp_path: Path) -> dict:
    make_visdrone_mot(tmp_path / "mot" / "val", seqs=(SEQ,), seed0=0)
    make_visdrone_mot(tmp_path / "mot" / "test", seqs=("uav0000099_00000_v", "uav0000098_00000_v"),
                      seed0=100)
    images_dir, gt_dir = make_uavdt(tmp_path / "u", seqs=("M0101", "M0202", "M0303"))
    seqs = ["M0101", "M0202"]  # M0303 is on disk but not a test sequence
    return {
        "sources": {
            "visdrone_mot": {"val_dir": str(tmp_path / "mot" / "val"),
                             "test_dir": str(tmp_path / "mot" / "test"), "rider_merge": True},
            "uavdt": {"images_dir": str(images_dir), "gt_dir": str(gt_dir),
                      "test_sequences": seqs, "attributes": uavdt_attributes(tuple(seqs))},
        }
    }


def test_manifest_includes_mot_sets_and_roles(tmp_path: Path):
    assert SET_ROLES["MOT-tune"] == "tune"
    assert SET_ROLES["T-MOT"] == "test" and SET_ROLES["T-UAVDT"] == "test"
    report: dict = {}
    images, boxes = build_manifest(_mot_cfg(tmp_path), set(), report)
    counts = images.groupby("set_name").agg(n=("image_id", "size"), s=("scene_id", "nunique"))
    assert counts.loc["MOT-tune"].tolist() == [2, 1]
    assert counts.loc["T-MOT"].tolist() == [4, 2]
    assert counts.loc["T-UAVDT"].tolist() == [4, 2]
    # ignore regions of MOT frames live in the shared box table
    ign = boxes[boxes.is_ignore].merge(images, on="image_id")
    assert (ign.set_name == "T-UAVDT").sum() == 8 and (ign.set_name == "MOT-tune").sum() == 2

    mot_gt, attrs = report["mot_gt"], report["seq_attrs"]
    assert set(mot_gt.image_id) <= set(images.image_id)
    assert set(mot_gt.seq_id) == {SEQ, "uav0000099_00000_v", "uav0000098_00000_v",
                                  "M0101", "M0202"}
    assert sorted(attrs.seq_id) == ["M0101", "M0202"]

    h = save_manifest(images, boxes, tmp_path / "m", mot_gt, attrs)
    gt2, attrs2 = load_mot(tmp_path / "m")
    pd.testing.assert_frame_equal(gt2, mot_gt)
    pd.testing.assert_frame_equal(attrs2, attrs)
    assert load_manifest(tmp_path / "m")[2] == h
    h_changed = save_manifest(images, boxes, tmp_path / "m2", mot_gt.assign(track_id=7), attrs)
    assert h_changed != h  # track ids are part of the manifest content
    gt3, _ = load_mot(tmp_path / "m2")
    assert (gt3.track_id == 7).all()


def test_exclusions_shrink_mot_tune_gt_but_never_tests(tmp_path: Path):
    cfg = _mot_cfg(tmp_path)
    drop = f"visdrone_mot/val/{SEQ}/0000001"
    keep_test = "uavdt/M0101/000001"
    report: dict = {}
    images, _ = build_manifest(cfg, {drop, keep_test}, report)
    assert drop not in set(images.image_id) and drop not in set(report["mot_gt"].image_id)
    assert keep_test in set(report["mot_gt"].image_id)


def _with_drops(cfg: dict, vm: list | None = None, ua: list | None = None) -> dict:
    src = cfg["sources"]
    return {"sources": {"visdrone_mot": {**src["visdrone_mot"], "drop_sequences": vm or []},
                        "uavdt": {**src["uavdt"], "drop_sequences": ua or []}}}


def test_drop_sequences_remove_whole_sequences_and_change_the_hash(tmp_path: Path):
    cfg = _mot_cfg(tmp_path)
    report: dict = {}
    images0, boxes0 = build_manifest(cfg, set(), report)
    h0 = save_manifest(images0, boxes0, tmp_path / "m0", report["mot_gt"], report["seq_attrs"],
                       report["seq_drops"])
    assert report["seq_drops"].empty
    assert h0 == manifest_hash(images0, boxes0, report["mot_gt"], report["seq_attrs"])

    gone_mot, gone_uav = "uav0000099_00000_v", "M0202"
    cfg = _with_drops(cfg, vm=[{"seq": gone_mot, "reason": "near-duplicate of a train scene"}],
                      ua=[{"seq": gone_uav, "reason": "same place as a training clip"}])
    report = {}
    images, boxes = build_manifest(cfg, set(), report)
    assert not images.scene_id.isin([f"visdrone_mot/{gone_mot}", f"uavdt/{gone_uav}"]).any()
    assert set(boxes.image_id) <= set(images.image_id)
    assert (images.set_name == "T-MOT").sum() == 2 and (images.set_name == "T-UAVDT").sum() == 2
    mot_gt, attrs, drops = report["mot_gt"], report["seq_attrs"], report["seq_drops"]
    assert not mot_gt.seq_id.isin([gone_mot, gone_uav]).any() and len(mot_gt)
    assert attrs.seq_id.tolist() == ["M0101"]
    assert drops.to_dict("records") == [
        {"source": "visdrone_mot", "seq_id": gone_mot, "set_name": "T-MOT", "n_images": 2,
         "reason": "near-duplicate of a train scene"},
        {"source": "uavdt", "seq_id": gone_uav, "set_name": "T-UAVDT", "n_images": 2,
         "reason": "same place as a training clip"}]
    h = save_manifest(images, boxes, tmp_path / "m", mot_gt, attrs, drops)
    assert h != h0 and load_manifest(tmp_path / "m")[2] == h
    assert h != manifest_hash(images, boxes, mot_gt, attrs)  # the drop record itself is hashed
    other = drops.assign(reason="another reason")
    assert manifest_hash(images, boxes, mot_gt, attrs, other) != h


def test_manifest_build_cli_saves_and_prints_the_drops(tmp_path: Path):
    import yaml
    from typer.testing import CliRunner

    from cvt.cli import app

    cfg = _with_drops(_mot_cfg(tmp_path), ua=[{"seq": "M0202", "reason": "leakage decision"}])
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "m"
    r = CliRunner().invoke(app, ["manifest", "build", "--datasets", str(datasets), "--out",
                                 str(out), "--exclusions", str(tmp_path / "none.txt")])
    assert r.exit_code == 0, r.output
    assert "dropped sequences" in r.output and "M0202" in r.output and "leakage decision" in r.output
    images, _, h = load_manifest(out)
    assert not (images.scene_id == "uavdt/M0202").any()
    assert h == (out / "HASH").read_text(encoding="utf-8") and h in r.output


def test_drop_sequences_config_errors(tmp_path: Path):
    cfg = _mot_cfg(tmp_path)
    with pytest.raises(ManifestError, match=r"uavdt\.drop_sequences: .*M0999.*not found"):
        build_manifest(_with_drops(cfg, ua=[{"seq": "M0999", "reason": "x"}]), set())
    with pytest.raises(ManifestError, match=r"visdrone_mot\.drop_sequences\[0\]"):
        build_manifest(_with_drops(cfg, vm=[{"seq": SEQ}]), set())  # no reason
    with pytest.raises(ManifestError, match="twice"):
        build_manifest(_with_drops(cfg, ua=[{"seq": "M0101", "reason": "a"},
                                            {"seq": "M0101", "reason": "b"}]), set())


def test_load_mot_without_mot_sets_is_empty_with_columns(tmp_path: Path):
    make_visdrone(tmp_path / "vd" / "train", n_images=2, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=1, seed0=100)
    cfg = {"sources": {"visdrone_det": {
        "train_dir": str(tmp_path / "vd" / "train"), "val_dir": str(tmp_path / "vd" / "val"),
        "rider_merge": True, "tune_frac": 0.5}}}
    report: dict = {}
    images, boxes = build_manifest(cfg, set(), report)
    assert report["mot_gt"].empty and report["seq_attrs"].empty
    h = save_manifest(images, boxes, tmp_path / "m")
    assert h == manifest_hash(images, boxes)  # no MOT tables: the hash is the DET-only hash
    gt, attrs = load_mot(tmp_path / "m")
    assert list(gt.columns) == MOT_GT_COLS and gt.empty
    assert list(attrs.columns) == SEQ_ATTR_COLS and attrs.empty
    # an older manifest directory with no MOT files at all also loads as empty
    (tmp_path / "m" / "mot_gt.parquet").unlink()
    (tmp_path / "m" / "seq_attrs.parquet").unlink()
    gt, attrs = load_mot(tmp_path / "m")
    assert gt.empty and attrs.empty


def test_mot_source_config_errors(tmp_path: Path):
    cfg = _mot_cfg(tmp_path)
    bad = {**cfg["sources"]["visdrone_mot"], "rider_merge": "yes"}
    with pytest.raises(ManifestError, match="rider_merge"):
        build_manifest({"sources": {"visdrone_mot": bad}}, set())
    bad = {**cfg["sources"]["visdrone_mot"], "val_dir": str(tmp_path / "nope")}
    with pytest.raises(ManifestError, match="VisDrone-MOT val"):
        build_manifest({"sources": {"visdrone_mot": bad}}, set())
    for key in ("test_sequences", "attributes"):
        bad = {k: v for k, v in cfg["sources"]["uavdt"].items() if k != key}
        with pytest.raises(ManifestError, match=rf"uavdt\.{key}"):
            build_manifest({"sources": {"uavdt": bad}}, set())
    bad = {**cfg["sources"]["uavdt"], "attr_dir": "data/external/uavdt/M_attr"}
    with pytest.raises(ManifestError, match=r"uavdt\.attr_dir"):  # removed key: fail loudly
        build_manifest({"sources": {"uavdt": bad}}, set())


def test_det_manifest_hash_unchanged(tmp_path: Path):
    """Hash of the DET+UIT fixture manifest, recorded before the adapter refactor."""
    make_visdrone(tmp_path / "vd" / "train", n_images=6, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=2, seed0=100)
    make_uitdrone7(tmp_path / "uit")
    cfg = {"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd" / "train"),
                         "val_dir": str(tmp_path / "vd" / "val"),
                         "rider_merge": True, "tune_frac": 0.5},
        "uitdrone7": {"images_dir": str(tmp_path / "uit" / "images"),
                      "labels_dir": str(tmp_path / "uit" / "annotations")}}}
    images, boxes = build_manifest(cfg, set())
    assert manifest_hash(images, boxes) == "e35099f1686d847f"
