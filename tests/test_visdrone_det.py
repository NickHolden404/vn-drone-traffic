from pathlib import Path

import numpy as np
import pytest

from cvt.data.adapters.visdrone_det import load_visdrone_det
from cvt.data.schema import CLASS_ID
from tests.helpers import make_visdrone, write_noise_jpg


def test_rider_merged_and_ignores_collected(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    (f,) = load_visdrone_det(root, "train", rider_merge=True)
    assert f.image_id == "visdrone/train/0000007_00001_d_0000001"
    assert f.scene_id == "visdrone/0000007"
    np.testing.assert_array_equal(f.labels, [CLASS_ID["motorbike"], CLASS_ID["pedestrian"]])
    np.testing.assert_allclose(f.boxes[0], [10, 8, 30, 24])  # motor grown to include rider
    np.testing.assert_allclose(f.ignore, [[50, 30, 60, 40], [0, 40, 10, 48]])


def test_without_rider_merge_people_become_pedestrians(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    (f,) = load_visdrone_det(root, "train", rider_merge=False)
    assert (f.labels == CLASS_ID["pedestrian"]).sum() == 2
    np.testing.assert_allclose(f.boxes[0], [10, 10, 30, 24])


def test_zero_size_boxes_are_dropped(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    ann = next((root / "annotations").glob("*.txt"))
    ann.write_text("5,5,0,10,1,4,0,0\n5,5,10,10,1,4,0,0\n")
    (f,) = load_visdrone_det(root, "val", rider_merge=True)
    assert len(f.boxes) == 1
    np.testing.assert_array_equal(f.labels, [CLASS_ID["car"]])


def _write(root: Path, text: str) -> None:
    next((root / "annotations").glob("*.txt")).write_text(text)


def test_score_zero_with_normal_category_is_ignore_region(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    _write(root, "5,5,10,10,0,4,0,0\n20,20,10,10,1,4,0,0\n")
    (f,) = load_visdrone_det(root, "val", rider_merge=True)
    np.testing.assert_allclose(f.ignore, [[5, 5, 15, 15]])
    np.testing.assert_array_equal(f.labels, [CLASS_ID["car"]])


def test_people_on_tricycle_join_its_ignore_region(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    _write(root, "50,30,10,10,1,7,0,0\n52,28,6,8,1,2,0,0\n")
    (f,) = load_visdrone_det(root, "val", rider_merge=True)
    assert len(f.labels) == 0
    np.testing.assert_allclose(f.ignore, [[50, 28, 60, 40]])


def test_duplicate_rows_dropped_and_bad_rows_raise(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    _write(root, "20,20,10,10,1,4,0,0\n20,20,10,10,1,4,0,0\n")
    (f,) = load_visdrone_det(root, "val", rider_merge=True)
    assert len(f.boxes) == 1
    _write(root, "20,20,10\n")
    with pytest.raises(ValueError, match=r":1: expected >= 6"):
        load_visdrone_det(root, "val", rider_merge=True)
    _write(root, "20,20,10,10,1,42,0,0\n")
    with pytest.raises(ValueError, match="unknown VisDrone categories"):
        load_visdrone_det(root, "val", rider_merge=True)


def test_annotation_without_image_raises(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    next((root / "images").glob("*.jpg")).unlink()
    with pytest.raises(FileNotFoundError):
        load_visdrone_det(root, "val", rider_merge=True)


def test_frames_of_one_sequence_share_a_scene(tmp_path: Path):
    root = make_visdrone(tmp_path / "vd", n_images=1, seed0=7)
    stem = "0000007_00099_d_0000002"
    write_noise_jpg(root / "images" / f"{stem}.jpg", 99)
    (root / "annotations" / f"{stem}.txt").write_text("20,20,10,10,1,4,0,0\n")
    frames = load_visdrone_det(root, "train", rider_merge=True)
    assert {f.scene_id for f in frames} == {"visdrone/0000007"}
