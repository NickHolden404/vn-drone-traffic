from pathlib import Path

import numpy as np
import pytest

from cvt.data.adapters.uitdrone7 import load_uitdrone7, scene_of
from cvt.data.schema import CLASS_ID
from tests.helpers import make_uitdrone7


@pytest.mark.parametrize(
    "stem,scene",
    [
        ("DJI_0063_DJI_0063_0446", "DJI_0063_DJI_0063"),
        ("345_60s_104s_345", "345_60s_104s"),
        ("AIUit000004_12", "AIUit000004"),
    ],
)
def test_scene_of(stem, scene):
    assert scene_of(stem) == scene


def test_scene_of_rejects_unnumbered_stem():
    with pytest.raises(ValueError):
        scene_of("no_frame_number_here")


def test_load_converts_yolo_to_pixels_and_maps_classes(tmp_path: Path):
    root = make_uitdrone7(tmp_path / "uit")
    frames = load_uitdrone7(root / "images", root / "annotations", "train")
    assert len(frames) == 3
    f = frames[0]
    assert f.image_id == "uitdrone7/train/CLIPtrain_1"
    assert f.scene_id == "uitdrone7/CLIPtrain"
    assert (f.width, f.height) == (64, 48)
    np.testing.assert_allclose(f.boxes[0], [24, 12, 40, 36], atol=1e-4)
    np.testing.assert_array_equal(
        f.labels, [CLASS_ID["motorbike"], CLASS_ID["car"], CLASS_ID["pedestrian"]]
    )


def test_images_without_labels_are_skipped(tmp_path: Path):
    root = make_uitdrone7(tmp_path / "uit")
    frames = load_uitdrone7(root / "images", root / "annotations", "test")
    assert [f.image_id.rsplit("/", 1)[1] for f in frames] == ["CLIPtest_1", "CLIPtest_2", "CLIPtest_3"]


def test_unknown_raw_class_raises(tmp_path: Path):
    root = make_uitdrone7(tmp_path / "uit", n_train=1, n_test=0)
    (root / "annotations" / "train" / "CLIPtrain_1.txt").write_text("9 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="unknown UIT-Drone7 class"):
        load_uitdrone7(root / "images", root / "annotations", "train")


def test_empty_label_file_gives_negative_frame(tmp_path: Path):
    root = make_uitdrone7(tmp_path / "uit", n_train=1, n_test=0)
    (root / "annotations" / "train" / "CLIPtrain_1.txt").write_text("")
    (f,) = load_uitdrone7(root / "images", root / "annotations", "train")
    assert f.boxes.shape == (0, 4)


def _one_label(tmp_path: Path, text: str) -> Path:
    root = make_uitdrone7(tmp_path / "uit", n_train=1, n_test=0)
    (root / "annotations" / "train" / "CLIPtrain_1.txt").write_text(text)
    return root


def test_duplicate_rows_are_dropped(tmp_path: Path):
    root = _one_label(tmp_path, "0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    (f,) = load_uitdrone7(root / "images", root / "annotations", "train")
    np.testing.assert_array_equal(f.labels, [CLASS_ID["pedestrian"], CLASS_ID["motorbike"]])


def test_malformed_lines_name_file_and_line(tmp_path: Path):
    root = _one_label(tmp_path, "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1\n")
    with pytest.raises(ValueError, match=r"CLIPtrain_1\.txt:2: expected 5 fields"):
        load_uitdrone7(root / "images", root / "annotations", "train")
    (root / "annotations" / "train" / "CLIPtrain_1.txt").write_text("1.5 0.2 0.2 0.1 0.1\n")
    with pytest.raises(ValueError, match="non-integer class id"):
        load_uitdrone7(root / "images", root / "annotations", "train")


def test_border_boxes_clipped_and_labels_stay_aligned(tmp_path: Path):
    # first box lies fully right of the 64 px image (dropped), second crosses the border
    root = _one_label(tmp_path, "3 1.5 0.5 0.1 0.1\n2 0.99 0.5 0.1 0.1\n")
    (f,) = load_uitdrone7(root / "images", root / "annotations", "train")
    np.testing.assert_array_equal(f.labels, [CLASS_ID["car"]])
    assert f.boxes[0, 2] == 64


def test_label_without_image_raises(tmp_path: Path):
    root = make_uitdrone7(tmp_path / "uit", n_train=1, n_test=0)
    (root / "images" / "train" / "CLIPtrain_1.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        load_uitdrone7(root / "images", root / "annotations", "train")
