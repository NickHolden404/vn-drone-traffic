import numpy as np
import pytest

from cvt.data.classmaps import IGNORE, NAME_MAPS, UITDRONE7_RAW, VISDRONE_RAW
from cvt.data.schema import CLASS_ID, CLASSES, FrameAnn, clip_boxes


def test_class_order_is_fixed():
    assert CLASSES == ("motorbike", "car", "bus", "truck", "bicycle", "pedestrian")
    assert CLASS_ID["pedestrian"] == 5


def test_frameann_validates_lengths_and_range():
    with pytest.raises(ValueError, match="boxes vs"):
        FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[0, 0, 1, 1]], labels=[])
    with pytest.raises(ValueError, match="out of range"):
        FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[0, 0, 1, 1]], labels=[6])
    f = FrameAnn("i", "p", 10, 10, "s", "sc", "train")
    assert f.boxes.shape == (0, 4) and f.labels.shape == (0,) and f.ignore.shape == (0, 4)


def test_clip_boxes_clips_and_drops_degenerate():
    boxes = np.array([[-5, -5, 5, 5], [8, 8, 30, 30], [3, 3, 3.5, 9]], np.float32)
    out, keep = clip_boxes(boxes, width=10, height=10)
    np.testing.assert_array_equal(keep, [True, True, False])
    np.testing.assert_allclose(out, [[0, 0, 5, 5], [8, 8, 10, 10]])


def test_uit_map_covers_raw_ids_0_to_6():
    assert sorted(UITDRONE7_RAW) == list(range(7))
    assert UITDRONE7_RAW[1] == CLASS_ID["motorbike"]
    assert UITDRONE7_RAW[4] == CLASS_ID["car"]  # van folds into car


def test_visdrone_map_ignores_tricycles_and_regions():
    for raw in (0, 7, 8, 11):
        assert VISDRONE_RAW[raw] == IGNORE
    assert VISDRONE_RAW[10] == CLASS_ID["motorbike"]


def test_name_map_targets_are_valid_classes():
    for mapping in NAME_MAPS.values():
        for target in mapping.values():
            assert target is None or target in CLASS_ID


def test_frameann_rejects_bad_shapes_values_and_sizes():
    with pytest.raises(ValueError, match="shape"):
        FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=np.zeros((2, 6)), labels=[0, 0, 0])
    with pytest.raises(ValueError, match="integers"):
        FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[0, 0, 1, 1]], labels=[0.7])
    with pytest.raises(ValueError, match="finite"):
        FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[0, 0, np.nan, 1]], labels=[0])
    with pytest.raises(ValueError, match="finite"):
        FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[5, 0, 1, 1]], labels=[0])
    with pytest.raises(ValueError, match="positive"):
        FrameAnn("i", "p", 0, 10, "s", "sc", "train")


def test_frameann_equality_is_identity():
    a = FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[0, 0, 1, 1]], labels=[0])
    b = FrameAnn("i", "p", 10, 10, "s", "sc", "train", boxes=[[0, 0, 1, 1]], labels=[0])
    assert a == a and a != b
