import pandas as pd
import pytest

from cvt.eval.coco_bridge import to_coco


def _frames():
    images = pd.DataFrame({"image_id": ["a", "b"], "width": [3840, 960], "height": [2160, 540]})
    gt = pd.DataFrame(
        [("a", 0, 0, 20, 10, 0, False), ("b", 10, 10, 20, 20, 1, False),
         ("b", 200, 200, 300, 300, -1, True)],
        columns=["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"],
    )
    dt = pd.DataFrame(
        [("a", 0, 0, 20, 10, 0.9, 0), ("b", 10, 10, 20, 20, 0.8, 1), ("b", 1, 1, 2, 2, 0.1, 4)],
        columns=["image_id", "x1", "y1", "x2", "y2", "score", "label"],
    )
    return images, gt, dt


def test_boxes_rescaled_to_1920_width():
    images, gt, dt = _frames()
    cgt, cdt, ids = to_coco(images, gt, dt, class_ids=[0, 1])
    assert ids == [1, 2]
    a = cgt.loadAnns(cgt.getAnnIds(imgIds=[1]))[0]
    assert a["bbox"] == pytest.approx([0, 0, 10, 5])  # 3840 -> 1920 halves coordinates
    b = [x for x in cgt.loadAnns(cgt.getAnnIds(imgIds=[2])) if not x["iscrowd"]][0]
    assert b["bbox"] == pytest.approx([20, 20, 20, 20])  # 960 -> 1920 doubles
    assert b["area"] == pytest.approx(400)


def test_ignore_region_replicated_per_class_as_crowd():
    images, gt, dt = _frames()
    cgt, _, _ = to_coco(images, gt, dt, class_ids=[0, 1])
    crowd = [x for x in cgt.dataset["annotations"] if x["iscrowd"]]
    assert sorted(x["category_id"] for x in crowd) == [1, 2]


def test_classes_outside_subset_are_dropped():
    images, gt, dt = _frames()
    _, cdt, _ = to_coco(images, gt, dt, class_ids=[0, 1])
    assert sorted(x["category_id"] for x in cdt.dataset["annotations"]) == [1, 2]


def test_empty_detections_are_allowed():
    images, gt, dt = _frames()
    _, cdt, _ = to_coco(images, gt, dt.iloc[0:0], class_ids=[0, 1])
    assert cdt.dataset["annotations"] == []


def test_detections_mostly_inside_ignore_regions_are_dropped():
    images, gt, dt = _frames()
    dt = pd.concat([dt, pd.DataFrame(
        [("b", 210, 210, 230, 230, 0.7, 0), ("b", 290, 290, 330, 330, 0.6, 0)],
        columns=dt.columns)], ignore_index=True)
    _, cdt, _ = to_coco(images, gt, dt, class_ids=[0, 1])
    scores = sorted(a["score"] for a in cdt.dataset["annotations"])
    assert 0.7 not in scores and 0.6 in scores  # fully inside dropped, 6% inside kept


def test_unknown_ids_bad_labels_and_duplicate_images_raise():
    images, gt, dt = _frames()
    with pytest.raises(ValueError, match="unknown images"):
        to_coco(images, gt, dt.assign(image_id="zzz"), class_ids=[0, 1])
    with pytest.raises(ValueError, match="labels outside"):
        to_coco(images, gt, dt.assign(label=6), class_ids=[0, 1])
    with pytest.raises(ValueError, match="duplicate image ids"):
        to_coco(pd.concat([images, images.iloc[:1]]), gt, dt, class_ids=[0, 1])


def test_gt_objects_mostly_inside_ignore_regions_are_dropped():
    images, gt, dt = _frames()
    gt = pd.concat([gt, pd.DataFrame([("b", 220, 220, 240, 240, 1, False)], columns=gt.columns)],
                   ignore_index=True)
    cgt, _, _ = to_coco(images, gt, dt, class_ids=[0, 1])
    objs = [a for a in cgt.loadAnns(cgt.getAnnIds(imgIds=[2])) if not a["iscrowd"]]
    assert len(objs) == 1  # only the original (10,10,20,20) object of image b remains
