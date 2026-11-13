import numpy as np
import pandas as pd
import pytest

from cvt.detect.base import Detections
from cvt.detect.composite import (
    FusedDetector,
    TiledDetector,
    fuse_predictions,
    shift_and_trim,
    tile_grid,
)


def test_tile_grid_covers_image_with_overlap():
    g = tile_grid(1280, 720, 640, 0.2)
    xs = sorted({t[0] for t in g})
    ys = sorted({t[1] for t in g})
    assert xs == [0, 512, 640] and ys == [0, 80] and len(g) == 6
    assert all(x1 - x0 == 640 and y1 - y0 == 640 for x0, y0, x1, y1 in g)
    assert tile_grid(500, 400, 640, 0.2) == [(0, 0, 500, 400)]


@pytest.mark.parametrize("tile,overlap", [(0, 0.2), (-5, 0.2), (640, -0.1), (640, 1.0)])
def test_tile_grid_validates_arguments(tile, overlap):
    with pytest.raises(ValueError):
        tile_grid(1280, 720, tile, overlap)


def test_shift_and_trim_drops_boxes_on_inner_edges():
    d = Detections(np.array([[0, 100, 10, 110], [300, 300, 320, 320]], np.float32),
                   np.array([0.9, 0.8], np.float32), np.array([0, 0]))
    out = shift_and_trim(d, 512, 0, 1152, 640, width=1280, height=720, edge_px=2)
    np.testing.assert_allclose(out.boxes, [[812, 300, 832, 320]])


def test_shift_and_trim_keeps_box_on_image_border_of_last_tile():
    d = Detections(np.array([[620, 100, 640, 120]], np.float32),
                   np.array([0.9], np.float32), np.array([0]))
    out = shift_and_trim(d, 640, 0, 1280, 640, width=1280, height=720, edge_px=2)
    np.testing.assert_allclose(out.boxes, [[1260, 100, 1280, 120]])


def test_shift_and_trim_cuts_box_touching_top_inner_edge():
    d = Detections(np.array([[100, 0, 120, 20], [100, 30, 120, 50]], np.float32),
                   np.array([0.9, 0.8], np.float32), np.array([0, 0]))
    out = shift_and_trim(d, 0, 80, 640, 720, width=1280, height=720, edge_px=2)
    np.testing.assert_allclose(out.boxes, [[100, 110, 120, 130]])


class FakeBase:
    cache_tag = "fake"

    def __init__(self):
        self.calls = []

    def predict_arrays(self, ims, imgsz=None):
        self.calls.append((len(ims), imgsz))
        out = []
        for im in ims:
            h, w = im.shape[:2]
            out.append(Detections(np.array([[w / 2 - 5, h / 2 - 5, w / 2 + 5, h / 2 + 5]], np.float32),
                                  np.array([0.9], np.float32), np.array([0])))
        return out


def test_tiled_detector_runs_full_frame_and_tiles():
    base = FakeBase()
    det = TiledDetector("t", base, tile=640, overlap=0.2,
                        reader=lambda p: np.zeros((720, 1280, 3), np.uint8))
    (out,) = list(det.predict_paths(["a.jpg"]))
    assert base.calls == [(1, None), (6, 640)]
    assert len(out.boxes) == 7  # full-frame centre + 6 distinct tile centres
    assert det.cache_tag != TiledDetector("t", base, tile=512, reader=lambda p: None).cache_tag


def test_tiled_detector_predict_paths_golden():
    """Output pinned from the implementation before `predict_arrays` was factored out."""
    base = FakeBase()
    det = TiledDetector("t", base, tile=640, overlap=0.2,
                        reader=lambda p: np.zeros((720, 1280, 3), np.uint8))
    (out,) = list(det.predict_paths(["a.jpg"]))
    assert out.boxes.tolist() == [
        [635.0, 355.0, 645.0, 365.0], [315.0, 315.0, 325.0, 325.0],
        [827.0, 315.0, 837.0, 325.0], [955.0, 315.0, 965.0, 325.0],
        [315.0, 395.0, 325.0, 405.0], [827.0, 395.0, 837.0, 405.0],
        [955.0, 395.0, 965.0, 405.0]]
    assert out.scores.tolist() == [pytest.approx(0.9)] * 7 and out.labels.tolist() == [0] * 7
    assert out.boxes.dtype == np.float32 and out.labels.dtype == np.int64
    (arr,) = det.predict_arrays([np.zeros((720, 1280, 3), np.uint8)])
    assert np.array_equal(arr.boxes, out.boxes)


def test_tiled_detector_default_edge_px_and_cache_tag():
    base = FakeBase()
    det = TiledDetector("t", base)
    assert det.edge_px == 8 and det.max_det == 1000
    assert det.cache_tag != TiledDetector("t", base, max_det=5).cache_tag


def test_tiled_detector_merges_duplicate_across_passes():
    abs_box = np.array([[100, 100, 140, 140]], np.float32)

    class Dup:
        cache_tag = "dup"

        def predict_arrays(self, ims, imgsz=None):
            s, lab = np.array([0.9], np.float32), np.array([0])
            if imgsz is None:
                return [Detections(abs_box, s, lab)]
            out = [Detections.empty() for _ in ims]
            out[0] = Detections(abs_box.copy(), s, lab)  # tile 0 has origin (0, 0)
            return out

    det = TiledDetector("t", Dup(), reader=lambda p: np.zeros((720, 1280, 3), np.uint8))
    (out,) = list(det.predict_paths(["a.jpg"]))
    np.testing.assert_allclose(out.boxes, abs_box)


def test_tiled_detector_max_det_caps_output():
    det = TiledDetector("t", FakeBase(), max_det=3,
                        reader=lambda p: np.zeros((720, 1280, 3), np.uint8))
    (out,) = list(det.predict_paths(["a.jpg"]))
    assert len(out.boxes) == 3


def _df(rows):
    return pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score", "label"])


def test_fuse_predictions_per_image():
    images = pd.DataFrame({"image_id": ["i1", "i2"]})
    a = _df([("i1", 0, 0, 10, 10, 0.8, 0)])
    b = _df([("i1", 0, 0, 10, 10, 0.6, 0), ("i2", 5, 5, 9, 9, 0.5, 1)])
    f = fuse_predictions([a, b], images, iou_thr=0.55, skip_thr=0.0)
    assert sorted(f.image_id) == ["i1", "i2"]
    assert f[f.image_id == "i1"].score.iloc[0] == pytest.approx(0.7)
    assert f[f.image_id == "i2"].score.iloc[0] == pytest.approx(0.25)


def test_fused_detector_streams_members():
    class M:
        def __init__(self, tag, score):
            self.name, self.cache_tag, self.score = tag, tag, score

        def predict_paths(self, paths):
            for _ in paths:
                yield Detections(np.array([[0, 0, 10, 10]], np.float32),
                                 np.array([self.score], np.float32), np.array([0]))

    f = FusedDetector("f", [M("a", 0.8), M("b", 0.4)])
    outs = list(f.predict_with_members(["x", "y"]))
    assert len(outs) == 2 and outs[0][0].scores[0] == pytest.approx(0.6)
    assert len(outs[0][1]) == 2


def test_fuse_predictions_max_det_keeps_best_per_image():
    images = pd.DataFrame({"image_id": ["i1", "i2"]})
    a = _df([("i1", 0, 0, 10, 10, 0.8, 0), ("i1", 50, 50, 60, 60, 0.9, 0),
             ("i2", 0, 0, 10, 10, 0.3, 0), ("i2", 50, 50, 60, 60, 0.5, 0)])
    f = fuse_predictions([a], images, iou_thr=0.55, skip_thr=0.0, max_det=1)
    assert sorted(f.image_id) == ["i1", "i2"]
    assert f.set_index("image_id").score.to_dict() == pytest.approx({"i1": 0.9, "i2": 0.5})


def test_fused_detector_max_det_in_cache_tag():
    class M:
        name, cache_tag = "m", "m"

    assert FusedDetector("f", [M()]).cache_tag != FusedDetector("f", [M()], max_det=5).cache_tag
