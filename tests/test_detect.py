import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt.data.schema import CLASS_ID
from cvt.detect.base import Detections
from cvt.detect.oracle import OracleDetector
from cvt.detect.ultralytics_det import build_class_map, map_and_merge, rider_ids
from cvt.eval.predictions import predict_frames

VISDRONE_NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van", 5: "truck",
                  6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"}


def test_class_map_drops_tricycles_and_folds_van():
    m = build_class_map(VISDRONE_NAMES, "visdrone")
    assert m[4] == CLASS_ID["car"] and m[6] == -1 and m[9] == CLASS_ID["motorbike"]


def test_class_map_rejects_unknown_names():
    with pytest.raises(KeyError, match="no entry"):
        build_class_map({0: "spaceship"}, "visdrone")


def test_map_and_merge_folds_predicted_riders():
    cmap = build_class_map(VISDRONE_NAMES, "visdrone")
    rider = rider_ids(VISDRONE_NAMES, "visdrone")
    xyxy = np.array([[10, 10, 30, 24], [14, 8, 22, 18], [50, 50, 60, 60]], np.float32)
    det = map_and_merge(xyxy, np.array([0.9, 0.8, 0.7]), np.array([9, 1, 6]), cmap, rider)
    np.testing.assert_array_equal(det.labels, [CLASS_ID["motorbike"]])  # rider merged, tricycle dropped
    np.testing.assert_allclose(det.boxes, [[10, 8, 30, 24]])
    np.testing.assert_allclose(det.scores, [0.9])


def test_low_confidence_vehicle_cannot_absorb_predicted_rider():
    cmap = build_class_map(VISDRONE_NAMES, "visdrone")
    rider = rider_ids(VISDRONE_NAMES, "visdrone")
    xyxy = np.array([[10, 10, 30, 24], [14, 8, 22, 18]], np.float32)
    det = map_and_merge(xyxy, np.array([0.05, 0.8]), np.array([9, 1]), cmap, rider)
    np.testing.assert_array_equal(det.labels, [CLASS_ID["motorbike"], CLASS_ID["pedestrian"]])
    np.testing.assert_allclose(det.scores, [0.05, 0.8])


def _manifest():
    images = pd.DataFrame({"image_id": ["a", "b"], "path": ["/x/a.jpg", "/x/b.jpg"]})
    boxes = pd.DataFrame(
        [("a", 0, 0, 5, 5, 0, False), ("a", 0, 0, 9, 9, -1, True)],
        columns=["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"],
    )
    return images, boxes


def test_oracle_returns_ground_truth_without_ignores():
    images, boxes = _manifest()
    dets = list(OracleDetector(images, boxes).predict_paths(["/x/a.jpg", "/x/b.jpg"]))
    assert len(dets[0].boxes) == 1 and len(dets[1].boxes) == 0


def test_predict_frames_caches(tmp_path: Path):
    images, boxes = _manifest()
    cache = tmp_path / "p.parquet"
    df = predict_frames(OracleDetector(images, boxes), images, cache)
    assert df.image_id.tolist() == ["a"] and cache.exists()

    class Boom:
        name, cache_tag = "boom", "boom"

        def predict_paths(self, paths):
            raise AssertionError("cache should have been used")

    pd.testing.assert_frame_equal(predict_frames(Boom(), images, cache), df)


def test_detections_empty():
    d = Detections.empty()
    assert d.boxes.shape == (0, 4) and d.scores.shape == (0,) and d.labels.shape == (0,)


class _T:
    def __init__(self, a):
        self.a = a

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, rows):
        self.data = _T(np.array(rows, np.float32).reshape(-1, 6))

    def __len__(self):
        return len(self.data.a)


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


def _install_fake_ultralytics(monkeypatch, calls, short=False):
    class FakeYOLO:
        names = VISDRONE_NAMES

        def __init__(self, weights):
            pass

        def predict(self, source, **kw):
            calls.append((len(source), kw))
            out = [_Result([[0, 0, 10, 10, 0.9, 3]] if im[0, 0, 0] else []) for im in source]
            return out[:-1] if short else out

    def imread(p):
        if "missing" in str(p):
            return None
        return np.full((8, 8, 3), 1 if "car" in str(p) else 0, np.uint8)

    ul = types.ModuleType("ultralytics")
    ul.YOLO = ul.RTDETR = FakeYOLO
    ul.__version__ = "fake"
    utils = types.ModuleType("ultralytics.utils")
    patches = types.ModuleType("ultralytics.utils.patches")
    patches.imread = imread
    utils.patches = patches
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.patches", patches)


def _detector(tmp_path, **kw):
    from cvt.detect.ultralytics_det import UltralyticsDetector

    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    args = dict(name="m", weights=str(w), arch="yolo", names_map="visdrone", imgsz=640,
                rider_merge=True, batch=2)
    args.update(kw)
    return UltralyticsDetector(**args)


def test_predict_paths_chunks_keeps_order_and_maps(tmp_path, monkeypatch):
    calls = []
    _install_fake_ultralytics(monkeypatch, calls)
    det = _detector(tmp_path)
    out = list(det.predict_paths(["a_car.jpg", "b.jpg", "c_car.jpg", "d.jpg", "e_car.jpg"]))
    assert [n for n, _ in calls] == [2, 2, 1]
    assert [len(d.labels) for d in out] == [1, 0, 1, 0, 1]
    assert out[0].labels.tolist() == [CLASS_ID["car"]]
    kw = calls[0][1]
    assert kw["quantize"] == 32 and "half" not in kw and "batch" not in kw


def test_predict_paths_raises_on_short_results_and_unreadable(tmp_path, monkeypatch):
    _install_fake_ultralytics(monkeypatch, [], short=True)
    with pytest.raises(RuntimeError, match="results for"):
        list(_detector(tmp_path).predict_paths(["a_car.jpg", "b.jpg"]))
    _install_fake_ultralytics(monkeypatch, [])
    with pytest.raises(OSError, match="unreadable"):
        list(_detector(tmp_path).predict_paths(["missing.jpg"]))


def test_cache_tag_tracks_prediction_config(tmp_path, monkeypatch):
    _install_fake_ultralytics(monkeypatch, [])
    base = _detector(tmp_path).cache_tag
    assert base == _detector(tmp_path).cache_tag
    assert base != _detector(tmp_path, rider_merge=False).cache_tag
    assert base != _detector(tmp_path, half=True).cache_tag
    assert base != _detector(tmp_path, iou=0.7).cache_tag


def test_rect_false_reaches_predict_and_default_tags_are_unchanged(tmp_path, monkeypatch):
    calls = []
    _install_fake_ultralytics(monkeypatch, calls)
    d = _detector(tmp_path)
    assert "rect" not in d.kw
    assert d.cache_tag == "2ea933eb507f"  # pinned before `rect` existed: caches stay valid
    assert _detector(tmp_path, half=True).cache_tag == "cb96cadeb84a"
    sq = _detector(tmp_path, rect=False)
    sq.predict_arrays([np.zeros((8, 8, 3), np.uint8)])
    assert calls[-1][1]["rect"] is False and sq.cache_tag != d.cache_tag
    assert _detector(tmp_path, rect=None).cache_tag == d.cache_tag


def test_predict_frames_short_detector_raises_and_writes_no_cache(tmp_path):
    images, boxes = _manifest()

    class Short:
        name, cache_tag = "short", "short"

        def predict_paths(self, paths):
            yield Detections.empty()

    cache = tmp_path / "p.parquet"
    with pytest.raises(ValueError):
        predict_frames(Short(), images, cache)
    assert not cache.exists() and not list(tmp_path.glob("*.tmp"))


def test_visdrone_others_class_is_dropped():
    names = {**VISDRONE_NAMES, 10: "others"}  # real dronefreak checkpoints have this 11th class
    assert build_class_map(names, "visdrone")[10] == -1


def test_cache_tag_tracks_rider_rule(tmp_path, monkeypatch):
    import cvt.data.riders as riders

    _install_fake_ultralytics(monkeypatch, [])
    base = _detector(tmp_path).cache_tag
    defaults = riders.merge_riders.__defaults__
    i = riders.merge_riders.__code__.co_varnames.index("feet_min_inside") - (
        riders.merge_riders.__code__.co_argcount - len(defaults)
    )
    monkeypatch.setattr(
        riders.merge_riders, "__defaults__", defaults[:i] + (0.3,) + defaults[i + 1 :]
    )
    assert _detector(tmp_path).cache_tag != base


def test_registry_composites_and_missing_weights(tmp_path, monkeypatch):
    import cvt.detect.ultralytics_det as ud
    from cvt.detect.composite import FusedDetector, TiledDetector
    from cvt.detect.registry import build_detector

    class FakeUD:
        def __init__(self, **kw):
            self.name, self.cache_tag, self.kw = kw["name"], "ud-" + kw["name"], kw

    monkeypatch.setattr(ud, "UltralyticsDetector", FakeUD)
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    cfg = {
        "t1": {"arch": "yolo", "weights": str(w), "names_map": "cvt"},
        "t2": {"arch": "rtdetr", "weights": str(w), "names_map": "cvt"},
        "t1-tiled": {"arch": "tiled", "base": "t1", "tile": 640, "overlap": 0.2},
        "t2-tiled": {"arch": "tiled", "base": "t2"},
        "fused": {"arch": "fused", "members": ["t1-tiled", "t2-tiled"], "iou": 0.55},
        "missing": {"arch": "yolo", "weights": str(tmp_path / "nope.pt"), "names_map": "cvt"},
    }
    f = build_detector("fused", cfg, 1280, 0.01, None, None)
    assert isinstance(f, FusedDetector) and all(isinstance(m, TiledDetector) for m in f.members)
    assert f.members[0].base.kw["arch"] == "yolo" and f.members[1].base.kw["arch"] == "rtdetr"
    with pytest.raises(FileNotFoundError, match="weights not found"):
        build_detector("missing", cfg, 1280, 0.01, None, None)


def _registry_cfg(tmp_path, monkeypatch):
    import cvt.detect.ultralytics_det as ud

    class FakeUD:
        def __init__(self, **kw):
            self.name, self.cache_tag, self.kw = kw["name"], "ud-" + kw["name"], kw

    monkeypatch.setattr(ud, "UltralyticsDetector", FakeUD)
    w = tmp_path / "w.pt"
    w.write_bytes(b"x")
    return {
        "t1": {"arch": "yolo", "weights": str(w), "names_map": "cvt"},
        "tiled": {"arch": "tiled", "base": "t1", "tile": 512, "tile_imgsz": "768",
                  "nms_iou": 0.5, "edge_px": 4, "max_det": 50},
        "tiled-default": {"arch": "tiled", "base": "t1"},
        "fused": {"arch": "fused", "members": ["tiled"], "max_det": 77},
        "tiled-fused": {"arch": "tiled", "base": "fused"},
        "tiled-oracle": {"arch": "tiled", "base": "oracle"},
        "oracle": {"arch": "oracle"},
        "cyc-a": {"arch": "fused", "members": ["cyc-b"]},
        "cyc-b": {"arch": "fused", "members": ["cyc-a"]},
        "cyc-f": {"arch": "fused", "members": ["cyc-f"]},
    }


def test_registry_passes_composite_options_through(tmp_path, monkeypatch):
    from cvt.detect.registry import build_detector

    cfg = _registry_cfg(tmp_path, monkeypatch)
    t = build_detector("tiled", cfg, 1280, 0.01, None, None)
    assert (t.tile, t.tile_imgsz, t.nms_iou, t.edge_px, t.max_det) == (512, 768, 0.5, 4, 50)
    assert isinstance(t.tile_imgsz, int)
    d = build_detector("tiled-default", cfg, 1280, 0.01, None, None)
    assert (d.edge_px, d.max_det, d.tile_imgsz) == (8, 1000, 640)
    assert build_detector("fused", cfg, 1280, 0.01, None, None).max_det == 77


def test_registry_rejects_bad_tiled_base_and_cycles(tmp_path, monkeypatch):
    from cvt.detect.registry import build_detector

    cfg = _registry_cfg(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must be a yolo/rtdetr model"):
        build_detector("tiled-fused", cfg, 1280, 0.01, None, None)
    with pytest.raises(ValueError, match="must be a yolo/rtdetr model"):
        build_detector("tiled-oracle", cfg, 1280, 0.01, None, None)
    with pytest.raises(ValueError, match="cyclic model reference"):
        build_detector("cyc-a", cfg, 1280, 0.01, None, None)
    with pytest.raises(ValueError, match="cyclic model reference"):
        build_detector("cyc-f", cfg, 1280, 0.01, None, None)


def test_cvt_names_map_is_identity():
    from cvt.data.schema import CLASSES

    names = dict(enumerate(CLASSES))
    assert build_class_map(names, "cvt").tolist() == list(range(len(CLASSES)))


def test_open_detector_missing_weights_fails_before_hashing(tmp_path):
    from cvt.eval.run import open_detector

    cfg = {"m": {"arch": "yolo", "weights": str(tmp_path / "nope.pt"), "names_map": "cvt"}}
    with pytest.raises(FileNotFoundError, match="m: weights not found"):
        open_detector("m", cfg, 1280, 0.01, None, None)
