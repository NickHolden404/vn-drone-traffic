"""Deployment grid, TensorRT export wrapper (mocked) and generated registry entries."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cvt.config import load_yaml
from cvt.data.manifest import BOX_COLS, IMAGE_COLS, save_manifest
from cvt.data.schema import CLASSES

STUDENTS = ["src_pseudo-yolo11n-s0", "src_pseudo-yolo11s-s0", "src_pseudo-yolo11m-s0"]


@pytest.fixture(autouse=True)
def _guard_calls(monkeypatch):
    """The real offline guard patches Ultralytics submodules; these tests fake `ultralytics`."""
    calls = []
    monkeypatch.setattr("cvt.deploy.engines._offline_guard", lambda: calls.append("guard"))
    return calls


def _cfg() -> dict:
    return load_yaml(Path(__file__).resolve().parents[1] / "configs" / "deploy.yaml")


def _students_cfg(tmp_path: Path) -> dict:
    out = {"oracle": {"arch": "oracle"}}
    for s in STUDENTS:
        w = tmp_path / "runs" / s / "weights" / "best.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        w.write_bytes(f"weights of {s}".encode())
        out[s] = {"arch": "yolo", "weights": w.as_posix(), "names_map": "cvt"}
    return out


def test_grid_has_36_points():
    from cvt.deploy.engines import Point, grid_points, registry_name

    cfg = _cfg()
    assert cfg["students"] == STUDENTS
    assert cfg["sizes"] == [640, 960, 1280, 1536] and cfg["precisions"] == ["fp16", "int8"]
    assert cfg["fp32_sizes"] == [640, 1280]
    assert cfg["tiled"] == {"full_imgsz": 1280, "tile": 640, "overlap": 0.2}
    assert cfg["calibration"] == {"list": "configs/train/src_pseudo.yaml", "n": 512, "seed": 0}
    assert cfg["bench"] == {"every": 30, "n": 199, "warmup": 20}
    pts = grid_points(cfg)
    assert len(pts) == 36
    engines = [p for p in pts if not p.tiled and p.precision != "fp32"]
    assert len(engines) == 24
    assert len([p for p in pts if p.precision == "fp32"]) == 6
    assert len([p for p in pts if p.tiled]) == 6
    assert len({registry_name(p) for p in pts}) == 36
    p = Point(STUDENTS[0], 960, "int8", False)
    assert p in pts and hash(p) == hash(Point(STUDENTS[0], 960, "int8", False))
    assert registry_name(p) == "src_pseudo-yolo11n-s0@960-int8"
    assert registry_name(Point(STUDENTS[2], 1280, "fp32", False)) == "src_pseudo-yolo11m-s0@1280-fp32"
    assert (registry_name(Point(STUDENTS[1], 1280, "fp16", True))
            == "src_pseudo-yolo11s-s0@1280+tiles640-fp16")
    with pytest.raises(ValueError, match="precision"):
        Point("m", 640, "fp8", False)


def test_engine_name_and_path():
    from cvt.deploy.engines import engine_name, engine_path

    assert engine_name("m", 640, "fp16") == "m@640-fp16"
    assert engine_path("m", 640, "int8") == Path("data/engines/m@640-int8.engine")


# --- registry: an entry with imgsz forces that size ------------------------------------------

def _fake_ud(monkeypatch):
    import cvt.detect.ultralytics_det as ud

    class FakeUD:
        def __init__(self, **kw):
            self.name, self.kw = kw["name"], kw
            self.cache_tag = f"ud-{kw['name']}-{kw['imgsz']}"

        def predict_arrays(self, ims, imgsz=None):
            raise AssertionError("not called")

    monkeypatch.setattr(ud, "UltralyticsDetector", FakeUD)


def test_engine_entry_forces_its_imgsz(tmp_path, monkeypatch):
    from cvt.detect.composite import TiledDetector
    from cvt.detect.registry import build_detector

    _fake_ud(monkeypatch)
    e640, e1280 = tmp_path / "a@640-fp16.engine", tmp_path / "a@1280-fp16.engine"
    e640.write_bytes(b"x")
    e1280.write_bytes(b"y")
    cfg = {
        "a@640-fp16": {"arch": "yolo", "weights": str(e640), "names_map": "cvt", "imgsz": 640},
        "a@1280-fp16": {"arch": "yolo", "weights": str(e1280), "names_map": "cvt", "imgsz": 1280},
        "a@1280+tiles640-fp16": {"arch": "tiled", "base": "a@1280-fp16", "tile_base": "a@640-fp16",
                                 "tile": 640, "overlap": 0.2},
        "bad-tiles": {"arch": "tiled", "base": "a@1280-fp16", "tile_base": "bad-tiles"},
    }
    with pytest.raises(ValueError, match="imgsz 640"):
        build_detector("a@640-fp16", cfg, 1280, 0.01, None, None)
    d = build_detector("a@640-fp16", cfg, 640, 0.01, None, None)
    assert d.kw["imgsz"] == 640 and "rect" not in d.kw  # no key: the detector default
    cfg["a@640-fp32"] = {"arch": "yolo", "weights": str(e640), "names_map": "cvt", "imgsz": 640,
                         "rect": False}
    assert build_detector("a@640-fp32", cfg, 640, 0.01, None, None).kw["rect"] is False
    t = build_detector("a@1280+tiles640-fp16", cfg, 1280, 0.01, None, None)
    assert isinstance(t, TiledDetector)
    assert t.base.kw["imgsz"] == 1280 and t.tile_base.kw["imgsz"] == 640 and t.tile_imgsz == 640
    with pytest.raises(ValueError, match="imgsz 1280"):
        build_detector("a@1280+tiles640-fp16", cfg, 960, 0.01, None, None)
    with pytest.raises(ValueError, match="must be a yolo/rtdetr model"):
        build_detector("bad-tiles", cfg, 1280, 0.01, None, None)


def test_tiled_cache_tag_unchanged_without_tile_base_and_uses_tile_base():
    from cvt.detect.base import Detections
    from cvt.detect.composite import TiledDetector

    class Base:
        def __init__(self, tag):
            self.cache_tag, self.calls = tag, []

        def predict_arrays(self, ims, imgsz=None):
            self.calls.append((len(ims), imgsz))
            return [Detections.empty() for _ in ims]

    full, tiles = Base("full"), Base("tiles")
    plain = TiledDetector("t", full, tile=32, overlap=0.0)
    assert plain.tile_base is full
    two = TiledDetector("t", full, tile=32, overlap=0.0, tile_base=tiles)
    assert two.cache_tag != plain.cache_tag
    assert TiledDetector("t", full, tile=32, overlap=0.0, tile_base=None).cache_tag == plain.cache_tag
    im = np.zeros((64, 64, 3), np.uint8)
    list(TiledDetector("t", full, tile=32, overlap=0.0, tile_base=tiles,
                       reader=lambda p: im).predict_paths(["x.jpg"]))
    assert full.calls == [(1, None)] and tiles.calls == [(4, 32)]


def _engine_file(path: Path, imgsz: int, batch: int = 1) -> Path:
    meta = json.dumps({"imgsz": [imgsz, imgsz], "batch": batch, "names": {"0": "car"}}).encode()
    path.write_bytes(len(meta).to_bytes(4, "little") + meta + b"\x00ENGINEBYTES" + path.name.encode())
    return path


def _install_fake_predict_ultralytics(monkeypatch, calls):
    class FakeYOLO:
        names = dict(enumerate(CLASSES))

        def __init__(self, weights, task=None):
            calls.append(("init", weights, task))

        def predict(self, source, **kw):
            calls.append(("predict", len(source), kw))
            return [types.SimpleNamespace(boxes=None) for _ in source]

    ul = types.ModuleType("ultralytics")
    ul.YOLO = ul.RTDETR = FakeYOLO
    ul.__version__ = "fake"
    monkeypatch.setitem(sys.modules, "ultralytics", ul)


def test_engine_detector_bakes_precision_and_checks_header(tmp_path, monkeypatch):
    from cvt.detect.ultralytics_det import UltralyticsDetector

    calls = []
    _install_fake_predict_ultralytics(monkeypatch, calls)

    def det(path, imgsz, **kw):
        return UltralyticsDetector(name="e", weights=str(path), arch="yolo", names_map="cvt",
                                   imgsz=imgsz, rider_merge=False, **kw)

    e = _engine_file(tmp_path / "e@640-int8.engine", 640)
    d = det(e, 640, batch=4)
    assert calls[0] == ("init", str(e), "detect")
    assert "quantize" not in d.kw and "half" not in d.kw and d.batch == 1
    d.predict_arrays([np.zeros((8, 8, 3), np.uint8)] * 3)
    assert [c[1] for c in calls if c[0] == "predict"] == [1, 1, 1]
    assert all("quantize" not in c[2] and c[2]["imgsz"] == 640 for c in calls if c[0] == "predict")
    with pytest.raises(ValueError, match="built for imgsz 640"):
        d.predict_arrays([np.zeros((8, 8, 3), np.uint8)], imgsz=1280)
    with pytest.raises(ValueError, match="built for imgsz 640"):
        det(e, 1280)
    assert det(_engine_file(tmp_path / "b2.engine", 640, batch=2), 640).batch == 2  # batched tiles
    headless = tmp_path / "raw.engine"
    headless.write_bytes(b"\xff\xff\xff\x7fnot an ultralytics engine")
    with pytest.raises(ValueError, match="metadata header"):
        det(headless, 640)
    # the tag pins the engine bytes: a rebuilt engine (same name and size) gets a new tag
    tag = d.cache_tag
    other = _engine_file(tmp_path / "other.engine", 640)
    assert det(other, 640).cache_tag != tag
    assert det(e, 640).cache_tag == tag


# --- calibration list: training side only ------------------------------------------------------

def _calib_setup(tmp_path: Path) -> dict:
    rows = []

    def add(iid, set_name, source="visdrone"):
        rows.append({"image_id": iid, "path": f"/raw/{iid}.jpg", "width": 64, "height": 48,
                     "source": source, "scene_id": iid, "split": "x", "set_name": set_name})

    for i in range(20):
        add(f"visdrone/train/a{i:03d}", "VD-train")
    for i in range(10):
        add(f"uitadrone/frames/v_{i:03d}", "P-train", "pool")
    add("visdrone/train/t000", "VD-tune")
    add("visdrone/val/z000", "T-VD")
    add("uitadrone/frames/w_000", "P-val", "pool")
    images = pd.DataFrame(rows)[IMAGE_COLS]
    boxes = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        BOX_COLS, ["str", "float64", "float64", "float64", "float64", "int64", "bool"],
        strict=True)})
    save_manifest(images, boxes, tmp_path / "manifest")
    yolo = tmp_path / "yolo"
    yolo.mkdir()

    def lst(name, ids):
        p = yolo / f"{name}.txt"
        p.write_text("".join(f"{(yolo / name / 'images' / (i.replace('/', '__') + '.jpg')).as_posix()}\n"
                             for i in ids), encoding="utf-8")
        return p.as_posix()

    vd = [r["image_id"] for r in rows if r["set_name"] == "VD-train"]
    pt = [r["image_id"] for r in rows if r["set_name"] == "P-train"]
    data = tmp_path / "src_pseudo.yaml"
    data.write_text(yaml.safe_dump({
        "train": [lst("VD-train", vd), lst("P-train-teacher-fused", pt)],
        "val": [lst("VD-tune", ["visdrone/train/t000"])],
        "names": dict(enumerate(CLASSES))}), encoding="utf-8")
    return {"data": data, "manifest": tmp_path / "manifest", "yolo": yolo, "lst": lst}


def test_calibration_list_is_training_side_only(tmp_path):
    from cvt.deploy.engines import calibration_list

    s = _calib_setup(tmp_path)
    cfg = {"calibration": {"list": str(s["data"]), "n": 12, "seed": 0}}
    a = calibration_list(cfg, s["manifest"])
    assert len(a) == 12 and len(set(a)) == 12
    assert a == calibration_list(cfg, s["manifest"])  # seeded
    assert a != calibration_list({"calibration": {**cfg["calibration"], "seed": 1}}, s["manifest"])
    every = calibration_list({"calibration": {**cfg["calibration"], "n": 30}}, s["manifest"])
    assert any("/VD-train/" in p for p in every) and any("/P-train-" in p for p in every)
    with pytest.raises(ValueError, match="only 30"):
        calibration_list({"calibration": {**cfg["calibration"], "n": 31}}, s["manifest"])
    for bad, set_name in (("visdrone/val/z000", "T-VD"), ("visdrone/train/t000", "VD-tune"),
                          ("uitadrone/frames/w_000", "P-val")):
        data = yaml.safe_load(s["data"].read_text(encoding="utf-8"))
        data["train"].append(s["lst"](f"leak-{set_name}", [bad]))
        leaky = tmp_path / f"leaky-{set_name}.yaml"
        leaky.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{set_name}.*role"):
            calibration_list({"calibration": {"list": str(leaky), "n": 31, "seed": 0}},
                             s["manifest"])
    data = yaml.safe_load(s["data"].read_text(encoding="utf-8"))
    data["train"].append(s["lst"]("stray", ["somewhere/else/q"]))
    stray = tmp_path / "stray.yaml"
    stray.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the manifest"):
        calibration_list({"calibration": {"list": str(stray), "n": 31, "seed": 0}}, s["manifest"])


# --- export wrapper (mocked Ultralytics exporter) ----------------------------------------------

def _install_fake_exporter(monkeypatch, calls):
    class FakeYOLO:
        def __init__(self, weights, task=None):
            self.weights = Path(weights)

        def export(self, **kw):
            calls.append(kw)
            out = self.weights.with_suffix(".engine")
            out.write_bytes(b"engine " + json.dumps(kw, sort_keys=True).encode())
            return str(out)

    ul = types.ModuleType("ultralytics")
    ul.YOLO = FakeYOLO
    ul.__version__ = "8.4.fake"
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    import cvt.deploy.engines as eng

    monkeypatch.setattr(eng, "tensorrt_version", lambda: "10.16.fake")


def test_tensorrt_version_requires_major_10(monkeypatch):
    from importlib import metadata

    import cvt.deploy.engines as eng

    installed: dict[str, str] = {}

    def version(dist):
        if dist not in installed:
            raise metadata.PackageNotFoundError(dist)
        return installed[dist]

    monkeypatch.setattr(eng.metadata, "version", version)
    with pytest.raises(RuntimeError, match="not installed"):
        eng.tensorrt_version()
    installed["tensorrt-cu12"] = "10.16.1.11"
    assert eng.tensorrt_version() == "10.16.1.11"
    for bad in ("11.0.0", "8.6.1", "x.y"):
        installed["tensorrt-cu12"] = bad
        with pytest.raises(RuntimeError, match=r"TensorRT 10 is required.*TensorRT 11 drops the "
                                               r"INT8 calibrator.*tensorrt-cu12==10\.16"):
            eng.tensorrt_version()


GLOBAL_ARGS = {"format": "engine", "batch": 1, "dynamic": False, "simplify": False,
               "workspace": 4, "device": 0}


def test_stale_engine_refused(tmp_path, monkeypatch):
    from cvt.deploy.engines import Point, export_engine
    from cvt.utils import file_sha256

    calls = []
    _install_fake_exporter(monkeypatch, calls)
    models = _students_cfg(tmp_path)
    out = tmp_path / "engines"
    p = Point(STUDENTS[0], 960, "fp16", False)
    path = export_engine(p, models, _cfg(), out)
    assert path == out / "src_pseudo-yolo11n-s0@960-fp16.engine" and path.exists()
    assert calls == [{**GLOBAL_ARGS, "quantize": 16, "imgsz": 960}]
    assert not Path(models[STUDENTS[0]]["weights"]).with_suffix(".engine").exists()  # moved
    meta = json.loads((out / "src_pseudo-yolo11n-s0@960-fp16.json").read_text(encoding="utf-8"))
    src = models[STUDENTS[0]]["weights"]
    assert meta == {"source_weights": src, "source_sha256": file_sha256(src),
                    "tensorrt": "10.16.fake", "ultralytics": "8.4.fake",
                    "args": {**GLOBAL_ARGS, "quantize": 16, "imgsz": 960},
                    "calibration_sha256": None}
    assert export_engine(p, models, _cfg(), out) == path and len(calls) == 1  # up to date: skip
    Path(src).write_bytes(b"retrained weights")
    with pytest.raises(RuntimeError, match="stale"):
        export_engine(p, models, _cfg(), out)
    assert len(calls) == 1
    (out / "src_pseudo-yolo11n-s0@960-fp16.json").unlink()
    with pytest.raises(RuntimeError, match="no sidecar"):
        export_engine(p, models, _cfg(), out)
    for bad in (Point(STUDENTS[0], 640, "fp32", False), Point(STUDENTS[0], 1280, "int8", True)):
        with pytest.raises(ValueError, match="not an engine point"):
            export_engine(bad, models, _cfg(), out)


def test_int8_export_writes_calibration_yaml(tmp_path, monkeypatch):
    from cvt.deploy.engines import Point, calibration_list, export_engine

    calls = []
    _install_fake_exporter(monkeypatch, calls)
    s = _calib_setup(tmp_path)
    cfg = {**_cfg(), "calibration": {"list": str(s["data"]), "n": 8, "seed": 0}}
    out = tmp_path / "engines"
    path = export_engine(Point(STUDENTS[1], 640, "int8", False), _students_cfg(tmp_path), cfg, out,
                         manifest_dir=s["manifest"])
    calib_yaml = out / "src_pseudo-yolo11s-s0@640-int8.calib.yaml"
    calib_txt = out / "src_pseudo-yolo11s-s0@640-int8.calib.txt"
    assert calls == [{**GLOBAL_ARGS, "quantize": 8, "imgsz": 640,
                      "data": calib_yaml.resolve().as_posix(), "fraction": 1.0}]
    data = load_yaml(calib_yaml)
    assert data["train"] == data["val"] == [calib_txt.resolve().as_posix()]
    assert data["names"] == dict(enumerate(CLASSES))
    assert calib_txt.read_text(encoding="utf-8").splitlines() == calibration_list(cfg, s["manifest"])
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    import hashlib

    assert meta["calibration_sha256"] == hashlib.sha256(calib_txt.read_bytes()).hexdigest()


def test_int8_up_to_date_by_calibration_content_not_path(tmp_path, monkeypatch):
    import shutil

    from cvt.deploy.engines import Point, export_engine

    calls = []
    _install_fake_exporter(monkeypatch, calls)
    s = _calib_setup(tmp_path)
    cfg = {**_cfg(), "calibration": {"list": str(s["data"]), "n": 8, "seed": 0}}
    models, p = _students_cfg(tmp_path), Point(STUDENTS[1], 640, "int8", False)
    out = tmp_path / "repo" / "engines"
    path = export_engine(p, models, cfg, out, manifest_dir=s["manifest"])
    assert export_engine(p, models, cfg, out, manifest_dir=s["manifest"]) == path
    # the repo moved (absolute calibration yaml path differs), same calibration list: skipped
    moved = tmp_path / "moved" / "engines"
    shutil.copytree(out, moved)
    assert export_engine(p, models, cfg, moved, manifest_dir=s["manifest"]) == (
        moved / path.name)
    assert len(calls) == 1
    # another calibration sample (seed) is stale, even at the same path
    other = {**cfg, "calibration": {**cfg["calibration"], "seed": 1}}
    with pytest.raises(RuntimeError, match="stale"):
        export_engine(p, models, other, out, manifest_dir=s["manifest"])
    # other export args are still compared
    meta_p = path.with_suffix(".json")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["args"]["workspace"] = 8
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        export_engine(p, models, cfg, out, manifest_dir=s["manifest"])
    assert len(calls) == 1


# --- generated registry ------------------------------------------------------------------------

def test_registry_entries_for_all_points(tmp_path):
    from cvt.deploy.engines import grid_points, registry_name, write_registry

    models = _students_cfg(tmp_path)
    out = write_registry(_cfg(), models, tmp_path / "models_deploy.yaml")
    reg = load_yaml(out)
    assert set(reg) == {registry_name(p) for p in grid_points(_cfg())} and len(reg) == 36
    assert reg["src_pseudo-yolo11n-s0@1536-int8"] == {
        "arch": "yolo", "weights": "data/engines/src_pseudo-yolo11n-s0@1536-int8.engine",
        "names_map": "cvt", "imgsz": 1536}
    assert reg["src_pseudo-yolo11m-s0@640-fp32"] == {  # square letterbox, like the engines
        "arch": "yolo", "weights": models["src_pseudo-yolo11m-s0"]["weights"],
        "names_map": "cvt", "imgsz": 640, "rect": False}
    assert reg["src_pseudo-yolo11s-s0@1280+tiles640-int8"] == {
        "arch": "tiled", "base": "src_pseudo-yolo11s-s0@1280-int8",
        "tile_base": "src_pseudo-yolo11s-s0@640-int8", "tile": 640, "overlap": 0.2}
    assert out.read_bytes().count(b"\r") == 0
    del models[STUDENTS[2]]
    with pytest.raises(KeyError, match="src_pseudo-yolo11m-s0"):
        write_registry(_cfg(), models, tmp_path / "x.yaml")
    bad = {**_cfg(), "tiled": {"full_imgsz": 1280, "tile": 512, "overlap": 0.2}}
    with pytest.raises(ValueError, match="tile 512"):
        write_registry(bad, _students_cfg(tmp_path), tmp_path / "x.yaml")


def test_models_files_merge_and_refuse_redefinition(tmp_path):
    from cvt.cli import app
    from cvt.config import load_models

    a, b, c = tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "c.yaml"
    a.write_text(yaml.safe_dump({"oracle": {"arch": "oracle"}, "m1": {"arch": "oracle"}},
                               sort_keys=False))
    b.write_text(yaml.safe_dump({"m2": {"arch": "oracle"}}))
    c.write_text(yaml.safe_dump({"m2": {"arch": "oracle"}, "m1": {"arch": "oracle"}},
                               sort_keys=False))
    assert list(load_models(f"{a},{b}")) == ["oracle", "m1", "m2"]
    assert list(load_models(str(a))) == ["oracle", "m1"] and load_models(a) == load_yaml(a)
    with pytest.raises(ValueError, match=r"'m2'.*already defined in .*b\.yaml"):
        load_models(f"{a}, {b} ,{c}")
    with pytest.raises(ValueError, match="no models file"):
        load_models(" , ")
    runner = CliRunner()
    r = runner.invoke(app, ["eval", "--model", "nope", "--models", f"{a},{b}"])
    assert r.exit_code == 1 and "m2" in r.output and "oracle" in r.output
    r = runner.invoke(app, ["eval", "--model", "m2", "--models", f"{a},{b},{c}"])
    assert r.exit_code == 1 and "already defined" in r.output


def test_deploy_cli_register_and_export_only(tmp_path, monkeypatch):
    from cvt.cli import app

    calls = []
    _install_fake_exporter(monkeypatch, calls)
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump(_students_cfg(tmp_path)))
    runner = CliRunner()
    reg = tmp_path / "models_deploy.yaml"
    conf = ["--config", str(Path(__file__).resolve().parents[1] / "configs" / "deploy.yaml")]
    r = runner.invoke(app, ["deploy", "register", *conf, "--models", str(models), "--out", str(reg)])
    assert r.exit_code == 0, r.output
    assert "36 entries" in r.output and len(load_yaml(reg)) == 36
    out = tmp_path / "engines"
    r = runner.invoke(app, ["deploy", "export", "--only", "src_pseudo-yolo11m-s0@1536-fp16",
                            *conf, "--models", str(models), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "src_pseudo-yolo11m-s0@1536-fp16.engine").exists() and len(calls) == 1
    r = runner.invoke(app, ["deploy", "export", "--only", "src_pseudo-yolo11m-s0@640-fp32",
                            *conf, "--models", str(models), "--out", str(out)])
    assert r.exit_code == 1 and "not an engine of the grid" in r.output


@pytest.mark.parametrize("command", ["export", "bench"])
def test_deploy_commands_set_ultralytics_offline_env_before_import(tmp_path, monkeypatch, command):
    import os

    import cvt.deploy.bench as bench
    import cvt.deploy.engines as eng
    from cvt.cli import app

    for k in ("YOLO_AUTOINSTALL", "YOLO_OFFLINE"):
        monkeypatch.setenv(k, "unset-me")  # so monkeypatch restores the original state
        monkeypatch.delenv(k)
    seen = {}

    def capture(*a, **kw):  # runs where Ultralytics would first be imported
        seen.update({k: os.environ.get(k) for k in ("YOLO_AUTOINSTALL", "YOLO_OFFLINE")})
        raise RuntimeError("stop here")

    monkeypatch.setattr(eng, "export_engine", capture)
    monkeypatch.setattr(bench, "cuda_device", capture)
    conf = ["--config", str(Path(__file__).resolve().parents[1] / "configs" / "deploy.yaml")]
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump(_students_cfg(tmp_path)))
    r = CliRunner().invoke(app, ["deploy", command, *conf, "--models", str(models)])
    assert r.exit_code == 1 and "stop here" in r.output
    assert seen == {"YOLO_AUTOINSTALL": "0", "YOLO_OFFLINE": "1"}
    # setdefault: a value already set is kept
    monkeypatch.setenv("YOLO_OFFLINE", "0")
    CliRunner().invoke(app, ["deploy", command, *conf, "--models", str(models)])
    assert seen["YOLO_OFFLINE"] == "0"


def test_offline_guard_runs_before_the_export(tmp_path, monkeypatch, _guard_calls):
    """2026-10-25: INT8 calibration fetched Arial.ttf; export must run the no-download guard first."""
    import cvt.deploy.engines as eng

    order = []
    monkeypatch.setattr(eng, "_offline_guard", lambda: order.append("guard"))
    monkeypatch.setattr(eng, "tensorrt_version", lambda: "10.16.1.11")

    class FakeYOLO:
        def __init__(self, src):
            pass

        def export(self, **kw):
            order.append("export")
            out = tmp_path / "best.engine"
            out.write_bytes(b"engine")
            return str(out)

    ul = types.ModuleType("ultralytics")
    ul.YOLO = FakeYOLO
    ul.__version__ = "fake"
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights")
    point = eng.Point("m", 640, "fp16", False)
    models_cfg = {"m": {"arch": "yolo", "weights": str(w), "names_map": "cvt"}}
    try:
        eng.export_engine(point, models_cfg, _cfg(), tmp_path / "engines")
    except Exception:
        pass  # later steps (sidecar, move) are covered elsewhere; only the order matters here
    assert order[:2] == ["guard", "export"]
