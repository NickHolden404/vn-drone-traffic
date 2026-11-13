"""Batched (static batch-32) tile engines: grid, export args, registry, engine check, the tiles README
block and its exclusion from the main table (no GPU)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cvt.cli import app
from cvt.config import load_yaml
from cvt.data.schema import CLASSES
from cvt.deploy.bench import LATENCY_COLS, append_latency
from cvt.eval.results import RESULT_COLS, append_results

ROOT = Path(__file__).resolve().parents[1]
M = "src_pseudo-yolo11m-s0"


def _cfg(name: str = "deploy_tiles.yaml") -> dict:
    return load_yaml(ROOT / "configs" / name)


def test_tile_batch_grid_names_and_export_args():
    from cvt.deploy.engines import (
        EXPORT_ARGS,
        engine_points,
        export_args,
        grid_points,
        grid_tile,
        registry_name,
        tile_batch,
    )

    cfg = _cfg()
    plan4 = _cfg("deploy.yaml")
    assert tile_batch(cfg) == 32 and tile_batch(plan4) == 1
    assert {k: v for k, v in cfg["tiled"].items() if k != "tile_batch"} == plan4["tiled"]
    assert cfg["students"] == plan4["students"] and cfg["calibration"] == plan4["calibration"]
    tiled = [registry_name(p, grid_tile(cfg)) for p in grid_points(cfg) if p.tiled]
    assert tiled == [f"src_pseudo-yolo11{s}-s0@1280+tiles640b32-{q}"
                     for s in "nsm" for q in ("fp16", "int8")]
    eng = {registry_name(p): p for p in engine_points(cfg)}
    assert f"{M}@1280-fp16" in eng and f"{M}@640-fp16-b32" in eng and f"{M}@640-int8-b32" in eng
    assert len(eng) == 12  # 6 full-frame (deployment grid, skipped when up to date) + 6 tile engines
    assert export_args(eng[f"{M}@640-fp16-b32"]) == {
        **EXPORT_ARGS, "quantize": 16, "imgsz": 640, "batch": 32}
    assert EXPORT_ARGS["dynamic"] is False  # static: a dynamic profile would reserve 2x imgsz
    # batch-1 engines keep the deployment grid's export args, so their sidecars stay up to date
    assert export_args(eng[f"{M}@1280-fp16"]) == {**EXPORT_ARGS, "quantize": 16, "imgsz": 1280}
    # the deployment grid and its names are unchanged
    names4 = [registry_name(p, grid_tile(plan4)) for p in grid_points(plan4)]
    assert len(names4) == 36 and f"{M}@1280+tiles640-fp16" in names4
    assert not any("b32" in n for n in names4)
    assert [registry_name(p) for p in engine_points(plan4)] == [
        n for n in names4 if "fp32" not in n and "+tiles" not in n]


def test_tile_batch_registry_entries(tmp_path):
    from cvt.deploy.engines import registry_entries

    cfg = _cfg()
    models = {s: {"arch": "yolo", "weights": f"{s}.pt", "names_map": "cvt"} for s in cfg["students"]}
    e = registry_entries(cfg, models, tmp_path)
    t = e[f"{M}@1280+tiles640b32-fp16"]
    assert t == {"arch": "tiled", "base": f"{M}@1280-fp16", "tile_base": f"{M}@640-fp16-b32",
                 "tile": 640, "overlap": 0.2}
    assert e[f"{M}@640-fp16-b32"] == {
        "arch": "yolo", "weights": (tmp_path / f"{M}@640-fp16-b32.engine").as_posix(),
        "names_map": "cvt", "imgsz": 640}
    assert len(e) == 18  # 6 full-frame + 6 tile engines + 6 tiled points
    al = _cfg("deploy_tiles_al.yaml")
    e = registry_entries(al, {"al-m-variety-800": models[M]}, tmp_path)
    assert sorted(e) == ["al-m-variety-800@1280+tiles640b32-fp16", "al-m-variety-800@1280-fp16",
                         "al-m-variety-800@640-fp16-b32"]


def _engine_file(path: Path, imgsz: int, batch: int, dynamic: bool) -> Path:
    meta = json.dumps({"imgsz": [imgsz, imgsz], "batch": batch, "dynamic": dynamic,
                       "names": {"0": "car"}}).encode()
    path.write_bytes(len(meta).to_bytes(4, "little") + meta + b"\x00ENGINE" + path.name.encode())
    return path


def test_batched_engine_predicts_in_batches(tmp_path, monkeypatch):
    from cvt.detect.ultralytics_det import UltralyticsDetector

    calls = []

    class FakeYOLO:
        names = dict(enumerate(CLASSES))

        def __init__(self, weights, task=None):
            pass

        def predict(self, source, **kw):
            calls.append(len(source))
            return [types.SimpleNamespace(boxes=None) for _ in source]

    ul = types.ModuleType("ultralytics")
    ul.YOLO = ul.RTDETR = FakeYOLO
    ul.__version__ = "fake"
    monkeypatch.setitem(sys.modules, "ultralytics", ul)

    def det(path):
        return UltralyticsDetector(name="e", weights=str(path), arch="yolo", names_map="cvt",
                                   imgsz=640, rider_merge=False, batch=4)

    d = det(_engine_file(tmp_path / "t@640-fp16-b32.engine", 640, 32, False))
    assert d.batch == 32
    d.predict_arrays([np.zeros((8, 8, 3), np.uint8)] * 40, imgsz=640)
    assert calls == [32, 8]  # the short batch is Ultralytics' to zero-pad
    with pytest.raises(ValueError, match="dynamic shape"):
        det(_engine_file(tmp_path / "d.engine", 640, 32, True))
    assert det(_engine_file(tmp_path / "one.engine", 640, 1, False)).batch == 1


def _res(model, eval_set, metric, value, imgsz=1280):
    row = {c: "" for c in RESULT_COLS}
    row.update(timestamp="2026-11-05T00:00:00", model=model, imgsz=imgsz, eval_set=eval_set,
               slice="all", metric=metric, value=value, ci_lo=value - 0.02, ci_hi=value + 0.02)
    return row


def _lat(model, ms):
    row = {c: "" for c in LATENCY_COLS}
    row.update(timestamp="2026-11-05T00:00:00", model=model, median_ms=ms)
    return row


def test_tiles_block_and_main_table_exclusion(tmp_path):
    cfg = _cfg("tiles.yaml")
    assert len(cfg["pairs"]) == 7 and cfg["latency"] == "reports/latency_tiles.csv"
    plan4 = {p["static"] for p in cfg["pairs"][:6]}
    from cvt.deploy.engines import grid_points, grid_tile, registry_name
    assert plan4 == {registry_name(p, grid_tile(_cfg("deploy.yaml")))
                     for p in grid_points(_cfg("deploy.yaml")) if p.tiled}
    assert {p["batched"] for p in cfg["pairs"][:6]} == {
        registry_name(p, 640) for p in grid_points(_cfg()) if p.tiled}

    static, batched = f"{M}@1280+tiles640-fp16", f"{M}@1280+tiles640b32-fp16"
    rpath, lpath, lplan4 = tmp_path / "results.csv", tmp_path / "lt.csv", tmp_path / "l4.csv"
    rows = [_res(m, s, k, v) for m, v in ((static, 0.477), (batched, 0.476))
            for s, k in (("T-VN", "mAP50_95"), ("T-VN", "AP50/motorbike"), ("T-VD", "mAP50_95"))]
    append_results(rpath, rows + [_res("teacher-x", "T-VN", "mAP50_95", 0.5)])
    append_latency(lpath, [_lat(f"{M}@1280-fp16", "12.0"), _lat(static, "150.0"),
                           _lat(batched, "50.0")])
    append_latency(lplan4, [_lat(f"{M}@1280-fp16", "10.575")])
    tcfg = tmp_path / "tiles.yaml"
    tcfg.write_text(json.dumps({**cfg, "latency": lpath.as_posix()}), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- tiles:start -->\n(no batched tile results yet)\n<!-- tiles:end -->\n",
                      encoding="utf-8")
    r = CliRunner().invoke(app, ["report", "--results", str(rpath), "--readme", str(readme),
                                 "--latency", str(lplan4), "--tiles-config", str(tcfg),
                                 "--al-config", str(tmp_path / "none.yaml")])
    assert r.exit_code == 0, r.output
    main, tiles = readme.read_text(encoding="utf-8").split("<!-- results:end -->")
    assert "| teacher-x |" in main and "b32" not in main
    line = next(ln for ln in tiles.splitlines() if ln.startswith(f"| {static} -> "))
    assert "47.7 [45.7, 49.7] / 47.6 [45.6, 49.6]" in line
    assert "| 150.0 | 50.0 | 20.0 | 3.0x |" in line
    assert "took 12.0 ms in that session vs 10.6 ms in the deployment benchmark" in tiles
    # pairs without a batched timing show dashes; nothing timed at all keeps the placeholder
    assert "| – | – | – |" in tiles
    lpath.unlink()
    assert CliRunner().invoke(app, ["report", "--results", str(rpath), "--readme", str(readme),
                                    "--latency", str(lplan4), "--tiles-config", str(tcfg),
                                    "--al-config", str(tmp_path / "none.yaml")]).exit_code == 0
    assert "(no batched tile results yet)" in readme.read_text(encoding="utf-8")

