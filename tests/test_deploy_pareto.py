"""Pareto table, plot, README block, `cvt deploy report|list` (no GPU)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cvt.cli import app
from cvt.deploy.bench import LATENCY_COLS
from cvt.eval.results import RESULT_COLS, append_results

CFG = {"students": ["a-yolo11n-s0", "a-yolo11s-s0"], "sizes": [640, 1280], "precisions": ["fp16"],
       "fp32_sizes": [640], "tiled": {"full_imgsz": 1280, "tile": 640, "overlap": 0.2},
       "bench": {"every": 30, "n": 200, "warmup": 20}}


def _res(model, eval_set, value, lo=None, hi=None, ts="2026-10-25T00:00:00", metric="mAP50_95",
         imgsz=None, **kw):
    """A results row; imgsz defaults to the size in the registry name (`name@640-...`)."""
    if imgsz is None:
        m = re.search(r"@(\d+)", model)
        imgsz = int(m.group(1)) if m else 1280
    row = {c: "" for c in RESULT_COLS}
    row.update(timestamp=ts, model=model, imgsz=imgsz, eval_set=eval_set, slice="all",
               metric=metric, value=value, ci_lo=value - 0.05 if lo is None else lo,
               ci_hi=value + 0.05 if hi is None else hi)
    row.update(kw)
    return row


def _lat(model, median, imgsz=640, precision="fp16", tiled=False, ts="2026-10-25T01:00:00", **kw):
    row = {c: "" for c in LATENCY_COLS}
    row.update(timestamp=ts, model=model, imgsz=imgsz, precision=precision, tiled=tiled,
               n_frames=200, warmup=20, median_ms=median, p90_ms=median * 1.2,
               fps=1000.0 / median, gpu_mem_mb=500.0, gpu="NVIDIA GeForce RTX 4070",
               driver="580.97", tensorrt="10.16.0", ultralytics="8.4.0", torch="2.9.0")
    row.update(kw)
    return row


def _frames(results, latency):
    return pd.DataFrame(results, columns=RESULT_COLS), pd.DataFrame(latency, columns=LATENCY_COLS)


# --- frontier -----------------------------------------------------------------------------------

def test_frontier_handles_ties_and_missing():
    from cvt.deploy.pareto import frontier_mask

    nan = float("nan")
    #       A     B     C     D     E     F     G    H     I
    acc = [0.40, 0.30, 0.50, 0.45, 0.50, 0.60, nan, 0.90, 0.40]
    ms = [10.0, 12.0, 20.0, 20.0, 20.0, 50.0, 5.0, nan, 10.0]
    got = frontier_mask(acc, ms).tolist()
    # B dominated by A; D dominated by C (equal latency, lower mAP); C and E equal on both:
    # both kept (as are A and I); G (no accuracy) and H (no latency) are never on the frontier
    # and do not dominate F
    assert got == [True, False, True, False, True, True, False, False, True]
    with pytest.raises(ValueError, match="same length"):
        frontier_mask([0.1], [1.0, 2.0])


# --- table --------------------------------------------------------------------------------------

def test_pareto_table_joins_points():
    from cvt.deploy.pareto import PARETO_COLS, pareto_table

    n, s = "a-yolo11n-s0", "a-yolo11s-s0"
    res = [
        _res(f"{n}@640-fp16", "T-VN", 0.30, 0.25, 0.35, ts="2026-10-24T00:00:00"),
        _res(f"{n}@640-fp16", "T-VN", 0.32, 0.27, 0.37),  # later row wins
        _res(f"{n}@640-fp16", "T-VD", 0.20, 0.18, 0.22),
        _res(f"{n}@640-fp16", "T-VN", 0.99, metric="mAP50"),  # other metric: ignored
        _res(f"{s}@1280-fp16", "T-VN", 0.40, 0.30, 0.50),
        _res(f"{s}@1280-fp16", "T-VD", 0.38, 0.35, 0.41),
        _res(f"{s}@1280+tiles640-fp16", "T-VN", 0.45, 0.40, 0.50),
        _res("other-model", "T-VN", 0.9),  # not a grid point
    ]
    lat = [
        _lat(f"{n}@640-fp16", 99.0, ts="2026-10-24T00:00:00"),
        _lat(f"{n}@640-fp16", 4.0),  # later row wins
        _lat(f"{s}@1280-fp16", 12.0, imgsz=1280, gpu_mem_mb=800.0),
        _lat(f"{s}@1280+tiles640-fp16", 80.0, imgsz=1280, tiled=True),
        _lat(f"{n}@640-fp32", 9.0, precision="fp32"),  # latency only
    ]
    results, latency = _frames(res, lat)
    cfg = {**CFG, "students": [n, s]}
    t = pareto_table(results, latency, cfg)
    assert list(t.columns) == PARETO_COLS
    # one row per grid point: 2 students x (2 sizes fp16 + 1 fp32 + 1 tiled) = 8
    assert len(t) == 8 and t.model.is_unique
    row = t.set_index("model").loc[f"{n}@640-fp16"]
    assert row["T-VN_mAP50_95"] == pytest.approx(0.32) and row["T-VN_ci_lo"] == pytest.approx(0.27)
    assert row["T-VD_mAP50_95"] == pytest.approx(0.20) and row["T-VD_ci_hi"] == pytest.approx(0.22)
    assert row.median_ms == pytest.approx(4.0) and row.p90_ms == pytest.approx(4.8)
    assert row.fps == pytest.approx(250.0) and row.student == n and not row.tiled
    assert row.imgsz == 640 and row.precision == "fp16"
    tiled = t.set_index("model").loc[f"{s}@1280+tiles640-fp16"]
    assert bool(tiled.tiled) and tiled.imgsz == 1280 and math.isnan(tiled["T-VD_mAP50_95"])
    assert t.set_index("model").loc[f"{s}@1280-fp16"].gpu_mem_mb == pytest.approx(800.0)
    # missing accuracy / latency -> no frontier; sorted by student, tiled, imgsz, precision
    fp32 = t.set_index("model").loc[f"{n}@640-fp32"]
    assert math.isnan(fp32["T-VN_mAP50_95"]) and not fp32.frontier
    order = t[["student", "tiled", "imgsz"]].itertuples(index=False, name=None)
    assert list(order) == sorted(t[["student", "tiled", "imgsz"]].itertuples(index=False, name=None))
    # n@640 (0.32, 4 ms) and s@1280 (0.40, 12 ms) and tiled (0.45, 80 ms) are on the frontier
    assert t[t.frontier].model.tolist() == [f"{n}@640-fp16", f"{s}@1280-fp16",
                                            f"{s}@1280+tiles640-fp16"]
    assert t.frontier.dtype == bool


def test_pareto_table_ignores_rows_at_another_imgsz():
    from cvt.deploy.pareto import pareto_table

    n = "a-yolo11n-s0"
    cfg = {**CFG, "students": [n]}
    full, tiled = f"{n}@1280-fp16", f"{n}@1280+tiles640-fp16"
    res = [
        _res(full, "T-VN", 0.30, ts="2026-10-24T00:00:00"),  # right size
        _res(full, "T-VN", 0.99, ts="2026-10-25T05:00:00", imgsz=640),  # later, stale size
        _res(f"{n}@640-fp16", "T-VN", 0.50, imgsz=1280),  # only a 1280 row for a 640 point
        _res(tiled, "T-VN", 0.45, imgsz=640),  # tiled points are evaluated at 1280
        _res(f"{n}@640-fp16", "T-VD", 0.20, imgsz=640),
    ]
    lat = [_lat(full, 12.0, imgsz=1280), _lat(f"{n}@640-fp16", 4.0),
           _lat(tiled, 80.0, imgsz=1280, tiled=True)]
    t = pareto_table(*_frames(res, lat), cfg).set_index("model")
    assert t.at[full, "T-VN_mAP50_95"] == pytest.approx(0.30)  # the stale later row does not win
    p640 = t.loc[f"{n}@640-fp16"]
    assert math.isnan(p640["T-VN_mAP50_95"]) and not p640.frontier  # missing accuracy
    assert p640["T-VD_mAP50_95"] == pytest.approx(0.20)
    assert math.isnan(t.at[tiled, "T-VN_mAP50_95"]) and not t.at[tiled, "frontier"]
    # without cfg the latency row's imgsz selects the accuracy rows
    t2 = pareto_table(*_frames(res, lat)).set_index("model")
    assert t2.at[full, "T-VN_mAP50_95"] == pytest.approx(0.30)
    assert math.isnan(t2.at[f"{n}@640-fp16", "T-VN_mAP50_95"])
    # only a stale-size row: NaN, not frontier
    only_stale = [_res(full, "T-VN", 0.99, imgsz=640)]
    t3 = pareto_table(*_frames(only_stale, lat), cfg).set_index("model")
    assert math.isnan(t3.at[full, "T-VN_mAP50_95"]) and not t3.at[full, "frontier"]


def test_pareto_table_without_cfg_uses_latency_points():
    from cvt.deploy.pareto import pareto_table

    res = [_res("m@640-fp16", "T-VN", 0.3)]
    lat = [_lat("m@640-fp16", 5.0), _lat("m@640+tiles640-int8", 30.0, precision="int8",
                                           tiled="True")]
    t = pareto_table(*_frames(res, lat))
    assert sorted(t.model) == ["m@640+tiles640-int8", "m@640-fp16"]
    assert t.set_index("model").loc["m@640+tiles640-int8"].tiled
    empty = pareto_table(*_frames([], []))
    assert empty.empty


def test_pareto_table_accepts_csv_dtypes(tmp_path: Path):
    """Latency rows read back from reports/latency.csv are all strings; slice may be absent."""
    from cvt.deploy.pareto import pareto_table

    p = tmp_path / "latency.csv"
    pd.DataFrame([_lat("m@640-fp16", 5.0)], columns=LATENCY_COLS).to_csv(p, index=False)
    lat = pd.read_csv(p, dtype=str, keep_default_na=False)
    res = pd.DataFrame([_res("m@640-fp16", "T-VN", 0.3)], columns=RESULT_COLS).drop(columns="slice")
    t = pareto_table(res, lat)
    r = t.iloc[0]
    assert r.median_ms == pytest.approx(5.0) and r.imgsz == 640 and r.frontier


# --- plot ---------------------------------------------------------------------------------------

def _plot_table():
    n, s = "a-yolo11n-s0", "a-yolo11s-s0"
    res, lat = [], []
    for st, sz, prec, tiled, v, ms in [
            (n, 640, "fp16", False, 0.30, 4.0), (n, 640, "int8", False, 0.29, 3.0),
            (s, 1280, "fp16", False, 0.41, 12.0), (s, 1280, "fp32", False, 0.40, 40.0),
            (s, 1280, "int8", True, 0.45, 80.0), (n, 960, "fp16", False, 0.25, 9.0)]:
        name = (f"{st}@{sz}+tiles640-{prec}" if tiled else f"{st}@{sz}-{prec}")
        res.append(_res(name, "T-VN", v))
        lat.append(_lat(name, ms, imgsz=sz, precision=prec, tiled=tiled))
    from cvt.deploy.pareto import pareto_table

    return pareto_table(*_frames(res, lat))


def test_plot_writes_png(tmp_path: Path):
    from cvt.deploy.pareto import plot_pareto

    table = _plot_table()
    out = tmp_path / "sub" / "pareto.png"
    assert plot_pareto(table, out) == out
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5_000
    # no frontier / no plottable points: fail loudly
    with pytest.raises(ValueError, match="no point with both"):
        plot_pareto(table.assign(median_ms=np.nan), tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()


# --- README block -------------------------------------------------------------------------------

def test_readme_block_renders():
    from cvt.deploy.pareto import PLACEHOLDER, pareto_block

    table = _plot_table()
    n = "a-yolo11n-s0"
    _, latency = _frames([], [_lat(f"{n}@640-fp16", 4.0)])
    md = pareto_block(table, latency, png="reports/pareto.png")
    assert "![Pareto](reports/pareto.png)" in md
    assert "![Pareto]" not in pareto_block(table, latency, png=None)
    head, _, rest = md.partition("\n\n")
    assert "frontier" in head.lower() and "|" not in head
    tables = [blk for blk in md.split("\n\n") if blk.startswith("| ")]
    front, full = tables
    n_front = int(table.frontier.sum())
    assert len(front.splitlines()) == 2 + n_front
    assert len(full.splitlines()) == 2 + len(table)
    assert "T-VN mAP50-95" in front and "median ms" in front and "gpu_mem_mb" in front
    # frontier table sorted by latency; CI in brackets, percent; fp32 point is in the full table
    assert "| a-yolo11n-s0@640-int8 |" in front
    assert "30.0 [25.0, 35.0]" in full or "29.0 [24.0, 34.0]" in full
    assert "a-yolo11s-s0@1280-fp32" in full and "a-yolo11s-s0@1280-fp32" not in front
    for needle in ("NVIDIA GeForce RTX 4070 desktop GPU", "580.97", "10.16.0", "batch 1",
                   "end-to-end", "200 fixed T-VN frames", "other than 1280", "inference-only",
                   "device-level", "steady-state",
                   "All points use a square letterbox at their input size",
                   "a Jetson-class device would be several times slower and was not measured",
                   "no other point is at least as accurate and at least as fast, and better on one"):
        assert needle in md, needle
    assert "4K" not in md and "Decoding" not in md  # no frame mix / decode time recorded
    assert "no other point has higher" not in md
    # sorted by student, tiled, imgsz, precision in the full table
    names = [ln.split("|")[1].strip() for ln in full.splitlines()[2:]]
    assert names == table.model.tolist()
    # no latency rows -> placeholder
    assert pareto_block(table.iloc[0:0], latency.iloc[0:0]) == PLACEHOLDER


def test_caption_frame_mix_and_decode_time_from_latency_rows():
    from cvt.deploy.pareto import pareto_block

    table = _plot_table()
    n = "a-yolo11n-s0"
    mix = "3840x2160:187;1920x1080:12"
    _, lat = _frames([], [_lat(f"{n}@640-fp16", 4.0, n_frames=199, frame_sizes=mix,
                               decode_ms_per_frame=31.25),
                          _lat(f"{n}@640-int8", 3.0, n_frames=199, frame_sizes=mix,
                               decode_ms_per_frame=30.75)])
    md = pareto_block(table, lat)
    assert "199 fixed T-VN frames (187 at 3840x2160, 12 at 1920x1080)" in md
    assert "Decoding a frame (cv2.imread, excluded from the timings) took 31.0 ms." in md
    assert md.count("Decoding") == 1
    _, one = _frames([], [_lat(f"{n}@640-fp16", 4.0, n_frames=199, frame_sizes="3840x2160:199")])
    assert "199 fixed T-VN frames (all 3840x2160)" in pareto_block(table, one)


# --- CLI ----------------------------------------------------------------------------------------

def _cli_files(tmp_path: Path):
    n = "a-yolo11n-s0"
    cfg = {**CFG, "students": [n]}
    cfg_path = tmp_path / "deploy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rpath, lpath = tmp_path / "results.csv", tmp_path / "latency.csv"
    names = [f"{n}@640-fp16", f"{n}@1280-fp16"]
    append_results(rpath, [_res(names[0], "T-VN", 0.3), _res(names[1], "T-VN", 0.4),
                           _res(names[1], "T-VD", 0.38)])
    pd.DataFrame([_lat(names[0], 4.0), _lat(names[1], 12.0, imgsz=1280)],
                 columns=LATENCY_COLS).to_csv(lpath, index=False)
    return cfg_path, rpath, lpath


def test_deploy_list_prints_name_and_imgsz(tmp_path: Path):
    cfg_path, _, _ = _cli_files(tmp_path)
    r = CliRunner().invoke(app, ["deploy", "list", "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    lines = r.stdout.strip().splitlines()
    n = "a-yolo11n-s0"
    # 1 student x (2 sizes fp16 + 1 fp32 + 1 tiled) = 4 points, `name imgsz`
    assert lines == [f"{n}@640-fp16 640", f"{n}@1280-fp16 1280", f"{n}@640-fp32 640",
                     f"{n}@1280+tiles640-fp16 1280"]


def test_deploy_report_writes_csv_and_png(tmp_path: Path):
    cfg_path, rpath, lpath = _cli_files(tmp_path)
    out = tmp_path / "out"
    r = CliRunner().invoke(app, [
        "deploy", "report", "--config", str(cfg_path), "--results", str(rpath),
        "--latency", str(lpath), "--out-csv", str(out / "pareto.csv"),
        "--out-png", str(out / "pareto.png")])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(out / "pareto.csv")
    assert len(df) == 4 and df.frontier.sum() == 2
    assert (out / "pareto.png").read_bytes()[:4] == b"\x89PNG"
    # missing latency file: loud
    r = CliRunner().invoke(app, ["deploy", "report", "--config", str(cfg_path), "--results",
                                 str(rpath), "--latency", str(tmp_path / "none.csv"),
                                 "--out-csv", str(out / "p.csv"), "--out-png", str(out / "p.png")])
    assert r.exit_code != 0 and "latency" in r.output


def test_report_fills_pareto_block(tmp_path: Path):
    cfg_path, rpath, lpath = _cli_files(tmp_path)
    readme = tmp_path / "README.md"
    base = ("<!-- results:start -->\n<!-- results:end -->\n"
            "<!-- pareto:start -->\n(no deployment results yet)\n<!-- pareto:end -->\n")
    readme.write_text(base, encoding="utf-8")
    args = ["report", "--results", str(rpath), "--readme", str(readme),
            "--latency", str(lpath), "--deploy-config", str(cfg_path),
            "--pareto-png", str(tmp_path / "pareto.png")]
    # no latency file: the pareto block keeps its placeholder, other blocks unchanged
    r = CliRunner().invoke(app, ["report", "--results", str(rpath), "--readme", str(readme),
                                 "--latency", str(tmp_path / "none.csv")])
    assert r.exit_code == 0, r.output
    assert "(no deployment results yet)" in readme.read_text(encoding="utf-8")
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    assert "(no deployment results yet)" not in text and "a-yolo11n-s0@640-fp16" in text
    assert "![Pareto]" not in text  # the PNG does not exist yet
    (tmp_path / "pareto.png").write_bytes(b"\x89PNG")
    CliRunner().invoke(app, args)
    assert "![Pareto](" in readme.read_text(encoding="utf-8")
    # a README without pareto markers is left alone (like compare / tracking)
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
    assert CliRunner().invoke(app, args).exit_code == 0


def test_report_main_table_excludes_grid_points(tmp_path: Path):
    cfg_path, rpath, lpath = _cli_files(tmp_path)
    append_results(rpath, [_res("teacher-x", "T-VN", 0.5)])  # not a grid point
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- pareto:start -->\n<!-- pareto:end -->\n", encoding="utf-8")
    args = ["report", "--results", str(rpath), "--readme", str(readme), "--latency", str(lpath),
            "--pareto-png", str(tmp_path / "pareto.png")]
    r = CliRunner().invoke(app, [*args, "--deploy-config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    main = text.split("<!-- results:end -->")[0]
    pareto = text.split("<!-- pareto:start -->")[1]
    assert "| teacher-x |" in main and "@640-fp16" not in main and "@1280-fp16" not in main
    assert "a-yolo11n-s0@640-fp16" in pareto
    # without a deploy config every model stays in the main table (no grid to exclude)
    r = CliRunner().invoke(app, [*args, "--deploy-config", str(tmp_path / "none.yaml")])
    assert r.exit_code == 0, r.output
    assert "| a-yolo11n-s0@640-fp16 |" in readme.read_text(encoding="utf-8").split(
        "<!-- results:end -->")[0]


def test_stale_accuracy_is_blanked_and_listed():
    """results.csv model_sha is file_sha256(weights)[:12] (engine file or .pt); a latency row
    with another weights_sha means the engine / checkpoint changed after `cvt eval`."""
    from cvt.deploy.pareto import pareto_block, pareto_table

    n = "a-yolo11n-s0"
    cfg = {**CFG, "students": [n]}
    fresh, stale, fp32, tiled = (f"{n}@640-fp16", f"{n}@1280-fp16", f"{n}@640-fp32",
                                 f"{n}@1280+tiles640-fp16")
    a, b = "a" * 64, "b" * 64
    res = [_res(fresh, "T-VN", 0.30, model_sha=a[:12]), _res(fresh, "T-VD", 0.2, model_sha=a[:12]),
           _res(stale, "T-VN", 0.90, model_sha=a[:12]), _res(stale, "T-VD", 0.8, model_sha=a[:12]),
           _res(fp32, "T-VN", 0.35, model_sha=""),  # no latency sha to compare: kept
           _res(tiled, "T-VN", 0.95, imgsz=1280, model_sha="cachetag1234")]  # tiled: not checked
    lat = [_lat(fresh, 4.0, weights_sha=a), _lat(stale, 12.0, imgsz=1280, weights_sha=b),
           _lat(fp32, 9.0, precision="fp32"),
           _lat(tiled, 80.0, imgsz=1280, tiled=True, weights_sha=b)]
    results, latency = _frames(res, lat)
    t = pareto_table(results, latency, cfg).set_index("model")
    assert t.at[fresh, "T-VN_mAP50_95"] == pytest.approx(0.30) and not t.at[fresh, "accuracy_stale"]
    assert math.isnan(t.at[stale, "T-VN_mAP50_95"]) and math.isnan(t.at[stale, "T-VD_mAP50_95"])
    assert t.at[stale, "accuracy_stale"] and not t.at[stale, "frontier"]
    assert t.at[fp32, "T-VN_mAP50_95"] == pytest.approx(0.35) and not t.at[fp32, "accuracy_stale"]
    assert t.at[tiled, "T-VN_mAP50_95"] == pytest.approx(0.95) and not t.at[tiled, "accuracy_stale"]
    assert t.accuracy_stale.dtype == bool
    md = pareto_block(t.reset_index(), latency)
    assert f"Accuracy stale: re-run eval for {stale}" in md
    assert "Accuracy stale" not in pareto_block(t.reset_index().assign(accuracy_stale=False),
                                                latency)
    # a re-evaluation with the new engine is fresh again
    res.append(_res(stale, "T-VN", 0.5, model_sha=b[:12], ts="2026-10-26T00:00:00"))
    t2 = pareto_table(*_frames(res, lat), cfg).set_index("model")
    assert t2.at[stale, "T-VN_mAP50_95"] == pytest.approx(0.5)
    assert math.isnan(t2.at[stale, "T-VD_mAP50_95"]) and t2.at[stale, "accuracy_stale"]  # T-VD old
    res.append(_res(stale, "T-VD", 0.4, model_sha=b[:12], ts="2026-10-26T00:00:00"))
    t3 = pareto_table(*_frames(res, lat), cfg).set_index("model")
    assert t3.at[stale, "T-VD_mAP50_95"] == pytest.approx(0.4) and not t3.at[stale, "accuracy_stale"]
