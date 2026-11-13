"""Active-learning curve table, plot, README block, headline sentence, the AL
engine grid (configs/deploy_al.yaml) and the `cvt al list|comparisons|report` helpers (no GPU)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cvt.cli import app
from cvt.config import load_yaml
from cvt.deploy.bench import LATENCY_COLS
from cvt.eval.results import RESULT_COLS, append_results

ROOT = Path(__file__).resolve().parents[1]
CLASSES6 = ("car", "motorbike", "bus", "truck", "pedestrian", "bicycle")
HEADLINE_ENGINE = "al-m-variety-800@1280-fp16"


def _cfg() -> dict:
    return load_yaml(ROOT / "configs" / "al.yaml")


def _res(model, eval_set, metric, value, lo=None, hi=None, ts="2026-10-29T00:00:00",
         imgsz=1280, **kw):
    row = {c: "" for c in RESULT_COLS}
    row.update(timestamp=ts, model=model, imgsz=imgsz, eval_set=eval_set, slice="all",
               metric=metric, value=value, ci_lo=value - 0.03 if lo is None else lo,
               ci_hi=value + 0.03 if hi is None else hi)
    row.update(kw)
    return row


def _model_rows(model, base, ts="2026-10-29T00:00:00", imgsz=1280):
    """T-VN mAP50-95 + six per-class AP50 and T-VD mAP50-95 rows around `base`."""
    rows = [_res(model, "T-VN", "mAP50_95", base, ts=ts, imgsz=imgsz),
            _res(model, "T-VD", "mAP50_95", base - 0.02, ts=ts, imgsz=imgsz)]
    rows += [_res(model, "T-VN", f"AP50/{c}", base + 0.3 - 0.05 * i, ts=ts, imgsz=imgsz)
             for i, c in enumerate(CLASSES6)]
    return rows


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLS)


# --- curve table --------------------------------------------------------------------------------

def test_curve_table_from_results():
    from cvt.al.report import POOL_FRAMES, curve_table

    cfg = _cfg()
    rows = _model_rows("src_pseudo-yolo11s-s0", 0.40)
    rows += _model_rows("al-s-random-100", 0.30, ts="2026-10-29T00:00:00")
    rows += _model_rows("al-s-random-100", 0.42, ts="2026-10-29T05:00:00")  # latest wins
    rows += _model_rows("al-s-random-100", 0.99, ts="2026-10-29T09:00:00", imgsz=640)  # other size
    rows.append(_res("al-s-random-100", "T-VN", "mAP50_95", 0.77, ts="2026-10-30T00:00:00",
                     slice="daylight"))  # not the `all` slice
    rows += _model_rows("al-s-all", 0.50)
    rows += _model_rows("al-s-random-400-r1", 0.44)
    t = curve_table(_df(rows), cfg)
    assert POOL_FRAMES == 9352
    # two starting students (budget 0) + the 20 inventory runs
    assert len(t) == 22 and list(t.run[:2]) == ["src_pseudo-yolo11s-s0", "src_pseudo-yolo11m-s0"]
    r = t.set_index("run")
    start = r.loc["src_pseudo-yolo11s-s0"]
    assert start.strategy == "start" and start.budget == 0 and start.student == "s"
    assert start["T-VN_mAP50_95"] == pytest.approx(0.40)
    assert r.loc["al-s-random-100", "T-VN_mAP50_95"] == pytest.approx(0.42)
    assert r.loc["al-s-random-100", "T-VN_mAP50_95_lo"] == pytest.approx(0.39)
    assert r.loc["al-s-random-100", "T-VN_AP50_motorbike"] == pytest.approx(0.42 + 0.25)
    assert r.loc["al-s-random-100", "T-VD_mAP50_95"] == pytest.approx(0.40)
    assert r.loc["al-s-all", "budget"] == 9352 and r.loc["al-s-all", "strategy"] == "all"
    assert r.loc["al-s-random-400-r1", "ranking_seed"] == 1
    assert r.loc["al-s-variety-400-t2", "seed"] == 2
    assert r.loc["al-m-variety-800", "student"] == "m"
    assert pd.isna(r.loc["al-s-variety-800", "T-VN_mAP50_95"])  # not evaluated yet
    for c in CLASSES6:
        assert f"T-VN_AP50_{c}" in t and f"T-VN_AP50_{c}_hi" in t


# --- headline sentence --------------------------------------------------------------------------

def _headline_rows(car, moto, model=HEADLINE_ENGINE, ts="2026-10-29T00:00:00"):
    return [_res(model, "T-VN", "AP50/car", car[0], car[1], car[2], ts=ts),
            _res(model, "T-VN", "AP50/motorbike", moto[0], moto[1], moto[2], ts=ts)]


def test_headline_sentence_rule():
    from cvt.al.report import headline_sentence

    cfg = _cfg()
    rows = _headline_rows((0.912, 0.850, 0.951), (0.896, 0.840, 0.930))
    s = headline_sentence(_df(rows), cfg)
    assert "car AP50 91.2 [85.0, 95.1]" in s and "motorbike AP50 89.6 [84.0, 93.0]" in s
    assert "on T-VN with 800 labelled frames" in s
    assert "car reaches 90" in s and "motorbike does not reach 90" in s
    assert HEADLINE_ENGINE in s
    # exactly 90 reaches (point estimate >= 90)
    s = headline_sentence(_df(_headline_rows((0.90, 0.85, 0.95), (0.95, 0.9, 0.97))), cfg)
    assert "car reaches 90" in s and "motorbike reaches 90" in s
    # the verdict follows the printed one-decimal value
    s = headline_sentence(_df(_headline_rows((0.8996, 0.85, 0.95), (0.8994, 0.85, 0.95))), cfg)
    assert "car AP50 90.0 [85.0, 95.0]" in s and "car reaches 90" in s
    assert "motorbike AP50 89.9 [85.0, 95.0]" in s and "motorbike does not reach 90" in s
    assert "one-decimal point estimate" in s
    # primary run only: other engines / the PyTorch model never stand in for it
    others = (_headline_rows((0.99, 0.98, 1.0), (0.99, 0.98, 1.0), model="al-m-all@1280-fp16")
              + _headline_rows((0.99, 0.98, 1.0), (0.99, 0.98, 1.0), model="al-m-variety-800")
              + _headline_rows((0.99, 0.98, 1.0), (0.99, 0.98, 1.0),
                               model="al-m-random-800@1280-fp16"))
    s = headline_sentence(_df(others), cfg)
    assert "not evaluated yet" in s and "reaches 90" not in s and "99.0" not in s
    s = headline_sentence(_df(others + rows), cfg)
    assert "car AP50 91.2" in s and "motorbike does not reach 90" in s
    # the latest row of the primary engine counts
    later = _headline_rows((0.80, 0.75, 0.85), (0.92, 0.9, 0.94), ts="2026-10-30T00:00:00")
    s = headline_sentence(_df(rows + later), cfg)
    assert "car does not reach 90" in s and "motorbike reaches 90" in s
    # one class missing: not evaluated
    assert "not evaluated yet" in headline_sentence(_df(rows[:1]), cfg)


# --- plot ---------------------------------------------------------------------------------------

def test_plot_writes_png(tmp_path: Path):
    from cvt.al.report import curve_table, plot_curve

    cfg = _cfg()
    rows = _model_rows("src_pseudo-yolo11s-s0", 0.40) + _model_rows("al-s-all", 0.55)
    for i, b in enumerate((100, 200, 400, 800)):
        for j, st in enumerate(("random", "disagree", "variety")):
            rows += _model_rows(f"al-s-{st}-{b}", 0.41 + 0.02 * i + 0.01 * j)
    out = plot_curve(curve_table(_df(rows), cfg), tmp_path / "sub" / "al_curve.png")
    assert out.read_bytes()[:4] == b"\x89PNG"
    # partial results (one budget) still plot; nothing to plot is loud
    part = _model_rows("al-s-random-100", 0.41)
    assert plot_curve(curve_table(_df(part), cfg), tmp_path / "p.png").exists()
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_curve(curve_table(_df([]), cfg), tmp_path / "none.png")


# --- AL engine grid -----------------------------------------------------------------------------

def _al_models(tmp_path: Path, names) -> dict:
    out = {}
    for n in names:
        w = tmp_path / "runs" / "al" / n / "weights" / "last.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        w.write_bytes(f"weights of {n}".encode())
        out[n] = {"arch": "yolo", "weights": w.as_posix(), "names_map": "cvt"}
    return out


def test_grid_without_tiled_or_fp32(tmp_path: Path):
    from cvt.deploy.engines import engine_points, grid_points, registry_entries, registry_name

    cfg = load_yaml(ROOT / "configs" / "deploy_al.yaml")
    plan4 = load_yaml(ROOT / "configs" / "deploy.yaml")
    al = _cfg()
    assert cfg["students"] == ["al-m-variety-800", "al-m-random-800", "al-m-all"]
    assert cfg["students"] == al["engines"]["models"] and al["headline"] == cfg["students"][0]
    assert cfg["sizes"] == [1280] and cfg["precisions"] == ["fp16"]
    assert cfg["fp32_sizes"] == [] and cfg["tiled"] is None
    assert cfg["calibration"] == plan4["calibration"] and cfg["bench"] == plan4["bench"]
    pts = grid_points(cfg)
    assert pts == engine_points(cfg) and len(pts) == 3
    assert [registry_name(p) for p in pts] == [
        HEADLINE_ENGINE, "al-m-random-800@1280-fp16", "al-m-all@1280-fp16"]
    assert len(grid_points(plan4)) == 36  # the deployment grid is unchanged
    models = _al_models(tmp_path, cfg["students"])
    entries = registry_entries(cfg, models, tmp_path / "engines")
    assert list(entries) == [registry_name(p) for p in pts]
    assert entries[HEADLINE_ENGINE] == {
        "arch": "yolo", "weights": (tmp_path / "engines" / f"{HEADLINE_ENGINE}.engine").as_posix(),
        "names_map": "cvt", "imgsz": 1280}
    # `cvt deploy list` / `register` with the AL config and an extra models file
    r = CliRunner().invoke(app, ["deploy", "list", "--config", str(ROOT / "configs/deploy_al.yaml")])
    assert r.exit_code == 0, r.output
    assert r.stdout.strip().splitlines()[0] == f"{HEADLINE_ENGINE} 1280"
    base = tmp_path / "models.yaml"
    base.write_text(yaml.safe_dump({"teacher": {"arch": "yolo", "weights": "t.pt",
                                                "names_map": "cvt"}}), encoding="utf-8")
    extra = tmp_path / "models_al.yaml"
    extra.write_text(yaml.safe_dump(models), encoding="utf-8")
    # default --out follows the config name: deploy_al.yaml -> models_deploy_al.yaml
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    shutil.copy(ROOT / "configs/deploy_al.yaml", cfgs / "deploy_al.yaml")
    out = cfgs / "models_deploy_al.yaml"
    r = CliRunner().invoke(app, [
        "deploy", "register", "--config", str(cfgs / "deploy_al.yaml"),
        "--models", f"{base.as_posix()},{extra.as_posix()}",
        "--engines", str(tmp_path / "engines")])
    assert r.exit_code == 0, r.output
    assert out.exists() and not (cfgs / "models_deploy.yaml").exists()
    text = out.read_text(encoding="utf-8")
    assert "deploy_al.yaml" in text.splitlines()[0]
    assert f"--models {base.as_posix()},{extra.as_posix()},{out.as_posix()}" in text.splitlines()[1]
    assert list(yaml.safe_load(text)) == list(entries)


def test_register_default_out_and_explicit_out(tmp_path: Path):
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    shutil.copy(ROOT / "configs/deploy.yaml", cfgs / "deploy.yaml")
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump(_al_models(tmp_path, load_yaml(cfgs / "deploy.yaml")[
        "students"])), encoding="utf-8")
    args = ["deploy", "register", "--config", str(cfgs / "deploy.yaml"), "--models", str(models)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert len(load_yaml(cfgs / "models_deploy.yaml")) == 36
    r = CliRunner().invoke(app, [*args, "--out", str(tmp_path / "x.yaml")])
    assert r.exit_code == 0, r.output
    assert len(load_yaml(tmp_path / "x.yaml")) == 36


def test_plan4_registry_header_unchanged(tmp_path: Path):
    from cvt.deploy.engines import write_registry

    cfg = load_yaml(ROOT / "configs" / "deploy.yaml")
    models = _al_models(tmp_path, cfg["students"])
    out = write_registry(cfg, models, tmp_path / "models_deploy.yaml")
    head = out.read_text(encoding="utf-8").splitlines()[:2]
    committed = (ROOT / "configs" / "models_deploy.yaml").read_text(encoding="utf-8").splitlines()
    # the defaults (config, models, out = configs/models_deploy.yaml) give the committed header
    assert head[0] == committed[0]
    assert head[1] == f"# Use with: --models configs/models.yaml,{out.as_posix()}"
    assert committed[1] == "# Use with: --models configs/models.yaml,configs/models_deploy.yaml"


# --- README block -------------------------------------------------------------------------------

def _lat(model, median, ts="2026-10-29T01:00:00"):
    row = {c: "" for c in LATENCY_COLS}
    row.update(timestamp=ts, model=model, imgsz=1280, precision="fp16", tiled=False,
               n_frames=199, warmup=20, median_ms=median, p90_ms=median * 1.1,
               fps=1000.0 / median, gpu_mem_mb=260.0)
    return row


def _delta(a, b, eval_set, metric, value):
    return _res(a, eval_set, f"delta/{metric}", value, value - 0.01, value + 0.01,
                notes=f"vs {b}@1280; mAP classes: motorbike,car,bus,truck,bicycle,pedestrian")


def test_al_block_sections():
    from cvt.al.report import PLACEHOLDER, al_block

    cfg = _cfg()
    start = _model_rows("src_pseudo-yolo11s-s0", 0.40) + _model_rows("src_pseudo-yolo11m-s0", 0.44)
    assert al_block(_df(start), pd.DataFrame(columns=LATENCY_COLS), cfg) == PLACEHOLDER
    rows = list(start)
    for name, base in (("al-s-random-400", 0.42), ("al-s-random-400-r1", 0.43),
                       ("al-s-random-400-r2", 0.41), ("al-s-variety-400", 0.45),
                       ("al-s-variety-400-t1", 0.44), ("al-s-variety-400-t2", 0.47),
                       ("al-s-all", 0.52), ("al-m-variety-800", 0.50)):
        rows += _model_rows(name, base)
    rows += _model_rows(HEADLINE_ENGINE, 0.62)  # AP50 car 0.92, motorbike 0.87
    rows += _model_rows("al-m-all@1280-fp16", 0.64)
    rows += [_delta("al-s-variety-400", "al-s-random-400", "T-VN", "mAP50_95", 0.031),
             _delta("al-s-variety-400", "al-s-random-400", "T-VN", "AP50/motorbike", 0.052),
             _delta("al-s-variety-400", "src_pseudo-yolo11s-s0", "T-VN", "mAP50_95", 0.5),
             _delta("al-m-variety-800", "src_pseudo-yolo11m-s0", "T-VD", "mAP50_95", -0.012)]
    lat = pd.DataFrame([_lat(HEADLINE_ENGINE, 10.6)], columns=LATENCY_COLS)
    block = al_block(_df(rows), lat, cfg, png="reports/al_curve.png")
    assert "![Active-learning curve](reports/al_curve.png)" in block
    assert "car AP50 92.0" in block and "car reaches 90" in block
    assert "motorbike does not reach 90" in block
    # headline table: three engines (+ the starting m engine as reference), CIs and latency
    head = block.split("| engine |")[1].split("\n\n")[0]
    assert f"| {HEADLINE_ENGINE} | 800 |" in head and "| al-m-all@1280-fp16 | 9352 |" in head
    assert "92.0 [89.0, 95.0]" in head and "10.6" in head
    assert "| al-m-random-800@1280-fp16 |" in head  # listed, not evaluated: dashes
    # curve table rows
    assert "| al-s-variety-400 | variety | 400 |" in block and "| al-s-all | all | 9352 |" in block
    assert "| src_pseudo-yolo11s-s0 | start | 0 |" in block
    # paired deltas: configured pairs only, from the delta rows
    assert "| al-s-variety-400 | al-s-random-400 | +3.1 [+2.1, +4.1] | +5.2 [+4.2, +6.2] |" in block
    assert "vs src_pseudo-yolo11s-s0" not in block and "+50.0" not in block
    assert "| al-m-variety-800 | src_pseudo-yolo11m-s0 |" in block and "-1.2 [-2.2, -0.2]" in block
    # seed spread at 400 (T-VN mAP50-95 in points)
    assert "| ranking seeds 0/1/2 | 42.0 / 43.0 / 41.0 | 2.0 |" in block
    assert "| training seeds 0/1/2 | 45.0 / 44.0 / 47.0 | 3.0 |" in block


def _readme(path: Path) -> None:
    path.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                    "<!-- compare:start -->\n<!-- compare:end -->\n"
                    "<!-- al:start -->\n(no active-learning results yet)\n<!-- al:end -->\n",
                    encoding="utf-8")


def test_report_keeps_placeholder_without_al_rows(tmp_path: Path):
    rpath, readme = tmp_path / "results.csv", tmp_path / "README.md"
    append_results(rpath, _model_rows("src_pseudo-yolo11s-s0", 0.40)
                   + _model_rows("teacher-x", 0.5))
    _readme(readme)
    args = ["report", "--results", str(rpath), "--readme", str(readme),
            "--latency", str(tmp_path / "none.csv"), "--al-config", str(ROOT / "configs/al.yaml"),
            "--al-png", str(tmp_path / "al_curve.png")]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    assert "(no active-learning results yet)" in text and "| teacher-x |" in text
    # AL rows fill the block; AL runs, engines and comparisons stay out of the main table and
    # the compare block, the starting students and their comparisons stay in them
    append_results(rpath, _model_rows("al-s-random-100", 0.42)
                   + _model_rows(HEADLINE_ENGINE, 0.62)
                   + [_delta("al-s-disagree-100", "al-s-random-100", "T-VN", "mAP50_95", 0.02),
                      _delta("src_pseudo-yolo11s-s0", "src-yolo11s-s0", "T-VN", "mAP50_95",
                             0.037)])
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    main, rest = text.split("<!-- results:end -->")
    compare, al = rest.split("<!-- compare:end -->")
    assert "(no active-learning results yet)" not in al and "| al-s-random-100 | random |" in al
    assert "car reaches 90" in al and "![Active-learning curve]" not in al
    assert "| al-s-disagree-100 | al-s-random-100 | +2.0" in al
    assert HEADLINE_ENGINE not in main and "al-s-" not in main
    assert "| src_pseudo-yolo11s-s0 |" in main
    assert "al-s-" not in compare and "| src_pseudo-yolo11s-s0@1280 | vs src-yolo11s-s0@1280 |" in compare
    (tmp_path / "al_curve.png").write_bytes(b"\x89PNG")
    assert CliRunner().invoke(app, args).exit_code == 0
    assert "![Active-learning curve](" in readme.read_text(encoding="utf-8")
    # a README without al markers is left alone
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
    assert CliRunner().invoke(app, args).exit_code == 0


def test_existing_results_page_unchanged(tmp_path: Path, monkeypatch):
    """`cvt report` on the committed results reproduces the committed docs/results.md byte for
    byte (every generated block unchanged; image links relative to docs/)."""
    readme = tmp_path / "docs" / "results.md"
    readme.parent.mkdir()
    shutil.copy(ROOT / "docs" / "results.md", readme)
    (tmp_path / "reports").mkdir()
    for name in ("pareto.png", "al_curve.png", "latency_tiles.csv", "analytics_clips.csv"):
        shutil.copy(ROOT / "reports" / name, tmp_path / "reports" / name)
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "tiles.yaml", tmp_path / "configs" / "tiles.yaml")
    monkeypatch.chdir(tmp_path)  # the page's repository root, as when run from the repo
    r = CliRunner().invoke(app, [
        "report", "--results", str(ROOT / "reports/results.csv"), "--readme", str(readme),
        "--latency", str(ROOT / "reports/latency.csv"),
        "--deploy-config", str(ROOT / "configs/deploy.yaml"),
        "--pareto-png", "reports/pareto.png", "--al-config", str(ROOT / "configs/al.yaml"),
        "--al-png", "reports/al_curve.png"])
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    assert "<!-- al:start -->" in text
    assert text == (ROOT / "docs" / "results.md").read_text(encoding="utf-8")


# --- CLI helpers for scripts/al.ps1 -------------------------------------------------------------

def test_al_list_comparisons_and_report(tmp_path: Path):
    models = tmp_path / "models_al.yaml"
    r = CliRunner().invoke(app, ["al", "list", "--models", str(models)])
    assert r.exit_code != 0 and "cvt al prepare" in r.output
    models.write_text("# generated\n" + yaml.safe_dump({"al-s-random-100": {"arch": "yolo"},
                                                       "al-m-all": {"arch": "yolo"}},
                                                      sort_keys=False), encoding="utf-8")
    r = CliRunner().invoke(app, ["al", "list", "--models", str(models)])
    assert r.exit_code == 0, r.output
    assert r.stdout.split() == ["al-s-random-100", "al-m-all"]
    r = CliRunner().invoke(app, ["al", "comparisons", "--config", str(ROOT / "configs/al.yaml")])
    assert r.exit_code == 0, r.output
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 10 and lines[0] == "al-s-disagree-100 al-s-random-100"
    assert lines[-1] == "al-m-variety-800 src_pseudo-yolo11m-s0"
    # `cvt al report`: curve csv + png; loud without anything to plot
    rpath = tmp_path / "results.csv"
    append_results(rpath, _model_rows("src_pseudo-yolo11s-s0", 0.40))
    args = ["al", "report", "--config", str(ROOT / "configs/al.yaml"), "--results", str(rpath),
            "--out-csv", str(tmp_path / "al_curve.csv"), "--out-png", str(tmp_path / "al.png")]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert len(pd.read_csv(tmp_path / "al_curve.csv")) == 22
    assert (tmp_path / "al.png").read_bytes()[:4] == b"\x89PNG"
    empty = tmp_path / "empty.csv"
    append_results(empty, _model_rows("teacher-x", 0.5))
    r = CliRunner().invoke(app, [*args[:5], str(empty), *args[6:]])
    assert r.exit_code != 0 and "nothing to plot" in r.output


def test_train_done_group(tmp_path: Path, monkeypatch):
    """`cvt train done --group g` (scripts/al.ps1 -Step train): exit 0 iff every job is done."""
    q = tmp_path / "queue.yaml"
    jobs = [{"name": n, "group": g, "model": "m.pt", "data": "d.yaml", "project": "p", "run": n}
            for n, g in (("a1", "ga"), ("a2", "ga"), ("b1", "gb"))]
    q.write_text(yaml.safe_dump({"jobs": jobs}), encoding="utf-8")
    states = {"a1": "done", "a2": "paused", "b1": "done"}
    monkeypatch.setattr("cvt.train.queue.job_state", lambda j, log=None: (states[j.name], None))
    run = CliRunner().invoke
    r = run(app, ["train", "done", "--group", "gb", "--queue", str(q)])
    assert r.exit_code == 0 and "b1: done" in r.stdout
    r = run(app, ["train", "done", "--group", "ga", "--queue", str(q)])
    assert r.exit_code == 1 and "a2: paused" in r.stdout
    assert run(app, ["train", "done", "--group", "nope", "--queue", str(q)]).exit_code == 2
    assert run(app, ["train", "done", "--queue", str(q)]).exit_code == 2  # neither given
    assert run(app, ["train", "done", "--name", "b1", "--queue", str(q)]).exit_code == 0
    # errors (missing queue, unreadable checkpoint) exit 2, never 1 ("not finished")
    r = run(app, ["train", "done", "--group", "ga", "--queue", str(tmp_path / "none.yaml")])
    assert r.exit_code == 2 and "error" in r.output

    def boom(j, log=None):
        raise RuntimeError("torn checkpoint")

    monkeypatch.setattr("cvt.train.queue.job_state", boom)
    r = run(app, ["train", "done", "--group", "ga", "--queue", str(q)])
    assert r.exit_code == 2 and "torn checkpoint" in r.output


# --- secondary fallbacks (pre-declared 2026-11-03) ----------------------------------------------

FALLBACKS = ("al-m-variety-800@1536-fp16", "al-m-variety-800@1280+tiles640-fp16")


def test_fallback_grid_and_registry(tmp_path: Path):
    from cvt.al.report import fallback_points
    from cvt.deploy.engines import grid_points, grid_tile, registry_entries, registry_name

    al = _cfg()
    cfg = load_yaml(ROOT / al["fallbacks"]["config"])
    fbs = fallback_points(al)
    assert [(f["name"], f["imgsz"]) for f in fbs] == [(FALLBACKS[0], 1536), (FALLBACKS[1], 1280)]
    assert cfg["students"] == [al["headline"]] and cfg["precisions"] == ["fp16"]
    assert cfg["tiled"] == load_yaml(ROOT / "configs/deploy.yaml")["tiled"]  # deployment-grid tiling
    names = {registry_name(p, grid_tile(cfg)): p.imgsz for p in grid_points(cfg)}
    assert all(names[f["name"]] == f["imgsz"] for f in fbs)  # every fallback is a grid point
    entries = registry_entries(cfg, _al_models(tmp_path, cfg["students"]), tmp_path / "engines")
    tiled = entries[FALLBACKS[1]]
    assert tiled["arch"] == "tiled" and tiled["base"] == HEADLINE_ENGINE
    assert tiled["tile_base"] == "al-m-variety-800@640-fp16" and tiled["tile"] == 640
    r = CliRunner().invoke(app, ["al", "fallbacks"])
    assert r.exit_code == 0, r.output
    assert r.stdout.strip().splitlines() == [f"{FALLBACKS[0]} 1536", f"{FALLBACKS[1]} 1280"]


def test_fallbacks_secondary_in_sentence_and_table():
    from cvt.al.report import al_block, headline_sentence

    cfg = _cfg()
    head = _headline_rows((0.912, 0.85, 0.95), (0.884, 0.84, 0.93))
    fb = (_model_rows(FALLBACKS[0], 0.60, imgsz=1536)  # motorbike 0.60 + 0.25 = 85.0
          + _headline_rows((0.95, 0.9, 0.97), (0.915, 0.9, 0.93), model=FALLBACKS[1]))
    # fallbacks alone never stand in for the headline
    assert "not evaluated yet" in headline_sentence(_df(fb), cfg)
    s = headline_sentence(_df(head + fb), cfg)
    assert "motorbike does not reach 90 (" in s  # headline verdict unchanged by a fallback at 91.5
    assert "Secondary, pre-declared fallbacks" in s
    assert "tiled (1280 + 640-px tiles): car 95.0 [90.0, 97.0] (reaches 90), motorbike 91.5" in s
    assert "1536 px input: car 90.0" in s  # 1536 rows read at imgsz 1536
    # a fallback evaluated at the wrong size is not reported
    wrong = _model_rows(FALLBACKS[0], 0.70, imgsz=1280)
    assert "1536 px input" not in headline_sentence(_df(head + wrong), cfg)
    block = al_block(_df(head + fb), pd.DataFrame(columns=LATENCY_COLS), cfg)
    lines = [ln for ln in block.splitlines() if ln.startswith("| al-m-variety-800@")]
    assert [ln.split(" | ")[0][2:] for ln in lines] == [HEADLINE_ENGINE, *FALLBACKS]
    assert lines[1].split(" | ")[1] == "800" and "never the headline" in block


def test_main_readme_tables_exclude_fallbacks(tmp_path: Path):
    rpath, readme = tmp_path / "results.csv", tmp_path / "README.md"
    append_results(rpath, _model_rows("src_pseudo-yolo11s-s0", 0.40)
                   + _model_rows(HEADLINE_ENGINE, 0.62)
                   + _model_rows(FALLBACKS[1], 0.61) + _model_rows(FALLBACKS[0], 0.63, imgsz=1536))
    _readme(readme)
    r = CliRunner().invoke(app, [
        "report", "--results", str(rpath), "--readme", str(readme),
        "--latency", str(tmp_path / "none.csv"), "--al-config", str(ROOT / "configs/al.yaml"),
        "--al-png", str(tmp_path / "al_curve.png")])
    assert r.exit_code == 0, r.output
    main, rest = readme.read_text(encoding="utf-8").split("<!-- results:end -->")
    assert "al-m-variety-800@" not in main and "| src_pseudo-yolo11s-s0 |" in main
    assert f"| {FALLBACKS[0]} | 800 |" in rest and f"| {FALLBACKS[1]} | 800 |" in rest
