"""`cvt al prepare`: rankings, oracle reveals, revealed-label exports, run yamls, queue, registry."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import cvt.al.prepare as prep
from cvt.al.oracle import Oracle
from cvt.cli import app
from cvt.config import load_yaml
from cvt.data.manifest import build_manifest, load_manifest, save_manifest
from cvt.eval.predictions import PRED_COLS
from cvt.pseudo.label import save_pseudo
from cvt.train.queue import data_sha256, load_queue
from tests.helpers import make_uitdrone7, make_visdrone
from tests.test_cli import _flat, _setup, _thr

REPO = Path(__file__).resolve().parents[1]
AL_YAML = REPO / "configs" / "al.yaml"


def _manifest(tmp_path: Path, n_train: int = 8) -> Path:
    make_visdrone(tmp_path / "vd" / "train", n_images=2, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=1, seed0=100)
    make_uitdrone7(tmp_path / "uit", n_train=n_train)
    cfg = {"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd/train"), "val_dir": str(tmp_path / "vd/val"),
                         "rider_merge": True, "tune_frac": 0.4},
        "uitdrone7": {"images_dir": str(tmp_path / "uit/images"),
                      "labels_dir": str(tmp_path / "uit/annotations")},
    }}
    images, boxes = build_manifest(cfg, exclusions=set())
    save_manifest(images, boxes, tmp_path / "manifest")
    return tmp_path / "manifest"


def _scores(path: Path, ids: list[str], seed: int, mhash: str = "h", sets=("UD-pool",),
            nan: bool = False) -> Path:
    d = np.random.default_rng(seed).random(len(ids))
    if nan:
        d[0] = np.nan
    frames = pd.DataFrame({"image_id": ids, "n_boxes": 0, "disagreement": d})
    save_pseudo(pd.DataFrame(columns=PRED_COLS), frames,
                {"sets": list(sets), "manifest_hash": mhash}, path)
    return path


def _cfg(tmp_path: Path, scores: Path, runs: list[dict] | None = None, budgets=(2, 4)) -> dict:
    """The real configs/al.yaml with test inputs (scores, lists) and optionally a small inventory."""
    cfg = copy.deepcopy(load_yaml(AL_YAML))
    lists = {}
    for name in ("VD-train", "P-train-teacher-fused", "VD-tune", "P-val-teacher-fused"):
        lists[name] = tmp_path / "yolo" / f"{name}.txt"
        lists[name].parent.mkdir(parents=True, exist_ok=True)
        lists[name].write_text(f"/x/{name}/a.jpg\n", encoding="utf-8")
    cfg.update(scores=str(scores), plan2_queue=str(REPO / "configs" / "train_queue.yaml"),
               base_train=[str(lists["VD-train"]), str(lists["P-train-teacher-fused"])],
               val=[str(lists["VD-tune"]), str(lists["P-val-teacher-fused"])])
    if runs is not None:
        cfg.update(runs=runs, budgets=list(budgets), comparisons=[])
    return cfg


SMALL = [
    {"name": "al-s-random-2", "group": "al_curve", "student": "s", "strategy": "random", "budget": 2},
    {"name": "al-s-disagree-2", "group": "al_curve", "student": "s", "strategy": "disagree", "budget": 2},
    {"name": "al-s-disagree-4", "group": "al_curve", "student": "s", "strategy": "disagree", "budget": 4},
    {"name": "al-s-variety-4", "group": "al_curve", "student": "s", "strategy": "variety", "budget": 4},
    {"name": "al-s-random-2-r1", "group": "al_variance", "student": "s", "strategy": "random",
     "budget": 2, "ranking_seed": 1},
    {"name": "al-s-variety-4-t1", "group": "al_variance", "student": "s", "strategy": "variety",
     "budget": 4, "seed": 1},
    {"name": "al-m-variety-4", "group": "al_headline", "student": "m", "strategy": "variety", "budget": 4},
]
ALL_RUN = {"name": "al-s-all", "group": "al_upper", "student": "s", "strategy": "all"}


def _stems(lst: Path) -> list[str]:
    return [Path(p).stem for p in lst.read_text(encoding="utf-8").split()]


def _stem(image_id: str) -> str:
    return image_id.replace("/", "__")


def _picked(rankings: pd.DataFrame, col: str, budget: int) -> list[str]:
    r = rankings[rankings[col] <= budget].sort_values(col)
    return list(r.image_id)


def _log(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_run_lists_and_repeat(tmp_path: Path):
    mdir = _manifest(tmp_path)
    images, _, _ = load_manifest(mdir)
    pool = sorted(images[images.set_name == "UD-pool"].image_id)
    scores = _scores(tmp_path / "scores", pool, seed=0, mhash=load_manifest(mdir)[2])
    cfg = _cfg(tmp_path, scores, [*SMALL, ALL_RUN])
    out = tmp_path / "out"
    runs = prep.prepare_runs(cfg, mdir, out)
    assert [r.name for r in runs] == [r["name"] for r in [*SMALL, ALL_RUN]]
    rankings = pd.read_parquet(out / "reports" / "al_rankings.parquet")
    base = [Path(p).resolve().as_posix() for p in cfg["base_train"]]
    val = [Path(p).resolve().as_posix() for p in cfg["val"]]
    for r in runs:
        data = yaml.safe_load((out / "configs" / "train" / "al" / f"{r.name}.yaml").read_text())
        assert data["val"] == val
        assert data["train"][:2] == base
        picks = data["train"][2:]
        assert len(set(picks)) == 1
        lst = Path(picks[0])
        if r.strategy == "all":
            assert len(picks) == 1 and lst.stem == "all"
            assert sorted(_stems(lst)) == sorted(_stem(i) for i in pool)
        else:
            assert len(picks) == 5
            col = {"random": f"rank_random_s{r.ranking_seed}", "disagree": "rank_disagree",
                   "variety": "rank_variety"}[r.strategy]
            assert _stems(lst) == [_stem(i) for i in _picked(rankings, col, r.budget)]
    by = {r.name: r for r in runs}
    assert by["al-s-variety-4-t1"].picks == by["al-s-variety-4"].picks == by["al-m-variety-4"].picks
    assert by["al-s-random-2-r1"].picks != by["al-s-random-2"].picks

    jobs = {j.name: j for j in load_queue(out / "configs" / "train_queue_al.yaml")}
    h = {n: data_sha256(Path(jobs[n].data)) for n in ("al-s-disagree-4", "al-s-random-2")}
    old = _stems(by["al-s-disagree-4"].picks)
    _scores(scores, pool, seed=1, mhash=load_manifest(mdir)[2])  # new disagreement scores: the disagree picks change
    runs = prep.prepare_runs(cfg, mdir, out)
    assert _stems({r.name: r for r in runs}["al-s-disagree-4"].picks) != old
    assert data_sha256(Path(jobs["al-s-disagree-4"].data)) != h["al-s-disagree-4"]
    assert data_sha256(Path(jobs["al-s-random-2"].data)) == h["al-s-random-2"]  # random: no scores


def test_prepare_reveals_only_picked_ids(tmp_path: Path):
    mdir = _manifest(tmp_path)
    images, _, _ = load_manifest(mdir)
    pool = sorted(images[images.set_name == "UD-pool"].image_id)
    cfg = _cfg(tmp_path, _scores(tmp_path / "scores", pool, seed=0, mhash=load_manifest(mdir)[2]), SMALL)
    out = tmp_path / "out"
    prep.prepare_runs(cfg, mdir, out)
    rankings = pd.read_parquet(out / "reports" / "al_rankings.parquet")
    assert list(rankings.columns) == ["image_id", "clip", "frame", "disagreement", "rank_random_s0",
                                      "rank_random_s1", "rank_disagree", "rank_variety"]
    assert list(rankings.image_id) == pool
    for c in rankings.columns[4:]:
        assert sorted(rankings[c]) == list(range(1, len(pool) + 1))
    sets = {("random-s0", 2): _picked(rankings, "rank_random_s0", 2),
            ("random-s1", 2): _picked(rankings, "rank_random_s1", 2),
            ("disagree", 2): _picked(rankings, "rank_disagree", 2),
            ("disagree", 4): _picked(rankings, "rank_disagree", 4),
            ("variety", 4): _picked(rankings, "rank_variety", 4)}
    log = _log(out / "reports" / "al_reveals.csv")
    got: dict[tuple[str, int], list[str]] = {}
    for row in log:
        got.setdefault((row["strategy"], int(row["budget"])), []).append(row["image_id"])
    assert got == sets  # one reveal per distinct pick set (variety-4 shared by three runs)
    union = set().union(*sets.values())
    assert {row["image_id"] for row in log} == union and len(union) < len(pool)
    exported = {p.name for p in (out / "data" / "yolo" / "al").iterdir() if p.is_dir()}
    assert exported == {"random-s0-2", "random-s1-2", "disagree-2", "disagree-4", "variety-4"}
    labelled = {p.stem for d in (out / "data" / "yolo" / "al").iterdir() if d.is_dir()
                for p in (d / "labels").glob("*.txt")}
    assert labelled == {_stem(i) for i in union}

    # the all-pool runs reveal the whole pool once, budget = pool size
    cfg = _cfg(tmp_path, Path(cfg["scores"]), [ALL_RUN, {**ALL_RUN, "name": "al-m-all", "student": "m",
                                                          "group": "al_headline"}])
    prep.prepare_runs(cfg, mdir, tmp_path / "out2")
    log = _log(tmp_path / "out2" / "reports" / "al_reveals.csv")
    assert [r["image_id"] for r in log] == pool
    assert {(r["strategy"], int(r["budget"])) for r in log} == {("all", len(pool))}


EXPECTED_RUNS = {
    **{f"al-s-{s}-{b}": ("al_curve", "s", 0)
       for s in ("random", "disagree", "variety") for b in (100, 200, 400, 800)},
    "al-s-random-400-r1": ("al_variance", "s", 0), "al-s-random-400-r2": ("al_variance", "s", 0),
    "al-s-variety-400-t1": ("al_variance", "s", 1), "al-s-variety-400-t2": ("al_variance", "s", 2),
    "al-s-all": ("al_upper", "s", 0),
    "al-m-variety-800": ("al_headline", "m", 0), "al-m-random-800": ("al_headline", "m", 0),
    "al-m-all": ("al_headline", "m", 0),
}


def _fake_pool(n_clips: int = 16, n_frames: int = 63) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [(f"C{c:02d}", f) for c in range(n_clips) for f in range(1, n_frames + 1)]
    images = pd.DataFrame({
        "image_id": [f"uitdrone7/train/{c}_{f}" for c, f in ids], "path": "x.jpg", "width": 64,
        "height": 48, "source": "uitdrone7", "scene_id": [f"uitdrone7/{c}" for c, _ in ids],
        "split": "train", "set_name": "UD-pool"})
    boxes = pd.DataFrame(columns=["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"])
    return images, boxes


def test_queue_and_registry_cover_the_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_yaml(AL_YAML)
    assert cfg["pool_set"] == "UD-pool" and cfg["scores"] == "data/pseudo/teacher-fused-udpool"
    assert cfg["budgets"] == [100, 200, 400, 800] and cfg["repeat"] == 5
    assert cfg["strategies"] == {"random": {"seed": 0}, "disagree": {}, "variety": {"gaps": [15, 5, 0]}}
    assert cfg["base_train"] == ["data/yolo/VD-train.txt", "data/yolo/P-train-teacher-fused.txt"]
    assert cfg["val"] == ["data/yolo/VD-tune.txt", "data/yolo/P-val-teacher-fused.txt"]
    init = {k: f"runs/students/src_pseudo-yolo11{k}-s0/weights/best.pt" for k in "sm"}
    assert {k: (v["init"], v["batch"]) for k, v in cfg["students"].items()} == {
        "s": (init["s"], 8), "m": (init["m"], 4)}

    images, boxes = _fake_pool()
    monkeypatch.setattr(prep, "oracle_from_manifest",
                        lambda mdir, log_path: Oracle(images, boxes, log_path, "h"))

    def fake_export(revealed, name, out_root):
        lst = Path(out_root) / f"{name}.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("".join(f"/x/{i}.jpg\n" for i in revealed.images.image_id), encoding="utf-8")
        return lst

    monkeypatch.setattr(prep, "export_revealed", fake_export)
    pool = sorted(images.image_id)
    out = tmp_path / "out"
    runs = prep.prepare_runs(_cfg(tmp_path, _scores(tmp_path / "scores", pool, 0)), Path("unused"), out)
    assert len(runs) == 20 and {r.name for r in runs} == set(EXPECTED_RUNS)

    jobs = load_queue(out / "configs" / "train_queue_al.yaml")
    assert [j.name for j in jobs] == [r.name for r in runs]
    groups = ["al_curve", "al_variance", "al_upper", "al_headline"]
    assert [j.group for j in jobs] == sorted((j.group for j in jobs), key=groups.index)
    for j in jobs:
        group, size, seed = EXPECTED_RUNS[j.name]
        assert (j.group, j.arch, j.run, j.model) == (group, "yolo", j.name, init[size])
        assert j.project == (out / "runs" / "al").as_posix()
        assert Path(j.data) == out / "configs" / "train" / "al" / f"{j.name}.yaml"
        assert Path(j.data).exists()
        assert j.args == {"optimizer": "AdamW", "lr0": 0.0005, "warmup_bias_lr": 0.0, "imgsz": 1280,
                          "epochs": 15, "batch": {"s": 8, "m": 4}[size], "patience": 0, "cos_lr": True,
                          "close_mosaic": 3, "workers": 4, "seed": seed, "warmup_epochs": 1,
                          "val": False}
    reg = load_yaml(out / "configs" / "models_al.yaml")
    assert list(reg) == [j.name for j in jobs]
    for name, e in reg.items():
        assert e == {"arch": "yolo", "names_map": "cvt",
                     "weights": (out / "runs" / "al" / name / "weights" / "last.pt").as_posix()}

    # 15 distinct pick sets, each revealed once; nested budgets re-log their ids
    log = _log(out / "reports" / "al_reveals.csv")
    sizes: dict[tuple[str, int], int] = {}
    for row in log:
        key = (row["strategy"], int(row["budget"]))
        sizes[key] = sizes.get(key, 0) + 1
    assert sizes == {**{(s, b): b for s in ("random-s0", "disagree", "variety")
                        for b in (100, 200, 400, 800)},
                     ("random-s1", 400): 400, ("random-s2", 400): 400, ("all", len(pool)): len(pool)}

    models = load_yaml(REPO / "configs" / "models.yaml")
    pairs = [(c["a"], c["b"]) for c in cfg["comparisons"]]
    expected = [(f"al-s-{s}-{b}", f"al-s-random-{b}") for b in (100, 200, 400, 800)
                for s in ("disagree", "variety")]
    assert pairs == [*expected, ("al-m-variety-800", "al-m-random-800"),
                     ("al-m-variety-800", "src_pseudo-yolo11m-s0")]
    assert all(n in reg or n in models for p in pairs for n in p)
    assert cfg["compare_sets"] == ["T-VN", "T-VD"] and cfg["headline"] == "al-m-variety-800"


def test_prepare_refuses_without_scores(tmp_path: Path):
    mdir = _manifest(tmp_path, n_train=3)
    scores = tmp_path / "scores"
    cfg = _cfg(tmp_path, scores, SMALL[:1], budgets=(2,))
    path = tmp_path / "al.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"
    args = ["al", "prepare", "--config", str(path), "--manifest", str(mdir), "--out", str(out)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1
    assert "scripts\\al.ps1 -Step score" in _flat(r.output)
    assert not out.exists()  # nothing revealed, nothing written

    images, _, mh = load_manifest(mdir)
    _scores(scores, sorted(images[images.set_name == "UD-pool"].image_id), 0, mh)
    (scores / "meta.json").unlink()  # an interrupted scoring run
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1 and "scripts\\al.ps1 -Step score" in _flat(r.output)
    assert not out.exists()

    _scores(scores, sorted(images[images.set_name == "UD-pool"].image_id)[1:], 0, mh)  # misses a frame
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1 and "1 pool frames have no score" in _flat(r.output)
    assert not out.exists()

    _scores(scores, sorted(images[images.set_name == "UD-pool"].image_id), 0, mh)
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert (out / "configs" / "train_queue_al.yaml").exists()


def test_prepare_refuses_stale_or_wrong_scores(tmp_path: Path):
    mdir = _manifest(tmp_path, n_train=3)
    images, _, mh = load_manifest(mdir)
    pool = sorted(images[images.set_name == "UD-pool"].image_id)
    cfg = _cfg(tmp_path, tmp_path / "scores", SMALL[:1], budgets=(2,))
    out = tmp_path / "out"
    rerun = r"scripts\al.ps1 -Step score"

    _scores(tmp_path / "scores", pool, 0, "some-other-hash")  # scored against another manifest
    with pytest.raises(ValueError, match="manifest") as e:
        prep.prepare_runs(cfg, mdir, out)
    assert rerun in str(e.value) and "some-other-hash" in str(e.value) and mh in str(e.value)

    _scores(tmp_path / "scores", pool, 0, mh, sets=("UD-other",))  # not the pool
    with pytest.raises(ValueError, match="UD-pool") as e:
        prep.prepare_runs(cfg, mdir, out)
    assert rerun in str(e.value)

    _scores(tmp_path / "scores", pool, 0, mh, nan=True)  # a NaN disagreement
    with pytest.raises(ValueError, match="NaN") as e:
        prep.prepare_runs(cfg, mdir, out)
    assert rerun in str(e.value)
    assert not out.exists()  # nothing revealed, nothing written

    _scores(tmp_path / "scores", pool, 0, mh)
    assert len(prep.prepare_runs(cfg, mdir, out)) == 1


def test_pseudolabel_accepts_pool_role(tmp_path: Path):
    p = _setup(tmp_path)
    p["models"].write_text(yaml.safe_dump({
        "oracle": {"arch": "oracle"}, "fused": {"arch": "fused", "members": ["oracle", "oracle"]},
    }))
    thr, members = tmp_path / "thr" / "fused.yaml", tmp_path / "thr" / "members"
    _thr(p, "fused", thr)
    _thr(p, "oracle", members / "oracle.yaml")
    out = tmp_path / "pseudo" / "udpool"

    def run(sets: str):
        return CliRunner().invoke(app, [
            "pseudolabel", "--model", "fused", "--sets", sets, "--thresholds", str(thr),
            "--member-thresholds-dir", str(members), "--out", str(out), "--models", str(p["models"]),
            "--manifest", str(p["manifest"]), "--hash-cache", str(p["hash"]),
            "--pred-cache", str(p["preds"])])

    r = run("UD-pool")
    assert r.exit_code == 0, r.output
    images, _, _ = load_manifest(p["manifest"])
    frames = pd.read_parquet(out / "frames.parquet")
    assert sorted(frames.image_id) == sorted(images[images.set_name == "UD-pool"].image_id)
    assert np.isfinite(frames.disagreement).all()
    for test_set in ("T-VN", "T-VD", "UD-pool,T-VN"):
        r = run(test_set)
        assert r.exit_code == 2 and "test sets" in _flat(r.output)
