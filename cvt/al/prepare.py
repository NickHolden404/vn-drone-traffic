"""`cvt al prepare`: pool rankings -> oracle reveals -> revealed-label exports, one dataset yaml
per run, the AL training queue and the AL model registry (configs/al.yaml holds every value).

Pool metadata comes from `Oracle.pool_frames()` and labels only from `Oracle.reveal`: one reveal
per distinct pick set (strategy ranking, budget), logged to reports/al_reveals.csv. All outputs
are written under `out` at the paths below (`out` = the repo root for the real runs)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from cvt.al.oracle import POOL_SET, oracle_from_manifest
from cvt.al.strategies import budget_picks, disagreement_ranking, random_ranking, variety_ranking
from cvt.config import load_yaml
from cvt.data.export_yolo import export_revealed, write_dataset_yaml
from cvt.utils import write_parquet_atomic

RANKINGS = Path("reports/al_rankings.parquet")
REVEALS = Path("reports/al_reveals.csv")
EXPORTS = Path("data/yolo/al")
TRAIN_YAMLS = Path("configs/train/al")
QUEUE = Path("configs/train_queue_al.yaml")
MODELS = Path("configs/models_al.yaml")
PROJECT = Path("runs/al")
STRATEGIES = ("random", "disagree", "variety", "all")
SCORE_HINT = r"run `scripts\al.ps1 -Step score` first"
SCORE_HINT_RERUN = r"rerun `scripts\al.ps1 -Step score`"


@dataclass(frozen=True)
class Run:
    name: str
    group: str
    student: str
    strategy: str  # random | disagree | variety | all
    budget: int | None  # None for all-pool runs
    ranking_seed: int | None  # random runs only
    seed: int  # training seed
    pick_set: str  # export name, e.g. random-s0-400, variety-800, all
    picks: Path  # the pick set's image list
    data: Path  # dataset yaml


def _parse_runs(cfg: dict) -> list[dict]:
    """Validated run specs; a run's name must spell out its student, strategy, budget and seeds."""
    budgets, students = list(cfg["budgets"]), cfg["students"]
    rand_seed = int(cfg["strategies"]["random"]["seed"])
    specs, names = [], set()
    for r in cfg["runs"]:
        name, strategy, student = r["name"], r["strategy"], r["student"]
        if strategy not in STRATEGIES:
            raise ValueError(f"{name}: unknown strategy {strategy!r} (one of {STRATEGIES})")
        if student not in students:
            raise ValueError(f"{name}: unknown student {student!r} (one of {sorted(students)})")
        budget = r.get("budget")
        if (strategy == "all") != (budget is None) or (budget is not None and budget not in budgets):
            raise ValueError(f"{name}: budget {budget!r} must be one of {budgets} (none for `all`)")
        if "ranking_seed" in r and strategy != "random":
            raise ValueError(f"{name}: ranking_seed is for random runs only")
        rs = int(r.get("ranking_seed", rand_seed)) if strategy == "random" else None
        seed = int(r.get("seed", 0))
        want = f"al-{student}-{strategy}" + (f"-{budget}" if budget is not None else "")
        want += (f"-r{rs}" if rs not in (None, rand_seed) else "") + (f"-t{seed}" if seed else "")
        if name != want:
            raise ValueError(f"run name {name!r} does not match its fields (expected {want!r})")
        if name in names:
            raise ValueError(f"duplicate run name {name!r}")
        names.add(name)
        key = "all" if strategy == "all" else (f"random-s{rs}" if strategy == "random" else strategy)
        specs.append({"name": name, "group": r["group"], "student": student, "strategy": strategy,
                      "budget": budget, "ranking_seed": rs, "seed": seed, "ranking": key})
    return specs


def _load_scores(scores_dir: Path, pool_ids: list[str]) -> pd.DataFrame:
    frames = pd.read_parquet(scores_dir / "frames.parquet", columns=["image_id", "disagreement"])
    have = set(frames.image_id)
    missing = [i for i in pool_ids if i not in have]
    if missing:
        raise ValueError(f"{len(missing)} pool frames have no score in {scores_dir} "
                         f"(e.g. {missing[:3]}): {SCORE_HINT}")
    frames = frames[frames.image_id.isin(set(pool_ids))].reset_index(drop=True)
    if frames.disagreement.isna().any():
        raise ValueError(f"{int(frames.disagreement.isna().sum())} pool frames have a NaN "
                         f"disagreement in {scores_dir}: {SCORE_HINT_RERUN}")
    return frames


def _check_scores_meta(scores_dir: Path, manifest_hash: str) -> None:
    """The scores must be from this manifest and cover the pool (a stale or other-set run is not)."""
    meta = json.loads((scores_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("manifest_hash") != manifest_hash:
        raise ValueError(f"{scores_dir} was scored against manifest {meta.get('manifest_hash')}, "
                         f"current is {manifest_hash}: {SCORE_HINT_RERUN}")
    if POOL_SET not in (meta.get("sets") or []):
        raise ValueError(f"{scores_dir} scored {meta.get('sets')}, not {POOL_SET}: {SCORE_HINT_RERUN}")


def _job_args(cfg: dict) -> dict[str, dict]:
    """Per student: the original job's args overridden by the AL recipe and the student's batch."""
    plan2 = {j["name"]: j for j in load_yaml(cfg["plan2_queue"])["jobs"]}
    out = {}
    for size, st in cfg["students"].items():
        if st["plan2_job"] not in plan2:
            raise ValueError(f"student {size}: {st['plan2_job']} not in {cfg['plan2_queue']}")
        base = dict(plan2[st["plan2_job"]].get("args") or {})
        if base.get("batch") != st["batch"]:
            raise ValueError(f"student {size}: batch {st['batch']} differs from the original job's "
                             f"{base.get('batch')} ({st['plan2_job']})")
        out[size] = {**base, **cfg["recipe"], "batch": int(st["batch"])}
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _posix(path: Path) -> str:
    return Path(path).as_posix()


def prepare_runs(cfg: dict, manifest_dir: Path, out: Path) -> list[Run]:
    """Rankings, reveals, exports, run yamls, queue and registry of the run inventory in `cfg`."""
    out = Path(out)
    if cfg["pool_set"] != POOL_SET:
        raise ValueError(f"pool_set must be {POOL_SET}, got {cfg['pool_set']!r}")
    scores_dir = Path(cfg["scores"])
    for f in ("frames.parquet", "meta.json"):
        if not (scores_dir / f).exists():
            raise FileNotFoundError(f"{scores_dir / f} missing (no or interrupted scoring run): "
                                    f"{SCORE_HINT}")
    specs = _parse_runs(cfg)
    args = _job_args(cfg)
    lists = [Path(p) for p in [*cfg["base_train"], *cfg["val"]]]
    missing = [str(p) for p in lists if not p.exists()]
    if missing:
        raise FileNotFoundError(f"image lists missing: {missing}")
    repeat = int(cfg["repeat"])
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    oracle = oracle_from_manifest(Path(manifest_dir), log_path=out / REVEALS)
    _check_scores_meta(scores_dir, oracle.manifest_hash)
    pool = oracle.pool_frames()
    pool_ids = list(pool.image_id)
    scores = _load_scores(scores_dir, pool_ids)

    # rankings, built once
    rand_seeds = sorted({s["ranking_seed"] for s in specs if s["strategy"] == "random"}
                        | {int(cfg["strategies"]["random"]["seed"])})
    clip_of = dict(zip(pool.image_id, pool["clip"], strict=True))
    frame_of = dict(zip(pool.image_id, pool["frame"].astype(int), strict=True))
    rankings = {f"random-s{s}": random_ranking(pool_ids, s) for s in rand_seeds}
    rankings["disagree"] = disagreement_ranking(scores)
    rankings["variety"] = variety_ranking(scores, clip_of, frame_of,
                                          gaps=cfg["strategies"]["variety"]["gaps"])
    table = pool[["image_id", "clip", "frame"]].merge(scores, on="image_id", how="left")
    for key, ranking in rankings.items():
        rank = {i: k + 1 for k, i in enumerate(ranking)}
        table[f"rank_{key.replace('-', '_')}"] = table.image_id.map(rank).astype("int64")
    write_parquet_atomic(table.sort_values("image_id").reset_index(drop=True), out / RANKINGS)

    # one reveal + export per distinct pick set, in inventory order
    pick_lists: dict[str, Path] = {}
    for s in specs:
        pid = "all" if s["strategy"] == "all" else f"{s['ranking']}-{s['budget']}"
        s["pick_set"] = pid
        if pid in pick_lists:
            continue
        if s["strategy"] == "all":
            ids, budget = pool_ids, len(pool_ids)
        else:
            ids, budget = budget_picks(rankings[s["ranking"]], s["budget"]), s["budget"]
        revealed = oracle.reveal(ids, strategy=s["ranking"], budget=budget)
        pick_lists[pid] = export_revealed(revealed, pid, out / EXPORTS)

    runs, jobs, registry = [], [], {}
    for s in specs:
        picks = pick_lists[s["pick_set"]]
        k = 1 if s["strategy"] == "all" else repeat
        data = write_dataset_yaml(out / TRAIN_YAMLS / f"{s['name']}.yaml",
                                  train=[*cfg["base_train"], *[picks] * k], val=list(cfg["val"]))
        run = Run(s["name"], s["group"], s["student"], s["strategy"], s["budget"],
                  s["ranking_seed"], s["seed"], s["pick_set"], picks, data)
        runs.append(run)
        jobs.append({"name": run.name, "group": run.group,
                     "model": str(cfg["students"][run.student]["init"]),
                     "data": _posix(run.data), "project": _posix(out / PROJECT), "run": run.name,
                     "arch": "yolo", "args": {**args[run.student], "seed": run.seed}})
        registry[run.name] = {"arch": "yolo",
                              "weights": _posix(out / PROJECT / run.name / "weights" / "last.pt"),
                              "names_map": "cvt"}
    header = "# Generated by `cvt al prepare` from configs/al.yaml; do not edit.\n"
    _write_text(out / QUEUE, header + "# Run with `cvt train run --queue configs/train_queue_al.yaml "
                "--group <g> --hours <h>`.\n" + _dump({"jobs": jobs}))
    _write_text(out / MODELS, header + "# AL students: the last weights (15 fixed epochs, no "
                "validation-based selection).\n" + _dump(registry))
    return runs


def _dump(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, width=110)
