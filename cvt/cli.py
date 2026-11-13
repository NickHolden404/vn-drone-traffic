"""cvt command line: manifest build | leakage check | eval | compare | thresholds | report |
track (tune, run, eval, counts, compare) | demo | deploy (export, register, bench, list, report) |
al (prepare, list, comparisons, report)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import typer
import yaml

from cvt.analytics.cli import analytics_app
from cvt.config import load_models, load_yaml, read_exclusions
from cvt.data.leakage import (
    LeakageError,
    assert_no_leakage,
    leakage_report,
    load_or_compute_hashes,
    scene_summary,
)
from cvt.data.manifest import SET_ROLES, build_manifest, load_manifest, save_manifest
from cvt.data.schema import CLASSES
from cvt.eval.report import (
    DET_SETS,
    analytics_table,
    compare_table,
    follow_table,
    inject_readme,
    is_analytics,
    is_follow,
    results_table,
    tracking_table,
)
from cvt.eval.results import append_results
from cvt.eval.run import compare_models, evaluate_model, model_thresholds

app = typer.Typer(no_args_is_help=True, add_completion=False)
manifest_app = typer.Typer(no_args_is_help=True)
leakage_app = typer.Typer(no_args_is_help=True)
app.add_typer(manifest_app, name="manifest")
app.add_typer(leakage_app, name="leakage")
pool_app = typer.Typer(no_args_is_help=True)
app.add_typer(pool_app, name="pool")

MANIFEST = Path("data/manifest")
HASH_CACHE = Path("data/cache/phash.parquet")
RESULTS = Path("reports/results.csv")
DEPLOY = Path("configs/deploy.yaml")
LATENCY = Path("reports/latency.csv")
PARETO_PNG = Path("reports/pareto.png")
AL_CONFIG = Path("configs/al.yaml")
TILES_CONFIG = Path("configs/tiles.yaml")
AL_PNG = Path("reports/al_curve.png")
# results.csv text columns read as str (hashes, ids and notes must not become numbers / NaN)
RESULT_STR_COLS = ("git_commit", "model_sha", "slice", "manifest_hash", "set_hash", "config_hash",
                   "notes")
MODELS_OPT = typer.Option(
    "configs/models.yaml", help="comma-separated registry yamls, e.g. "
    "configs/models.yaml,configs/models_deploy.yaml (later files may not redefine names)",
)


@manifest_app.command("build")
def manifest_build(
    datasets: Path = Path("configs/datasets.yaml"),
    out: Path = MANIFEST,
    exclusions: Path = Path("configs/exclusions.txt"),
) -> None:
    report: dict = {}
    images, boxes = build_manifest(load_yaml(datasets), read_exclusions(exclusions), report)
    h = save_manifest(images, boxes, out, report["mot_gt"], report["seq_attrs"],
                      report["seq_drops"])
    summary = images.groupby("set_name").agg(
        images=("image_id", "size"), scenes=("scene_id", "nunique")
    )
    summary["boxes"] = boxes[~boxes.is_ignore].groupby(
        boxes.image_id.map(images.set_index("image_id").set_name)
    ).size()
    typer.echo(summary.to_string())
    if len(report["seq_drops"]):
        typer.echo(f"dropped sequences (drop_sequences, recorded in {out / 'seq_drops.parquet'}):")
        typer.echo(report["seq_drops"].to_string(index=False))
    typer.echo(f"manifest hash {h}")
    if report["unmatched_exclusions"]:
        n = len(report["unmatched_exclusions"])
        typer.echo(f"warning: {n} exclusion ids not found in the manifest", err=True)


@pool_app.command("extract")
def pool_extract(datasets: Path = Path("configs/datasets.yaml")) -> None:
    """Sample frames from the configured pool videos and write the pool index."""
    from cvt.data.adapters.video_pool import extract_pool

    pc = load_yaml(datasets)["sources"]["pool"]
    videos = [Path(pc["videos_dir"]) / v for v in pc["videos"]]
    index = extract_pool(videos, Path(pc["frames_dir"]), float(pc["every_s"]), pc["source"],
                         Path(pc["index"]), log=lambda m: typer.echo(m, err=True))
    typer.echo(f"{len(index)} frames from {index.video_id.nunique()} videos -> {pc['index']}")


@leakage_app.command("check")
def leakage_check(
    manifest: Path = MANIFEST,
    cache: Path = HASH_CACHE,
    max_dist: int = 6,
    out: Path = Path("reports/leakage_pairs.csv"),
    write_exclusions: Path | None = None,
    exclude_scenes: bool = typer.Option(
        False, help="with --write-exclusions: write every image of each flagged training scene"
    ),
    scenes_out: Path | None = typer.Option(
        None, help="scene-level summary CSV [default: leakage_scenes.csv next to --out]"
    ),
) -> None:
    if exclude_scenes and write_exclusions is None:
        raise typer.BadParameter("requires --write-exclusions", param_hint="--exclude-scenes")
    images, _, _ = load_manifest(manifest)
    hashes = load_or_compute_hashes(images, cache, log=lambda m: typer.echo(m, err=True))
    report = leakage_report(images, hashes, max_dist)
    out.parent.mkdir(parents=True, exist_ok=True)
    # one row per training-side image: its nearest test image
    nearest = report.sort_values("dist").drop_duplicates("train_id")
    nearest.to_csv(out, index=False)
    scenes = scene_summary(report, images)
    scenes_out = scenes_out or out.with_name("leakage_scenes.csv")
    scenes_out.parent.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(scenes_out, index=False)
    if report.empty:
        typer.echo("no near-duplicates")
    else:
        typer.echo(
            report.groupby(["train_set", "test_set"])
            .agg(pairs=("dist", "size"), images=("train_id", "nunique"))
            .to_string()
        )
        typer.echo(f"\nby scene (top 30 of {len(scenes)}, all in {scenes_out}):")
        typer.echo(scenes.head(30).to_string(index=False))
    if write_exclusions is not None and not report.empty:
        if exclude_scenes:
            train_side = images.set_name.map(SET_ROLES) != "test"
            flagged = images.scene_id.isin(scenes.train_scene) & train_side
            ids = sorted(set(images.loc[flagged, "image_id"]))
        else:
            ids = sorted(set(report.train_id))
        needs_nl = (
            write_exclusions.exists() and write_exclusions.stat().st_size > 0
            and not write_exclusions.read_bytes().endswith(b"\n")
        )
        with open(write_exclusions, "a", encoding="utf-8", newline="\n") as f:
            if needs_nl:
                f.write("\n")
            f.writelines(f"{i}\n" for i in ids)
        typer.echo(f"appended {len(ids)} ids to {write_exclusions}; rebuild the manifest")
    try:
        assert_no_leakage(report)
    except LeakageError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(..., help="name in configs/models.yaml"),
    sets: str = "T-VD,T-VN",
    imgsz: int = 1280,
    conf: float = 0.01,
    n_boot: int = 1000,
    seed: int = 0,
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = Path("data/cache/preds"),
    results: Path = RESULTS,
    map_classes: str = typer.Option(
        "", help="comma-separated classes averaged into mAP for every set "
        "[default: per-set automatic eligibility]"
    ),
) -> None:
    names = [c.strip() for c in map_classes.split(",") if c.strip()]
    unknown = [c for c in names if c not in CLASSES]
    if unknown:
        raise typer.BadParameter(
            f"unknown classes {unknown}; choose from {list(CLASSES)}", param_hint="--map-classes"
        )
    try:
        for rows in evaluate_model(
            model, [s.strip() for s in sets.split(",") if s.strip()], imgsz, n_boot, seed, conf,
            models_cfg=load_models(models), manifest_dir=manifest, hash_cache=hash_cache,
            pred_cache=pred_cache, log=lambda m: typer.echo(m, err=True),
            map_classes=[CLASSES.index(c) for c in names] or None,
        ):
            append_results(results, rows)  # per set, so a later crash keeps finished sets
            df = pd.DataFrame(rows)
            show = df[df.metric.isin(["mAP50", "mAP50_95"])][
                ["eval_set", "metric", "value", "ci_lo", "ci_hi"]
            ]
            typer.echo(show.to_string(index=False))
    except LeakageError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    except (KeyError, ValueError) as e:  # unknown model / set, results header mismatch
        typer.echo(f"error: {e.args[0] if e.args else e}", err=True)
        raise typer.Exit(1) from e
    except FileNotFoundError as e:  # untrained / missing checkpoint
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command("compare")
def compare_cmd(
    a: str = typer.Option(..., help="model A in configs/models.yaml"),
    b: str = typer.Option(..., help="model B (reference); deltas are A - B"),
    sets: str = "T-VD,T-VN",
    imgsz_a: int = 1280,
    imgsz_b: int = 1280,
    conf: float = 0.01,
    n_boot: int = 1000,
    seed: int = 0,
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = Path("data/cache/preds"),
    results: Path = RESULTS,
) -> None:
    """Paired cluster-bootstrap comparison of two models on the same images."""
    try:
        for rows in compare_models(
            a, b, [s.strip() for s in sets.split(",") if s.strip()], imgsz_a, imgsz_b, n_boot,
            seed, conf, models_cfg=load_models(models), manifest_dir=manifest,
            hash_cache=hash_cache, pred_cache=pred_cache, log=lambda m: typer.echo(m, err=True),
        ):
            append_results(results, rows)  # per set, so a later crash keeps finished sets
            df = pd.DataFrame(rows)
            show = df[df.metric == "delta/mAP50_95"][["eval_set", "value", "ci_lo", "ci_hi"]]
            typer.echo(show.to_string(index=False))
    except LeakageError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    except (KeyError, ValueError) as e:  # unknown model / set, results header mismatch
        typer.echo(f"error: {e.args[0] if e.args else e}", err=True)
        raise typer.Exit(1) from e
    except FileNotFoundError as e:  # untrained / missing checkpoint
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command("thresholds")
def thresholds_cmd(
    model: str = typer.Option(..., help="name in configs/models.yaml"),
    set_name: str = typer.Option("VD-tune", "--set", help="never a test set"),
    imgsz: int = 1280,
    conf: float = 0.01,
    out: Path = Path("configs/pseudo_thresholds.yaml"),
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = Path("data/cache/preds"),
) -> None:
    """Per-class F1-optimal score thresholds of a model on a non-test set."""
    if SET_ROLES.get(set_name) == "test":
        raise typer.BadParameter("thresholds are never tuned on a test set", param_hint="--set")
    try:
        th, cache_tag = model_thresholds(
            model, set_name, imgsz, conf, models_cfg=load_models(models), manifest_dir=manifest,
            hash_cache=hash_cache, pred_cache=pred_cache, log=lambda m: typer.echo(m, err=True),
        )
    except LeakageError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    except (KeyError, ValueError) as e:  # unknown model / set
        typer.echo(f"error: {e.args[0] if e.args else e}", err=True)
        raise typer.Exit(1) from e
    doc = {"model": model, "set": set_name, "cache_tag": cache_tag, "imgsz": imgsz, "conf": conf,
           "thresholds": th}
    text = yaml.safe_dump(doc, sort_keys=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


@app.command("report")
def _link(png: Path, page: Path) -> str:
    """Markdown link to `png` (relative to the current directory, the repository root) from the
    page `page`: relative to the page's folder when the page lives in the repository (e.g.
    `../reports/pareto.png` from `docs/results.md`); as given for a page elsewhere."""
    root, where = Path.cwd().resolve(), Path(page).resolve().parent
    if not where.is_relative_to(root):
        return Path(png).as_posix()
    return Path(os.path.relpath(root / png, where)).as_posix()


@app.command("report")
def report_cmd(
    results: Path = RESULTS,
    readme: Path = typer.Option(Path("docs/results.md"), "--out", "--readme",
                                help="results page with the generated blocks"),
    latency: Path = LATENCY,
    deploy_config: Path = DEPLOY, pareto_png: Path = PARETO_PNG,
    al_config: Path = AL_CONFIG, al_png: Path = AL_PNG,
    tiles_config: Path = TILES_CONFIG,
    clips_csv: Path = Path("reports/analytics_clips.csv"),
) -> None:
    df = pd.read_csv(results, dtype={c: str for c in RESULT_STR_COLS})
    cfg = load_yaml(deploy_config) if deploy_config.exists() else None
    al_cfg = load_yaml(al_config) if al_config.exists() else None
    grid: set[str] = set()
    if cfg is not None:  # deployment grid points appear only in the Pareto block
        from cvt.deploy.engines import grid_points, grid_tile, registry_name

        grid = {registry_name(p, grid_tile(cfg)) for p in grid_points(cfg)}
    al_runs: set[str] = set()
    if al_cfg is not None:  # AL runs, their engines and comparisons appear only in the AL block
        from cvt.al.prepare import _parse_runs
        from cvt.deploy.engines import engine_name

        e = al_cfg["engines"]
        grid |= {engine_name(m, int(e["imgsz"]), e["precision"]) for m in e["models"]}
        grid |= {str(p["name"]) for p in (al_cfg.get("fallbacks") or {}).get("points") or []}
        al_runs = {s["name"] for s in _parse_runs(al_cfg)}  # starting students stay
    tiles_cfg = load_yaml(tiles_config) if tiles_config.exists() else None
    if tiles_cfg is not None:  # batched tiled points appear only in the tiles block
        grid |= {p["batched"] for p in tiles_cfg["pairs"]}
    analytics = df[is_analytics(df)]  # anomaly / OD rows appear only in the analytics block
    df = df[~is_analytics(df)]
    follow = df[is_follow(df)]  # lock / gimbal rows appear only in the follow block
    df = df[~is_follow(df)]
    # MOT-set mAP goes to the tracking block
    table = results_table(df[~df.model.isin(grid | al_runs)], sets=DET_SETS)
    text = inject_readme(readme.read_text(encoding="utf-8"), table)
    if "<!-- compare:start -->" in text:  # every AL comparison has an AL run as `a`
        text = inject_readme(text, compare_table(df[~df.model.isin(al_runs)]), name="compare")
    if "<!-- tracking:start -->" in text:
        text = inject_readme(text, tracking_table(df), name="tracking")
    if "<!-- follow:start -->" in text:
        text = inject_readme(text, follow_table(follow), name="follow")
    if "<!-- analytics:start -->" in text:
        clips = pd.read_csv(clips_csv) if clips_csv.exists() else None
        text = inject_readme(text, analytics_table(analytics, clips), name="analytics")
    if "<!-- pareto:start -->" in text and latency.exists():  # no latency rows: keep the block
        from cvt.deploy.bench import _read_latency
        from cvt.deploy.pareto import pareto_block, pareto_table

        lat = _read_latency(latency)
        if len(lat):
            block = pareto_block(pareto_table(df, lat, cfg), lat,
                                 png=_link(pareto_png, readme) if pareto_png.exists() else None)
            text = inject_readme(text, block, name="pareto")
    if "<!-- al:start -->" in text and al_cfg is not None:
        from cvt.al.report import al_block
        from cvt.deploy.bench import _read_latency

        block = al_block(df, _read_latency(latency), al_cfg,
                         _link(al_png, readme) if al_png.exists() else None)
        text = inject_readme(text, block, name="al")
    if "<!-- tiles:start -->" in text and tiles_cfg is not None:
        from cvt.deploy.bench import _read_latency
        from cvt.deploy.tiles import tiles_block, tiles_table

        lat_t = _read_latency(Path(tiles_cfg["latency"]))
        block = tiles_block(tiles_table(df, lat_t, tiles_cfg), lat_t, tiles_cfg,
                            _read_latency(latency))
        text = inject_readme(text, block, name="tiles")
    readme.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(table)


yolo_app = typer.Typer(no_args_is_help=True)
app.add_typer(yolo_app, name="yolo")


@yolo_app.command("export")
def yolo_export(
    sets: str = typer.Option(..., help="comma-separated training-side sets, e.g. VD-train,VD-tune"),
    pseudo: Path | None = typer.Option(None, help="pseudo-label dir (required for P-train/P-val)"),
    manifest: Path = MANIFEST,
    out: Path = Path("data/yolo"),
) -> None:
    from cvt.data.export_yolo import export_labels, manifest_labels
    from cvt.pseudo.label import load_pseudo

    images, boxes, mhash = load_manifest(manifest)
    names = [x.strip() for x in sets.split(",") if x.strip()]
    if any(s.startswith("P-") for s in names) and pseudo is None:
        raise typer.BadParameter("P sets need --pseudo", param_hint="--pseudo")
    p_labels = p_frames = None
    if pseudo is not None:
        meta_path = pseudo / "meta.json"
        if not meta_path.exists():
            raise typer.BadParameter(f"{meta_path} missing (interrupted run?): re-run cvt pseudolabel",
                                     param_hint="--pseudo")
        made = json.loads(meta_path.read_text(encoding="utf-8")).get("manifest_hash")
        if made != mhash:
            typer.echo(f"warning: pseudo-labels were made against manifest {made}, current is {mhash}",
                       err=True)
        p_labels, p_frames = load_pseudo(pseudo)
    for s in names:
        im = images[images.set_name == s]
        if im.empty:
            raise typer.BadParameter(f"set {s!r} is empty or missing", param_hint="--sets")
        if s.startswith("P-"):
            n = len(set(im.image_id) - set(p_frames.image_id))
            if n:
                raise typer.BadParameter(f"pseudo-labels in {pseudo} do not cover {n} images of {s}; "
                                         "re-run cvt pseudolabel", param_hint="--pseudo")
            labels = p_labels[p_labels.image_id.isin(im.image_id)]
            ignore = labels.iloc[0:0]
            name = f"{s}-{pseudo.name}"
        else:
            labels, ignore = manifest_labels(im, boxes)
            name = s
        lst = export_labels(im, labels, ignore, name, out)
        typer.echo(f"{s}: {len(im)} images, {len(labels)} boxes -> {lst}")


@yolo_app.command("yaml")
def yolo_yaml(
    out: Path = typer.Option(...), train: str = typer.Option(...), val: str = typer.Option(...),
    root: Path = Path("data/yolo"),
) -> None:
    from cvt.data.export_yolo import write_dataset_yaml

    def lists(spec: str) -> list[Path]:
        ps = [root / f"{x.strip()}.txt" for x in spec.split(",") if x.strip()]
        missing = [str(p) for p in ps if not p.exists()]
        if missing:
            raise typer.BadParameter(f"missing list files {missing}: run `cvt yolo export` first")
        return ps

    typer.echo(write_dataset_yaml(out, lists(train), lists(val)))


def _threshold_file(path: Path, model: str, hint: str) -> tuple[dict, dict[int, float]]:
    """(file contents, {class id: tau}) of a `cvt thresholds` file computed for `model`,
    with exactly the known classes."""
    cfg = load_yaml(path)
    if cfg.get("model") != model:
        raise typer.BadParameter(f"{path} was computed for {cfg.get('model')!r}, not {model!r}: "
                                 "thresholds only apply to the scores of the same detector",
                                 param_hint=hint)
    given = cfg.get("thresholds") or {}
    unknown = sorted(set(given) - set(CLASSES))
    missing = [c for c in CLASSES if c not in given]
    if unknown or missing:
        raise typer.BadParameter(f"{path}: unknown classes {unknown}, missing classes {missing}",
                                 param_hint=hint)
    return cfg, {CLASSES.index(k): float(v) for k, v in given.items()}


def _check_cache_tag(path: Path, cfg: dict, cache_tag: str, hint: str) -> None:
    if cfg.get("cache_tag") != cache_tag:
        raise typer.BadParameter(
            f"{path}: thresholds were computed for a different detector configuration "
            f"(cache_tag {cfg.get('cache_tag')!r}, now {cache_tag!r}; check --imgsz/--conf "
            "or re-run cvt thresholds)", param_hint=hint)


@app.command("pseudolabel")
def pseudolabel_cmd(
    model: str = typer.Option("teacher-fused", help="a fused detector in configs/models.yaml"),
    sets: str = "P-train,P-val",
    thresholds: Path = Path("configs/pseudo_thresholds.yaml"),
    member_thresholds_dir: Path = typer.Option(
        Path("configs/thresholds"), help="<dir>/<member>.yaml per fused member (disagreement)"
    ),
    out: Path = Path("data/pseudo/teacher-fused"),
    cross_class_iou: float = typer.Option(
        0.5, help="drop a box overlapping a higher-scored box of another class (IoU >=); <= 0 disables"
    ),
    imgsz: int = 1280,
    conf: float = 0.01,
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = Path("data/cache/preds"),
) -> None:
    """Run the fused teacher's members on the pool (cached) and write pseudo-labels."""
    from cvt.detect.composite import FusedDetector
    from cvt.eval.run import load_checked_manifest, open_detector, set_predictions
    from cvt.pseudo.label import pseudo_label_frames, save_pseudo

    names = [s.strip() for s in sets.split(",") if s.strip()]
    # the AL pool (role pool) is scored only: its pseudo-labels cannot be exported for training
    bad = [s for s in names if SET_ROLES.get(s) not in ("train", "tune", "pool")]
    if bad:
        raise typer.BadParameter(f"pseudo-labels are only made for training-side sets and the AL "
                                 f"pool, not for test sets or unknown sets: {bad}", param_hint="--sets")
    if not thresholds.exists():
        raise typer.BadParameter(f"{thresholds} missing: run `cvt thresholds --model {model} "
                                 f"--set VD-tune --out {thresholds}` first", param_hint="--thresholds")
    th_cfg, th = _threshold_file(thresholds, model, "--thresholds")
    log = lambda m: typer.echo(m, err=True)  # noqa: E731
    images, boxes, mhash = load_checked_manifest(names, manifest, hash_cache, log)
    det, _ = open_detector(model, load_models(models), imgsz, conf, images, boxes)
    if not isinstance(det, FusedDetector):
        raise typer.BadParameter(f"{model} is not a fused detector", param_hint="--model")
    _check_cache_tag(thresholds, th_cfg, det.cache_tag, "--thresholds")
    hint = "--member-thresholds-dir"
    member_th, member_meta = [], []
    for m in det.members:
        path = member_thresholds_dir / f"{m.name}.yaml"
        if not path.exists():
            raise typer.BadParameter(f"{path} missing: run `cvt thresholds --model {m.name} "
                                     f"--set VD-tune --out {path}` first", param_hint=hint)
        cfg, t = _threshold_file(path, m.name, hint)
        _check_cache_tag(path, cfg, m.cache_tag, hint)
        member_th.append(t)
        member_meta.append({"name": m.name, "cache_tag": m.cache_tag,
                            "thresholds_file": str(path), "thresholds": cfg})
    # predict per named set (reusable caches; a change to one set does not re-predict the other)
    parts = [images[images.set_name == s].reset_index(drop=True) for s in names]
    im = pd.concat(parts, ignore_index=True)
    member_dts = [
        pd.concat([set_predictions(m, m.name, part, s, pred_cache, log)
                   for s, part in zip(names, parts, strict=True)], ignore_index=True)
        for m in det.members
    ]
    xc = cross_class_iou if cross_class_iou > 0 else None
    labels, frames = pseudo_label_frames(member_dts, im, th, det.iou_thr, det.skip_thr, det.max_det,
                                         member_thresholds=member_th, cross_class_iou=xc)
    save_pseudo(labels, frames, {"model": model, "cache_tag": det.cache_tag, "sets": names,
                                 "thresholds": th_cfg, "thresholds_file": str(thresholds),
                                 "members": member_meta, "manifest_hash": mhash,
                                 "cross_class_iou": xc}, out)
    typer.echo(f"{len(frames)} frames, {len(labels)} boxes, mean disagreement "
               f"{frames.disagreement.mean():.3f} -> {out}")


al_app = typer.Typer(no_args_is_help=True)
app.add_typer(al_app, name="al")


@al_app.command("prepare")
def al_prepare(
    config: Path = Path("configs/al.yaml"),
    manifest: Path = MANIFEST,
    out: Path = typer.Option(Path("."), help="root the generated files are written under"),
) -> None:
    """Rank the pool, reveal and export each run's picks, write the run yamls, the AL queue and
    the AL model registry (CPU; needs the pool scores of `scripts\\al.ps1 -Step score`)."""
    from cvt.al.prepare import QUEUE, prepare_runs

    def go() -> None:
        runs = prepare_runs(load_yaml(config), manifest, out)
        sets = sorted({r.pick_set for r in runs})
        typer.echo(f"{len(runs)} runs, {len(sets)} pick sets revealed and exported -> "
                   f"{(out / QUEUE).as_posix()}")

    _track_errors(go)


@al_app.command("list")
def al_list(models: Path = Path("configs/models_al.yaml")) -> None:
    """Print the AL registry names, one per line (what scripts/al.ps1 evaluates)."""

    def go() -> None:
        if not models.exists():
            raise FileNotFoundError(f"{models} missing: run `cvt al prepare` first")
        for name in load_yaml(models) or {}:
            typer.echo(name)

    _track_errors(go)


@al_app.command("comparisons")
def al_comparisons(config: Path = AL_CONFIG) -> None:
    """Print the pre-declared paired comparisons as `a b` lines (deltas a - b)."""

    def go() -> None:
        for c in load_yaml(config)["comparisons"]:
            typer.echo(f"{c['a']} {c['b']}")

    _track_errors(go)


@al_app.command("fallbacks")
def al_fallbacks(config: Path = AL_CONFIG) -> None:
    """Print the pre-declared secondary points of the primary run as `name imgsz` lines."""
    from cvt.al.report import fallback_points

    def go() -> None:
        for fb in fallback_points(load_yaml(config)):
            typer.echo(f"{fb['name']} {fb['imgsz']}")

    _track_errors(go)


@al_app.command("report")
def al_report(
    config: Path = AL_CONFIG,
    results: Path = RESULTS,
    out_csv: Path = Path("reports/al_curve.csv"),
    out_png: Path = AL_PNG,
) -> None:
    """Curve table (every run and the starting students at 1280) -> al_curve.csv and the
    accuracy-vs-budget plot al_curve.png (CPU); `cvt report` then fills the README block."""
    from cvt.al.report import curve_table, plot_curve

    def go() -> None:
        if not results.exists():
            raise FileNotFoundError(f"results file not found: {results}")
        table = curve_table(pd.read_csv(results, dtype={c: str for c in RESULT_STR_COLS}),
                            load_yaml(config))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False, lineterminator="\n")
        plot_curve(table, out_png)
        n = int(table["T-VN_mAP50_95"].notna().sum())
        typer.echo(f"{len(table)} runs ({n} with T-VN results) -> {out_csv}, {out_png}")

    _track_errors(go)


train_app = typer.Typer(no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("run")
def train_run(
    group: str = typer.Option(..., help="teachers | students | seeds | smoke"),
    hours: float = typer.Option(4.0, help="time budget for this piece"),
    queue: Path = Path("configs/train_queue.yaml"),
) -> None:
    """Train the group's next unfinished jobs for at most --hours; resumable.
    CVT_MIN_START_S (default 600) overrides the minimum time left to start a job (smoke check only)."""
    import os

    from cvt.train.queue import MIN_START_S, _subprocess_runner, load_queue, run_queue

    min_start = float(os.environ.get("CVT_MIN_START_S", MIN_START_S))
    out = run_queue(load_queue(queue), group, hours, _subprocess_runner(queue),
                    log=lambda m: typer.echo(m, err=True), min_start=min_start)
    typer.echo(f"{group}: {out}")


@train_app.command("status")
def train_status(queue: Path = Path("configs/train_queue.yaml")) -> None:
    """Per job: state, epochs done / planned, and remaining hours estimated from results.csv."""
    from cvt.train.queue import job_state, load_queue, seconds_per_epoch

    rows = []
    for j in load_queue(queue):
        state, done = job_state(j, log=lambda m: typer.echo(m, err=True))
        planned = int(j.args.get("epochs", 0))
        res = j.wdir.parent / "results.csv"
        eta = "?"
        if state == "paused" and res.exists():
            r = pd.read_csv(res)
            r.columns = [c.strip() for c in r.columns]
            per_epoch = seconds_per_epoch(r["time"].tolist()) if "time" in r else None
            if per_epoch is not None:
                eta = f"{per_epoch * (planned - (done or 0)) / 3600:.1f} h"
        if state == "done" and res.exists():  # early stopping (patience) can finish before `planned`
            done = len(pd.read_csv(res))
            if done < planned:
                state = "done (early stop)"
        rows.append({"job": j.name, "group": j.group, "state": state,
                     "epochs": f"{done or (planned if state == 'done' else 0)}/{planned}",
                     "remaining": "0" if state.startswith("done") else eta})
    typer.echo(pd.DataFrame(rows).to_string(index=False))


@train_app.command("done")
def train_done(
    name: str | None = typer.Option(None, help="one job"),
    group: str | None = typer.Option(None, help="every job of this group"),
    queue: Path = Path("configs/train_queue.yaml"),
) -> None:
    """Exit 0 if the job (or every job of --group) is trained and finalised, 1 otherwise
    (for scripts); 2 on an error (bad option, unreadable queue or checkpoint)."""
    from cvt.train.queue import job_state, load_queue

    if (name is None) == (group is None):
        raise typer.BadParameter("give exactly one of --name and --group")
    try:
        jobs = load_queue(queue)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    if name is not None:
        sel = [j for j in jobs if j.name == name]
        if not sel:
            raise typer.BadParameter(f"unknown job {name}", param_hint="--name")
    else:
        sel = [j for j in jobs if j.group == group]
        if not sel:
            raise typer.BadParameter(f"no job in group {group}", param_hint="--group")
    done = True
    for j in sel:
        try:
            state, _ = job_state(j, log=lambda m: typer.echo(m, err=True))
        except Exception as e:
            typer.echo(f"error: {j.name}: {e}", err=True)
            raise typer.Exit(2) from e
        typer.echo(f"{j.name}: {state}")
        done &= state == "done"
    raise typer.Exit(0 if done else 1)


track_app = typer.Typer(no_args_is_help=True)
app.add_typer(track_app, name="track")
TRACKS = Path("data/tracks")
WARP_CACHE = Path("data/cache/gmc")
PRED_CACHE = Path("data/cache/preds")


def _sets(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def _track_errors(fn) -> None:
    """Run `fn`; known failures exit 1 with one clear line (leakage keeps its own message)."""
    try:
        fn()
    except LeakageError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as e:
        typer.echo(f"error: {e.args[0] if e.args else e}", err=True)
        raise typer.Exit(1) from e


follow_app = typer.Typer(no_args_is_help=True)
app.add_typer(follow_app, name="follow")
app.add_typer(analytics_app, name="analytics")
FEATS = Path("data/cache/feats")


@follow_app.command("features")
def follow_features(
    weights_model: str = typer.Option(..., help="model whose .pt weights give the P3 feature map"),
    boxes_of: str = typer.Option(..., help="detector whose cached detections are pooled"),
    imgsz: int = typer.Option(1536, help="letterbox size the .pt is run at"),
    sets: str = "MOT-tune,T-MOT,T-UAVDT",
    min_score: float = 0.1,
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    out_dir: Path = FEATS,
) -> None:
    """Appearance features of cached vehicle detections -> <out-dir>/<boxes-of>_<tag>_<set>.npz
    (GPU when available; resumable per set)."""
    from cvt.follow.features import compute_features

    def go() -> None:
        for p in compute_features(
            weights_model, boxes_of, imgsz, _sets(sets), min_score, models_cfg=load_models(models),
            manifest_dir=manifest, hash_cache=hash_cache, pred_cache=pred_cache,
            cache_dir=out_dir, log=lambda m: typer.echo(m, err=True),
        ):
            typer.echo(str(p))

    _track_errors(go)


FOLLOW_CONFIG = Path("configs/follow.yaml")
FOLLOW_TUNED = Path("configs/follow.tuned.yaml")
FOLLOW_MODELS_OPT = typer.Option(None, help="registry yamls [default: the config's `models`]")


@follow_app.command("tune")
def follow_tune(
    config: Path = FOLLOW_CONFIG,
    out: Path = FOLLOW_TUNED,
    set_name: str = typer.Option("MOT-tune", "--set", help="never a test set"),
    reports_dir: Path = Path("reports"),
    models: str | None = FOLLOW_MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    out_root: Path = TRACKS,
    warp_cache: Path = WARP_CACHE,
    feats_dir: Path = FEATS,
) -> None:
    """Pre-declared lock (18) and gimbal (12) grids on MOT-tune -> <out> and
    <reports-dir>/follow_grid_{lock,gimbal}.csv (tracks MOT-tune first if needed; CPU)."""
    from cvt.follow import pipeline as fp

    def go() -> None:
        if SET_ROLES.get(set_name) != "tune":  # before reading any config or data
            raise ValueError(f"{set_name} has role {SET_ROLES.get(set_name)!r}: follow parameters "
                             "are tuned only on a tune set (MOT-tune), never on a test set")
        cfg = fp.load_follow_cfg(config)
        tuned, lock, gim = fp.tune_follow(
            cfg, set_name, models_cfg=load_models(models or cfg["models"]), manifest_dir=manifest,
            hash_cache=hash_cache, pred_cache=pred_cache, out_root=out_root,
            warp_cache=warp_cache, feats_dir=feats_dir, reports_dir=reports_dir,
            log=lambda m: typer.echo(m, err=True))
        obj = tuned["objective"]
        head = (f"# follow parameters tuned on {set_name} ({tuned['detector']}+{tuned['tracker']}, "
                f"{obj['n_targets']} lock tests): lock best of {len(lock)} by success AUC of "
                f"lock-{fp.FULL} ({obj['lock_success_auc']:.4f}); gimbal best of {len(gim)} by "
                f"centring error ({obj['gimbal_centre_err']:.4f}, smoothness among points within "
                f"5%); written by `cvt follow tune`, grids in {reports_dir}/follow_grid_lock.csv "
                f"and {reports_dir}/follow_grid_gimbal.csv\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(head + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8",
                       newline="\n")
        typer.echo(f"{out}: lock {tuned['lock']}, gimbal {tuned['gimbal']}")

    _track_errors(go)


@follow_app.command("eval")
def follow_eval(
    config: Path = FOLLOW_CONFIG,
    tuned: Path = FOLLOW_TUNED,
    sets: str = "T-MOT,T-UAVDT",
    n_boot: int = 1000,
    seed: int = 0,
    models: str | None = FOLLOW_MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    out_root: Path = TRACKS,
    warp_cache: Path = WARP_CACHE,
    feats_dir: Path = FEATS,
    results: Path = RESULTS,
) -> None:
    """The 5 lock variants and the gimbal (GT- and lock-driven) at the frozen params ->
    reports/results.csv (CPU)."""
    from cvt.follow import pipeline as fp

    def go() -> None:
        cfg = fp.load_follow_cfg(config)
        if not tuned.exists():
            raise FileNotFoundError(f"{tuned} not found: run `cvt follow tune` first")
        params = yaml.safe_load(tuned.read_text(encoding="utf-8"))
        for rows in fp.eval_follow_sets(
            cfg, params, _sets(sets), models_cfg=load_models(models or cfg["models"]),
            manifest_dir=manifest, hash_cache=hash_cache, pred_cache=pred_cache,
            out_root=out_root, warp_cache=warp_cache, feats_dir=feats_dir, n_boot=n_boot,
            seed=seed, log=lambda m: typer.echo(m, err=True),
        ):
            append_results(results, rows)  # per set, so a later crash keeps finished sets
            df = pd.DataFrame(rows)
            show = df[df.metric.isin(["lock/success_auc", "lock/lock_kept", "gimbal/centre_err"])]
            typer.echo(show[["eval_set", "model", "metric", "value", "ci_lo", "ci_hi"]]
                       .to_string(index=False))

    _track_errors(go)


FOLLOW_SOT = Path("configs/follow_sot.yaml")
SOT_CACHE = Path("data/cache/sot")
SOT_SETS_OPT = "UAVDT-S,UAV123-veh,UAV20L-veh"


@follow_app.command("sot-detect")
def follow_sot_detect(
    weights_model: str = typer.Option(..., help="model whose .pt weights give the P3 feature map"),
    config: Path = FOLLOW_CONFIG,
    sot: Path = typer.Option(FOLLOW_SOT, help="declared benchmark sequences"),
    sets: str = SOT_SETS_OPT,
    imgsz: int = typer.Option(1536, help="detector and feature input size"),
    conf: float = typer.Option(0.01, help="detector confidence floor (as the tracking evals)"),
    min_score: float = 0.1,
    models: str | None = FOLLOW_MODELS_OPT,
    cache: Path = SOT_CACHE,
) -> None:
    """Detections (the config's detector, m@1536 engine) and detector features on every frame of
    the declared benchmark videos -> <cache>/<detector>/<video>.{parquet,json,feats.npz} (GPU;
    resumable per video)."""
    from cvt.follow import pipeline as fp
    from cvt.follow import sot_data as sd

    def go() -> None:
        sd.check_sets(_sets(sets))  # before reading any config, data or model
        cfg = fp.load_follow_cfg(config)
        videos = fp.detect_sot(
            cfg, sd.load_sot_cfg(sot), _sets(sets), models_cfg=load_models(models or cfg["models"]),
            weights_model=weights_model, imgsz=imgsz, conf=conf, min_score=min_score,
            cache_dir=cache, log=lambda m: typer.echo(m, err=True))
        typer.echo(f"{len(videos)} videos cached in {cache}")

    _track_errors(go)


@follow_app.command("sot-eval")
def follow_sot_eval(
    config: Path = FOLLOW_CONFIG,
    sot: Path = typer.Option(FOLLOW_SOT, help="declared benchmark sequences"),
    tuned: Path = FOLLOW_TUNED,
    sets: str = SOT_SETS_OPT,
    n_boot: int = 1000,
    seed: int = 0,
    models: str | None = FOLLOW_MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    sot_hash_cache: Path = Path("data/cache/phash_sot.parquet"),
    cache: Path = SOT_CACHE,
    warp_cache: Path = WARP_CACHE,
    reports_dir: Path = Path("reports"),
    results: Path = RESULTS,
) -> None:
    """UAVDT-S / UAV123-veh / UAV20L-veh end to end (cached detections -> BoT-SORT-3b -> lock):
    the 5 lock variants and the gimbal (GT- and lock-driven) at the frozen params ->
    reports/results.csv; UAVDT-S leakage pairs -> <reports-dir>/follow_sot_leakage.csv (CPU)."""
    from cvt.follow import pipeline as fp
    from cvt.follow import sot_data as sd

    def go() -> None:
        sd.check_sets(_sets(sets))
        cfg = fp.load_follow_cfg(config)
        if not tuned.exists():
            raise FileNotFoundError(f"{tuned} not found: run `cvt follow tune` first")
        params = yaml.safe_load(tuned.read_text(encoding="utf-8"))
        fp.check_tuned(cfg, params)  # before any benchmark file or the manifest is read

        def log(m: str) -> None:
            typer.echo(m, err=True)

        train = (fp.training_hashes(manifest, hash_cache, log) if "UAVDT-S" in _sets(sets)
                 else pd.DataFrame({"image_id": [], "set_name": [], "phash": []}))  # unused
        for rows in fp.eval_sot_sets(
            cfg, params, sd.load_sot_cfg(sot), _sets(sets),
            models_cfg=load_models(models or cfg["models"]), cache_dir=cache,
            train_hashes=train, sot_hash_cache=sot_hash_cache, warp_cache=warp_cache,
            reports_dir=reports_dir, n_boot=n_boot, seed=seed, log=log,
        ):
            append_results(results, rows)  # per set, so a later crash keeps finished sets
            df = pd.DataFrame(rows)
            show = df[df.metric.isin(["lock/success_auc", "lock/lock_kept", "gimbal/centre_err"])]
            typer.echo(show[["eval_set", "model", "metric", "value", "ci_lo", "ci_hi"]]
                       .to_string(index=False))

    _track_errors(go)


def _demo_pick(click: str):
    """--click: 'x,y' pixels -> (x, y); 'auto' -> 'any'; a vehicle class name -> itself."""
    from cvt.follow.ope import VEHICLES

    c = click.strip().lower()
    if c in ("auto", "any"):
        return "any"
    if c in VEHICLES:
        return c
    try:
        x, y = (float(v) for v in c.split(","))
    except ValueError:
        raise typer.BadParameter(f"expected x,y pixels (e.g. 1920,1080), auto, or one of "
                                 f"{', '.join(VEHICLES)}", param_hint="--click") from None
    return (x, y)


def _follow_detector(config: Path, models: str | None, imgsz: int, conf: float):
    """(follow config, the config's detector opened offline)."""
    from cvt.follow import pipeline as fp

    cfg = fp.load_follow_cfg(config)
    from cvt.train import job

    job.no_network()
    from cvt.eval.run import open_detector

    det, _ = open_detector(cfg["detector"], load_models(models or cfg["models"]), imgsz, conf,
                           pd.DataFrame(), pd.DataFrame())
    return cfg, det


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


def _demo_videos(spec: str) -> list[Path]:
    """Comma-separated files or folders (their videos, sorted by name)."""
    out: list[Path] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        p = Path(part)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            raise typer.BadParameter(f"{p} not found", param_hint="--videos")
    if not out:
        raise typer.BadParameter(f"no videos in {spec}", param_hint="--videos")
    return out


@follow_app.command("demo")
def follow_demo(
    video: Path = typer.Option(..., help="source drone video (never committed)"),
    click: str = typer.Option(..., help="x,y pixel in the segment's first frame: the target; or "
                              "auto (any vehicle) / a class name (motorbike, car, truck, bus) "
                              "to pick it automatically"),
    frame: float = typer.Option(0.0, "--frame", help="segment start t0, seconds"),
    seconds: float = typer.Option(20.0, help="segment length, seconds"),
    out: Path = Path("reports/demo/follow.mp4"),
    config: Path = FOLLOW_CONFIG,
    tuned: Path = FOLLOW_TUNED,
    conf: float = typer.Option(0.01, help="detector confidence floor (as the tracking evals)"),
    imgsz: int = typer.Option(1536, help="detector input size"),
    keep_frames: bool = typer.Option(False, help="keep the segment's JPEGs in data/cache/demo"),
    models: str | None = FOLLOW_MODELS_OPT,
) -> None:
    """Click a vehicle, lock it (reacq variant, tuned point) and follow it with the virtual
    gimbal: side-by-side mp4 + MAVLink-shaped command log CSV (needs a GPU for the engine)."""
    def go() -> None:
        if not video.exists():
            raise typer.BadParameter(f"{video} not found", param_hint="--video")
        pick = _demo_pick(click)
        from cvt.follow.demo import load_demo_params, render_follow
        from cvt.track.run import load_tracker_cfg

        lock, gimbal, hfov = load_demo_params(config, tuned)  # fails before any model load
        cfg, det = _follow_detector(config, models, imgsz, conf)
        res = render_follow(
            video, pick, frame, seconds, out, detector=det,
            tracker_cfg=load_tracker_cfg(Path(cfg["tracker"])), lock=lock, gimbal=gimbal,
            hfov_deg=hfov, keep_frames=keep_frames, log=lambda m: typer.echo(m, err=True))
        typer.echo(f"{res} (+ {res.with_suffix('.csv')})")

    _track_errors(go)


@follow_app.command("demo-all")
def follow_demo_all(
    videos: str = typer.Option(..., help="comma-separated video files or folders ("
                               "clips, never committed)"),
    classes: str = typer.Option("motorbike,car,truck,bus",
                                help="one follow video per class, each a different vehicle"),
    seconds: float = typer.Option(60.0, help="segment length, from the middle of each clip"),
    out_dir: Path = Path("data/demo_out/follow"),
    config: Path = FOLLOW_CONFIG,
    tuned: Path = FOLLOW_TUNED,
    conf: float = typer.Option(0.01, help="detector confidence floor (as the tracking evals)"),
    imgsz: int = typer.Option(1536, help="detector input size"),
    models: str | None = FOLLOW_MODELS_OPT,
) -> None:
    """Follow demos on many clips: one mid-clip segment per video, the target picked
    automatically per class; out_dir/<clip>_<class>.mp4 + .csv and out_dir/summary.csv (one row
    per run, for picking the best). Resumable: clips already in summary.csv are skipped."""
    def go() -> None:
        from cvt.follow.ope import VEHICLES

        cls = [c.strip().lower() for c in classes.split(",") if c.strip()]
        bad = [c for c in cls if c not in VEHICLES]
        if bad or not cls:
            raise typer.BadParameter(f"classes must be among {', '.join(VEHICLES)}",
                                     param_hint="--classes")
        vids = _demo_videos(videos)
        from cvt.follow.demo import SUMMARY_COLS, demo_plan, load_demo_params, render_follows
        from cvt.track.run import load_tracker_cfg

        lock, gimbal, hfov = load_demo_params(config, tuned)  # fails before any model load
        plan = demo_plan(vids, seconds, cls, out_dir)
        summary = out_dir / "summary.csv"
        done = pd.read_csv(summary) if summary.exists() else pd.DataFrame(columns=SUMMARY_COLS)
        todo = [p for p in plan if p["video"].name not in set(done["video"].astype(str))]
        typer.echo(f"{len(plan)} clips, {len(plan) - len(todo)} already done", err=True)
        if not todo:
            return
        cfg, det = _follow_detector(config, models, imgsz, conf)
        tracker_cfg = load_tracker_cfg(Path(cfg["tracker"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(todo, 1):
            typer.echo(f"[{k}/{len(todo)}] {p['video'].name} from {p['t0_s']:g} s", err=True)
            rows = render_follows(p["video"], p["picks"], p["t0_s"], seconds, p["outs"],
                                  detector=det, tracker_cfg=tracker_cfg, lock=lock,
                                  gimbal=gimbal, hfov_deg=hfov,
                                  log=lambda m: typer.echo(m, err=True))
            new = pd.DataFrame(rows).reindex(columns=SUMMARY_COLS)
            done = new if done.empty else pd.concat([done, new], ignore_index=True)
            done.to_csv(summary, index=False, lineterminator="\n")
        typer.echo(f"{summary} ({len(done)} runs)")

    _track_errors(go)


@track_app.command("tune")
def track_tune(
    model: str = typer.Option(..., help="detector in configs/models.yaml (cached detections)"),
    tracker_type: str = typer.Option(..., help="bytetrack | botsort"),
    out: Path | None = typer.Option(None, help="[default: configs/trackers/<type>.tuned.yaml; "
                                    "--grid 3b: configs/trackers/<type>.tuned3b.<model>.yaml]"),
    set_name: str = typer.Option("MOT-tune", "--set", help="never a test set"),
    base: Path | None = typer.Option(None, help="[default: configs/trackers/<type>.yaml]"),
    grid: str = typer.Option("plan3", help="plan3 (rule 6, 54 points) | 3b (192 points)"),
    reports_dir: Path = Path("reports"),
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    warp_cache: Path = WARP_CACHE,
) -> None:
    """Pre-declared grid (rule 6, or 3b) on MOT-tune; writes the best config by class-combined
    HOTA."""
    from cvt.track import pipeline

    def go() -> None:
        best, table = pipeline.tune_tracker(
            model, tracker_type, set_name, models_cfg=load_models(models), manifest_dir=manifest,
            hash_cache=hash_cache, pred_cache=pred_cache, base_cfg=base, grid_name=grid,
            reports_dir=reports_dir, warp_cache=warp_cache,
            log=lambda m: typer.echo(m, err=True))
        csv = f"{reports_dir}/{pipeline.grid_report_name(tracker_type, grid, model)}"
        if grid == "plan3":
            label, cmd = "rule-6 grid", "`cvt track tune`"
            dest = out or Path("configs/trackers") / f"{tracker_type}.tuned.yaml"
        else:
            label, cmd = f"grid {grid}", f"`cvt track tune --grid {grid}`"
            dest = out or Path("configs/trackers") / f"{tracker_type}.tuned{grid}.{model}.yaml"
        head = (f"# {best['name']}: {label} on {set_name}, detector {model}, best of "
                f"{len(table)} by class-combined HOTA ({table['HOTA'].max():.4f}); written by "
                f"{cmd}, grid in {csv}\n")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(head + yaml.safe_dump(best, sort_keys=False), encoding="utf-8", newline="\n")
        typer.echo(f"{dest}: {best['name']} HOTA {table['HOTA'].max():.4f}")

    _track_errors(go)


@track_app.command("run")
def track_run(
    model: str = typer.Option(..., help="detector in configs/models.yaml (cached detections)"),
    tracker: Path = typer.Option(..., help="tracker yaml"),
    sets: str = "T-MOT,T-UAVDT",
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    out_root: Path = TRACKS,
    warp_cache: Path = WARP_CACHE,
) -> None:
    """Track each set on cached detections -> <out-root>/<model>+<tracker name>/<set>.parquet."""
    from cvt.track.pipeline import run_tracking

    def go() -> None:
        for s in _sets(sets):
            path = run_tracking(model, tracker, s, models_cfg=load_models(models),
                                manifest_dir=manifest, hash_cache=hash_cache, pred_cache=pred_cache,
                                out_root=out_root, warp_cache=warp_cache,
                                log=lambda m: typer.echo(m, err=True))
            typer.echo(f"{s}: {path}")

    _track_errors(go)


@track_app.command("eval")
def track_eval(
    model: str = typer.Option(..., help="detector in configs/models.yaml"),
    tracker: Path = typer.Option(..., help="tracker yaml used by `cvt track run`"),
    sets: str = "T-MOT,T-UAVDT",
    scenes: Path | None = typer.Option(None, help="scene yaml dir for counting, e.g. configs/scenes"),
    n_boot: int = 1000,
    seed: int = 0,
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    out_root: Path = TRACKS,
    stab_cache: Path = Path("data/cache/stab"),
    results: Path = RESULTS,
) -> None:
    """MOT metrics with sequence-bootstrap CIs (and line counts) -> reports/results.csv."""
    from cvt.track.pipeline import eval_tracking_sets

    def go() -> None:
        for rows in eval_tracking_sets(
            model, tracker, _sets(sets), models_cfg=load_models(models), manifest_dir=manifest,
            hash_cache=hash_cache, pred_cache=pred_cache, out_root=out_root, scenes_dir=scenes,
            stab_cache=stab_cache, n_boot=n_boot, seed=seed,
            log=lambda m: typer.echo(m, err=True),
        ):
            append_results(results, rows)  # per set, so a later crash keeps finished sets
            df = pd.DataFrame(rows)
            show = df[(df["slice"] == "all") & df.metric.isin(["HOTA", "MOTA", "IDF1", "count_MAE"])]
            typer.echo(show[["eval_set", "metric", "value", "ci_lo", "ci_hi"]].to_string(index=False))

    _track_errors(go)


@track_app.command("counts")
def track_counts(
    model: str = typer.Option(..., help="detector in configs/models.yaml"),
    tracker: Path = typer.Option(..., help="tracker yaml used by `cvt track run`"),
    sets: str = "T-MOT,T-UAVDT",
    scenes: Path = typer.Option(Path("configs/scenes"), help="scene yaml dir"),
    out: Path | None = typer.Option(
        None, help="rule 1 [default: reports/counts_<model>+<tracker name>.csv]"),
    out2: Path | None = typer.Option(
        None, help="rule 2 [default: reports/counts2_<model>+<tracker name>.csv]"),
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    pred_cache: Path = PRED_CACHE,
    out_root: Path = TRACKS,
    stab_cache: Path = Path("data/cache/stab"),
) -> None:
    """GT vs predicted line counts per (scene, line, class, direction) of saved tracks -> csv,
    one per counting rule (the cells behind the count / count2 rows of `cvt track eval`;
    results.csv is not touched)."""
    from cvt.track.pipeline import count_cells_sets
    from cvt.track.run import load_tracker_cfg

    def go() -> None:
        cells = count_cells_sets(
            model, tracker, _sets(sets), scenes, models_cfg=load_models(models),
            manifest_dir=manifest, hash_cache=hash_cache, pred_cache=pred_cache,
            out_root=out_root, stab_cache=stab_cache, rules=(1, 2),
            log=lambda m: typer.echo(m, err=True))
        name = f"{model}+{load_tracker_cfg(tracker)['name']}"
        for rule, dest in ((1, out or Path("reports") / f"counts_{name}.csv"),
                           (2, out2 or Path("reports") / f"counts2_{name}.csv")):
            dest.parent.mkdir(parents=True, exist_ok=True)
            cells[rule].to_csv(dest, index=False, lineterminator="\n")
            typer.echo(f"rule {rule}: {len(cells[rule])} cells -> {dest}")

    _track_errors(go)


@track_app.command("compare")
def track_compare(
    a: str = typer.Option(..., help="<model>+<tracker name>, e.g. M+botsort-tuned"),
    b: str = typer.Option(..., help="reference run; deltas are A - B"),
    sets: str = "T-MOT,T-UAVDT",
    n_boot: int = 1000,
    seed: int = 0,
    cross_detector: bool = typer.Option(
        False, help="A and B use different detectors (a detector+tracker comparison)"),
    manifest: Path = MANIFEST,
    hash_cache: Path = HASH_CACHE,
    out_root: Path = TRACKS,
    results: Path = RESULTS,
) -> None:
    """Paired sequence-bootstrap comparison of two tracker runs on the same sets."""
    from cvt.track.pipeline import compare_tracking_sets

    def go() -> None:
        for rows in compare_tracking_sets(
            a, b, _sets(sets), manifest_dir=manifest, hash_cache=hash_cache, out_root=out_root,
            n_boot=n_boot, seed=seed, cross_detector=cross_detector,
            log=lambda m: typer.echo(m, err=True),
        ):
            append_results(results, rows)
            df = pd.DataFrame(rows)
            typer.echo(df[["eval_set", "metric", "value", "ci_lo", "ci_hi"]].to_string(index=False))

    _track_errors(go)


@app.command("demo")
def demo_cmd(
    video: Path = typer.Option(..., help="source drone video"),
    scene: Path = typer.Option(..., help="scene yaml: counting line(s) in the pixels of --ref-frame"),
    model: str = typer.Option(..., help="yolo/rtdetr detector in configs/models.yaml"),
    tracker: Path = typer.Option(..., help="tracker yaml (e.g. configs/trackers/botsort.tuned.yaml)"),
    out: Path = Path("reports/demo/demo.mp4"),
    start: float = typer.Option(60.0, help="segment start, seconds"),
    duration: float = typer.Option(60.0, help="segment length, seconds"),
    conf: float = typer.Option(
        0.05, help="detector confidence floor (demo rendering; tracking evals use 0.01)"
    ),
    fps_out: float | None = typer.Option(
        None, help="output frame rate (every segment frame is written) [default: the source's]"
    ),
    width_out: int = typer.Option(1920, help="output width in pixels (aspect kept)"),
    keep_frames: bool = typer.Option(False, help="keep the segment's JPEGs in data/cache/demo"),
    models: str = MODELS_OPT,
) -> None:
    """Render a demo mp4 (+ 3 PNG stills, + the middle one as <out stem>_still.jpg for the README):
    detections, track ids, counting line and live counts."""
    import pandas as pd

    from cvt.eval.run import open_detector
    from cvt.track.count import load_scene
    from cvt.track.demo import render_demo
    from cvt.track.run import load_tracker_cfg

    def go() -> None:
        for p, opt in ((video, "--video"), (scene, "--scene"), (tracker, "--tracker")):
            if not p.exists():
                raise typer.BadParameter(f"{p} not found", param_hint=opt)
        det, _ = open_detector(model, load_models(models), 1280, conf, pd.DataFrame(),
                               pd.DataFrame())
        res = render_demo(
            video, start, duration, det, load_tracker_cfg(tracker), load_scene(scene), out,
            fps_out=fps_out, width_out=width_out, keep_frames=keep_frames,
            log=lambda m: typer.echo(m, err=True),
        )
        typer.echo(json.dumps(res["counts"], indent=2))
        typer.echo(f"{res['total']} crossings, {res['tracks']} tracks -> {res['out']}")

    _track_errors(go)


deploy_app = typer.Typer(no_args_is_help=True)
app.add_typer(deploy_app, name="deploy")
ENGINES = Path("data/engines")


def _ultralytics_offline() -> None:
    """Before Ultralytics is imported (it reads these at import): no pip auto-install and no
    network lookups during the GPU deploy runs; values already set are kept."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "0")
    os.environ.setdefault("YOLO_OFFLINE", "1")


@deploy_app.command("export")
def deploy_export(
    only: str | None = typer.Option(None, help="one engine registry name, e.g. <model>@640-int8"),
    config: Path = DEPLOY,
    models: str = MODELS_OPT,
    manifest: Path = MANIFEST,
    out: Path = ENGINES,
) -> None:
    """Export the grid's TensorRT engines (GPU, user-run); up-to-date engines are skipped,
    stale ones refused."""
    _ultralytics_offline()
    from cvt.deploy.engines import engine_points, export_engine, grid_tile, registry_name

    def go() -> None:
        cfg, models_cfg = load_yaml(config), load_models(models)
        tile = grid_tile(cfg)
        pts = engine_points(cfg)
        if only is not None:
            names = [registry_name(p, tile) for p in pts]
            if only not in names:
                raise ValueError(f"--only {only!r} is not an engine of the grid ({names})")
            pts = [p for p in pts if registry_name(p, tile) == only]
        for p in pts:
            path = export_engine(p, models_cfg, cfg, out, manifest_dir=manifest,
                                 log=lambda m: typer.echo(m, err=True))
            typer.echo(f"{registry_name(p, tile)}: {path}")

    _track_errors(go)


@deploy_app.command("register")
def deploy_register(
    config: Path = DEPLOY,
    models: str = MODELS_OPT,
    out: Path | None = typer.Option(
        None, help="[default: models_<config stem>.yaml next to the config, e.g. "
        "configs/models_deploy.yaml, configs/models_deploy_al.yaml]"),
    engines: Path = ENGINES,
) -> None:
    """Write the generated registry entries of every grid point (CPU)."""
    from cvt.deploy.engines import write_registry

    def go() -> None:
        dst = out if out is not None else config.with_name(f"models_{config.stem}.yaml")
        path = write_registry(load_yaml(config), load_models(models), dst, engines,
                              config=config, models=models)
        typer.echo(f"{len(load_yaml(path))} entries -> {path}")

    _track_errors(go)


@deploy_app.command("bench")
def deploy_bench(
    only: str | None = typer.Option(None, help="comma-separated registry names of grid points"),
    config: Path = DEPLOY,
    models: str = typer.Option(
        "configs/models.yaml,configs/models_deploy.yaml",
        help="comma-separated registry yamls (the generated one is `cvt deploy register`'s)",
    ),
    manifest: Path = MANIFEST,
    out: Path = Path("reports/latency.csv"),
) -> None:
    """End-to-end latency of the grid's points on 199 fixed T-VN frames (GPU, user-run);
    points already benchmarked (same frames and engine/weights sha) are skipped."""
    _ultralytics_offline()
    from cvt.deploy import bench

    def go() -> None:
        cfg = load_yaml(config)
        points = bench.select_points(cfg, only)
        device = bench.cuda_device()  # refuses to run without CUDA
        for f in (s.strip() for s in models.split(",") if s.strip()):
            if not Path(f).exists():
                raise FileNotFoundError(f"models file not found: {f} (generate the grid's "
                                        "entries with `cvt deploy register`)")
        models_cfg = load_models(models)
        images, boxes, _ = load_manifest(manifest)
        rows = bench.run_bench(points, cfg, models_cfg, images, boxes, out, device,
                               log=lambda m: typer.echo(m, err=True))
        typer.echo(f"{len(rows)} of {len(points)} points benchmarked -> {out}")

    _track_errors(go)


@deploy_app.command("list")
def deploy_list(config: Path = DEPLOY) -> None:
    """Print one `name imgsz` line per grid point (what scripts/deploy.ps1 iterates over)."""
    from cvt.deploy.engines import grid_points, grid_tile, registry_name

    def go() -> None:
        cfg = load_yaml(config)
        tile = grid_tile(cfg)
        for p in grid_points(cfg):
            typer.echo(f"{registry_name(p, tile)} {p.imgsz}")

    _track_errors(go)


@deploy_app.command("report")
def deploy_report(
    config: Path = DEPLOY,
    results: Path = RESULTS,
    latency: Path = LATENCY,
    out_csv: Path = Path("reports/pareto.csv"),
    out_png: Path = PARETO_PNG,
) -> None:
    """Join accuracy (results.csv) and latency (latency.csv) per grid point -> pareto.csv and
    the latency-vs-mAP plot pareto.png (CPU)."""
    from cvt.deploy.bench import _read_latency
    from cvt.deploy.pareto import pareto_table, plot_pareto

    def go() -> None:
        for f, what in ((results, "results"), (latency, "latency")):
            if not f.exists():
                raise FileNotFoundError(f"{what} file not found: {f}"
                                        + (" (run `cvt deploy bench` first)" if f == latency
                                           else ""))
        lat = _read_latency(latency)
        if not len(lat):
            raise ValueError(f"{latency} has no rows: run `cvt deploy bench` first")
        table = pareto_table(pd.read_csv(results, dtype={c: str for c in RESULT_STR_COLS}), lat,
                             load_yaml(config))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False, lineterminator="\n")
        plot_pareto(table, out_png)
        typer.echo(f"{len(table)} points ({int(table.frontier.sum())} on the frontier) -> "
                   f"{out_csv}, {out_png}")

    _track_errors(go)
