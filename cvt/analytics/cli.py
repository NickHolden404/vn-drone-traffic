"""`cvt analytics detect | track | analyze`: the per-clip traffic-analytics pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer

from cvt.analytics import pipeline as pl
from cvt.analytics.anomaly import load_labels
from cvt.analytics.maps import render_map
from cvt.analytics.od import arm_name

analytics_app = typer.Typer(no_args_is_help=True)

CONFIG = Path("configs/analytics.yaml")
CACHE = Path("data/cache/analytics")
CFG_OPT = typer.Option(CONFIG, "--config", help="analytics config")
CLIPS_OPT = typer.Option(None, "--clips", help="comma-separated clip names [default: all]")
CACHE_OPT = typer.Option(CACHE, "--cache", help="detection / track cache")
MAP_UNITS = {"speed": ("speed", "km/h"), "density": ("density", "veh-s"),
             "stationary": ("stationary", "s")}


def _log(msg: str) -> None:
    typer.echo(msg, err=True)


def _select(cfg: dict, only: str | None) -> list[Path]:
    names = [c.strip() for c in only.split(",") if c.strip()] if only else None
    try:
        return pl.clips(cfg, names)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


def _load(config: Path, clips: str | None) -> tuple[dict, list[Path]]:
    try:
        cfg = pl.load_cfg(config)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    return cfg, _select(cfg, clips)


@analytics_app.command("detect")
def detect(config: Path = CFG_OPT, clips: str | None = CLIPS_OPT, cache: Path = CACHE_OPT) -> None:
    """Detect on every frame of each clip (GPU), cached per clip."""
    from cvt.config import load_models
    from cvt.train import job

    cfg, vids = _load(config, clips)
    job.no_network()
    from cvt.eval.run import open_detector

    det, _ = open_detector(cfg["detector"], load_models(cfg["models"]), int(cfg["imgsz"]),
                           float(cfg["conf"]), pd.DataFrame(), pd.DataFrame())
    for k, v in enumerate(vids, 1):
        _log(f"[{k}/{len(vids)}] {v.stem}")
        pl.detect_clip(v, cfg, det, cache, _log)


@analytics_app.command("track")
def track(config: Path = CFG_OPT, clips: str | None = CLIPS_OPT, cache: Path = CACHE_OPT) -> None:
    """Track and stabilize each clip from its cached detections (CPU)."""
    cfg, vids = _load(config, clips)
    from cvt.train import job

    job.no_network()  # GMC / BoT-SORT import Ultralytics
    try:
        for k, v in enumerate(vids, 1):
            _log(f"[{k}/{len(vids)}] {v.stem}")
            pl.track_clip(v, cfg, cache, _log)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


def _ref_image(path: Path) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f"cannot read reference frame {path}: {pl.HINT}")
    return im


def _json_safe(x):
    """Recursively replace NaN / inf by None and numpy scalars by Python ones."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _cell_px(cfg: dict, r: pl.ClipResult) -> float:
    """Map cell size in reference-frame pixels."""
    return float(cfg["grid_m"]) / float(r.scale["mpp"])


def _od_csv(r: pl.ClipResult) -> pd.DataFrame:
    t = r.od.reset_index()
    t["origin"] = [arm_name(int(i)) for i in t["origin"]]
    t["dest"] = [arm_name(int(i)) for i in t["dest"]]
    return t


@analytics_app.command("analyze")
def analyze(config: Path = CFG_OPT, clips: str | None = CLIPS_OPT, cache: Path = CACHE_OPT,
            out: Path = typer.Option(Path("data/demo_out/analytics"), help="output folder")) -> None:
    """Scale, OD, events and maps per clip from the caches (CPU) -> out/<clip>/."""
    cfg, vids = _load(config, clips)
    for k, v in enumerate(vids, 1):
        _log(f"[{k}/{len(vids)}] {v.stem}")
        try:
            r = pl.analyze_clip(v, cfg, cache)
            d = out / v.stem
            d.mkdir(parents=True, exist_ok=True)
            _od_csv(r).to_csv(d / "od.csv", index=False, lineterminator="\n")
            r.events.to_csv(d / "events.csv", index=False, lineterminator="\n")
            (d / "scale.json").write_text(
                json.dumps(_json_safe({**r.scale, "noise": r.noise, "od": r.od_summary}),
                           indent=2, allow_nan=False),
                encoding="utf-8", newline="\n")
            if r.maps:
                ref = _ref_image(pl.paths(cache, cfg, v.stem)["ref_jpg"])
                for name, (title, unit) in MAP_UNITS.items():
                    img = render_map(ref, r.maps[name], f"{v.stem} {title}", unit,
                                     cell_px=_cell_px(cfg, r))
                    if not cv2.imwrite(str(d / f"maps_{name}.jpg"), img):
                        raise OSError(f"cannot write {d / f'maps_{name}.jpg'}")
        except (FileNotFoundError, ValueError, OSError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e


@analytics_app.command("render")
def render(config: Path = CFG_OPT, clips: str | None = CLIPS_OPT, cache: Path = CACHE_OPT,
           out: Path = typer.Option(Path("data/demo_out/analytics"), help="output folder"),
           seconds: float | None = typer.Option(None, help="render only the first n seconds"),
           ) -> None:
    """Overlay video per clip from the caches (CPU) -> out/<clip>_analytics.mp4."""
    from cvt.analytics.render import render_clip

    cfg, vids = _load(config, clips)
    for k, v in enumerate(vids, 1):
        _log(f"[{k}/{len(vids)}] {v.stem}")
        try:
            r = pl.analyze_clip(v, cfg, cache)
            n = None if seconds is None else round(seconds * float(r.meta["fps"]))
            dest = render_clip(v, r, out / f"{v.stem}_analytics.mp4", max_frames=n,
                               max_dist=float(cfg["od"]["max_dist"]), log=_log)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
        _log(f"wrote {dest}")


@analytics_app.command("eval")
def eval_(
    config: Path = CFG_OPT, clips: str | None = CLIPS_OPT, cache: Path = CACHE_OPT,
    results: Path = typer.Option(Path("reports/results.csv"), help="results file to append to"),
    clips_csv: Path = typer.Option(Path("reports/analytics_clips.csv"), help="clip summary"),
    skip_od: bool = typer.Option(False, "--skip-od", help="skip the OD count rows"),
    od_sets: str | None = typer.Option(None, "--od-sets", help="comma-separated [default: config]"),
) -> None:
    """Anomaly AUC and OD count rows -> results.csv; per-clip summary -> clips csv."""
    from datetime import UTC, datetime

    from cvt.cli import HASH_CACHE, MANIFEST, TRACKS
    from cvt.config import load_models
    from cvt.eval.results import append_results
    from cvt.utils import git_commit

    cfg, vids = _load(config, clips)
    try:
        res = [pl.analyze_clip(v, cfg, cache) for v in vids]
        labels = {}
        for r in res:
            try:
                labels[r.clip] = load_labels(cfg["labels_dir"], r.clip, cfg["label_alias"])
            except FileNotFoundError:
                _log(f"{r.clip}: no label file, left out of the anomaly rows")
        base = pl.result_base(res, cfg, datetime.now(UTC).isoformat(timespec="seconds"),
                              git_commit())
        rows = pl.anomaly_rows(res, cfg, labels, base)
        if not skip_od:
            sets = [s.strip() for s in od_sets.split(",") if s.strip()] if od_sets \
                else cfg["od_sets"]
            rows += pl.od_rows(cfg, sets, models_cfg=load_models(cfg["models"]),
                               manifest_dir=MANIFEST, hash_cache=HASH_CACHE, out_root=TRACKS,
                               base=base, log=_log)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    summary = pd.DataFrame([pl.clip_summary(r, labels.get(r.clip), cfg) for r in res])
    clips_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(clips_csv, index=False, lineterminator="\n")
    append_results(results, rows)
    out = pd.DataFrame(rows)
    typer.echo(out[["eval_set", "slice", "model", "metric", "value", "ci_lo", "ci_hi",
                    "n_scenes"]].to_string(index=False))
    typer.echo(summary.to_string(index=False))


STILL_WIDTH = 1280


@analytics_app.command("stills")
def stills(config: Path = CFG_OPT, cache: Path = CACHE_OPT,
           out: Path = typer.Option(Path("reports/demo"), help="output folder")) -> None:
    """README images per `readme_clips` clip: mid-clip overlay frame and speed map."""
    from cvt.analytics.render import still

    try:
        cfg = pl.load_cfg(config)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    vids = _select(cfg, ",".join(cfg["readme_clips"]))
    out.mkdir(parents=True, exist_ok=True)
    for v in vids:
        try:
            r = pl.analyze_clip(v, cfg, cache)
            img = still(v, r, frame=int(r.meta["n_frames"]) // 2,
                        max_dist=float(cfg["od"]["max_dist"]))
            h, w = img.shape[:2]
            img = cv2.resize(img, (STILL_WIDTH, round(h * STILL_WIDTH / w)),
                             interpolation=cv2.INTER_AREA)
            dest = out / f"analytics_{v.stem}.jpg"
            if not cv2.imwrite(str(dest), img):
                raise OSError(f"cannot write {dest}")
            _log(f"wrote {dest}")
            if not r.maps:
                _log(f"{v.stem}: no maps, speed image skipped")
                continue
            ref = _ref_image(pl.paths(cache, cfg, v.stem)["ref_jpg"])
            dest = out / f"analytics_{v.stem}_speed.jpg"
            img = render_map(ref, r.maps["speed"], "mean speed", "km/h", cell_px=_cell_px(cfg, r))
            if not cv2.imwrite(str(dest), img):
                raise OSError(f"cannot write {dest}")
            _log(f"wrote {dest}")
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
