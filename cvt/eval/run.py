"""Evaluate detectors on named frozen sets -> result rows (single model or paired comparison)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cvt.data.leakage import assert_no_leakage, leakage_report, load_or_compute_hashes
from cvt.data.manifest import load_manifest
from cvt.data.schema import CLASSES
from cvt.detect.base import Detector
from cvt.detect.composite import FusedDetector, fuse_predictions
from cvt.detect.registry import build_detector
from cvt.eval.ap import per_class_thresholds
from cvt.eval.bootstrap import bootstrap_ci, paired_bootstrap
from cvt.eval.predictions import predict_frames
from cvt.eval.score import SetScore, score_set
from cvt.utils import file_sha256, git_commit, stable_hash, write_parquet_atomic

Log = Callable[[str], None]


def _quiet(msg: str) -> None:
    pass


def load_checked_manifest(
    sets: list[str], manifest_dir: Path, hash_cache: Path, log: Log = _quiet
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load the manifest, check the sets exist, and run the leakage preflight."""
    images, boxes, mhash = load_manifest(manifest_dir)
    missing = [s for s in sets if s not in set(images.set_name)]
    if missing:
        raise ValueError(f"sets {missing} are empty or missing from the manifest")
    assert_no_leakage(leakage_report(images, load_or_compute_hashes(images, hash_cache, log=log)))
    return images, boxes, mhash


def open_detector(
    model: str, models_cfg: dict, imgsz: int, conf: float, images: pd.DataFrame,
    boxes: pd.DataFrame,
) -> tuple[Detector, str]:
    """(detector, model sha). The checkpoint is hashed before the detector is built, so a
    missing file fails before Ultralytics could auto-download a same-named checkpoint."""
    if model not in models_cfg:
        raise KeyError(f"model {model!r} not in models config ({sorted(models_cfg)})")
    cfg = models_cfg[model]
    if "weights" in cfg and not Path(cfg["weights"]).exists():
        raise FileNotFoundError(f"{model}: weights not found: {cfg['weights']}")
    sha = file_sha256(cfg["weights"])[:12] if "weights" in cfg else None
    detector = build_detector(model, models_cfg, imgsz, conf, images, boxes)
    return detector, sha or detector.cache_tag


def _require_gt(boxes: pd.DataFrame, im: pd.DataFrame, set_name: str) -> None:
    if not boxes.image_id.isin(im.image_id).any():
        raise ValueError(f"set {set_name!r} has no ground truth (e.g. a pseudo-label pool set)")


def _set_images(
    images: pd.DataFrame, boxes: pd.DataFrame, sets: list[str]
) -> list[tuple[str, pd.DataFrame]]:
    """(set name, its images in manifest order) per set; every set must have ground truth."""
    out = []
    for set_name in sets:
        im = images[images.set_name == set_name].reset_index(drop=True)
        _require_gt(boxes, im, set_name)
        out.append((set_name, im))
    return out


def set_hash_of(im: pd.DataFrame) -> str:
    return stable_hash(im[["image_id", "width", "height"]].values.tolist())[:8]


def set_predictions(
    detector: Detector, model: str, im: pd.DataFrame, set_name: str, pred_cache: Path,
    log: Log = _quiet,
) -> pd.DataFrame:
    """Predictions for one set, cached per (model, cache_tag, set, set content). Fused
    detectors are computed from their members' cached predictions."""
    set_hash = set_hash_of(im)
    cache = None
    if detector.cache_tag != "oracle":
        cache = Path(pred_cache) / f"{model}_{detector.cache_tag}_{set_name}_{set_hash}.parquet"
    if isinstance(detector, FusedDetector) and not (cache and cache.exists()):
        members = [set_predictions(m, m.name, im, set_name, pred_cache, log)
                   for m in detector.members]
        log(f"[{set_name}] fusing {len(members)} members (WBF)")
        dt = fuse_predictions(members, im, detector.iou_thr, detector.skip_thr, detector.max_det)
        if cache is not None:
            write_parquet_atomic(dt, cache)
        return dt
    cached = cache is not None and cache.exists()
    log(f"[{set_name}] {model}: {'loading cached' if cached else 'predicting'} {len(im)} images")
    return predict_frames(detector, im, cache)


def _row(**kw) -> dict:
    return {"slice": "all", **kw}


def evaluate_model(
    model: str, sets: list[str], imgsz: int, n_boot: int, seed: int, conf: float, *,
    models_cfg: dict, manifest_dir: Path, hash_cache: Path, pred_cache: Path,
    log: Log = _quiet, map_classes: list[int] | None = None,
) -> Iterator[list[dict]]:
    """Yield the result rows of each set as soon as that set is evaluated."""
    if model not in models_cfg:
        raise KeyError(f"model {model!r} not in models config ({sorted(models_cfg)})")
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    set_images = _set_images(images, boxes, sets)  # before any inference
    detector, sha = open_detector(model, models_cfg, imgsz, conf, images, boxes)
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_commit()
    hashed = {"model": models_cfg[model], "detector": detector.cache_tag, "imgsz": imgsz,
              "conf": conf, "n_boot": n_boot, "seed": seed}
    if map_classes is not None:  # only when given, so default-run config hashes stay stable
        hashed["map_classes"] = sorted(set(map_classes))
    chash = stable_hash(hashed)

    for set_name, im in set_images:
        dt = set_predictions(detector, model, im, set_name, pred_cache, log)
        log(f"[{set_name}] matching {len(dt)} detections (pycocotools)")
        sc = score_set(im, boxes, dt, map_classes)
        log(f"[{set_name}] bootstrap: {n_boot} resamples over {im.scene_id.nunique()} scenes")
        ci = bootstrap_ci(sc.resample_metrics, sc.scenes, n_boot=n_boot, seed=seed)
        notes = "mAP classes: " + ",".join(CLASSES[c] for c in sc.map_classes)
        if map_classes is not None:
            notes += " (pre-declared)"
        rows = []
        for metric, value in sc.metrics().items():
            lo, hi = ci.get(metric, (float("nan"), float("nan")))
            rows.append(_row(
                timestamp=stamp, git_commit=commit, model=model, model_sha=sha,
                eval_set=set_name, imgsz=imgsz, metric=metric, value=value, ci_lo=lo, ci_hi=hi,
                n_images=len(im), n_scenes=im.scene_id.nunique(), manifest_hash=mhash,
                set_hash=set_hash_of(im), config_hash=chash, notes=notes,
            ))
        yield rows


def compare_models(
    a: str, b: str, sets: list[str], imgsz_a: int, imgsz_b: int, n_boot: int, seed: int,
    conf: float, *, models_cfg: dict, manifest_dir: Path, hash_cache: Path, pred_cache: Path,
    log: Log = _quiet,
) -> Iterator[list[dict]]:
    """Paired cluster bootstrap of a − b on the same images and resamples, per set."""
    for m in (a, b):
        if m not in models_cfg:
            raise KeyError(f"model {m!r} not in models config ({sorted(models_cfg)})")
    if a == b and imgsz_a == imgsz_b:
        raise ValueError("comparing a model with itself")
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    set_images = _set_images(images, boxes, sets)  # before any inference
    det_a, sha_a = open_detector(a, models_cfg, imgsz_a, conf, images, boxes)
    det_b, sha_b = open_detector(b, models_cfg, imgsz_b, conf, images, boxes)
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_commit()
    chash = stable_hash({"a": [models_cfg[a], det_a.cache_tag, imgsz_a],
                         "b": [models_cfg[b], det_b.cache_tag, imgsz_b],
                         "conf": conf, "n_boot": n_boot, "seed": seed})
    for set_name, im in set_images:
        dt_a = set_predictions(det_a, a, im, set_name, pred_cache, log)
        log(f"[{set_name}] matching {len(dt_a)} detections (pycocotools)")
        sa = score_set(im, boxes, dt_a)
        dt_b = set_predictions(det_b, b, im, set_name, pred_cache, log)
        log(f"[{set_name}] matching {len(dt_b)} detections (pycocotools)")
        sb: SetScore = score_set(im, boxes, dt_b, sa.map_classes)
        log(f"[{set_name}] paired bootstrap: {n_boot} resamples over {im.scene_id.nunique()} scenes")
        res = paired_bootstrap(sa.resample_metrics, sb.resample_metrics, sa.scenes,
                               n_boot=n_boot, seed=seed)
        notes = f"vs {b}@{imgsz_b}; mAP classes: " + ",".join(CLASSES[c] for c in sa.map_classes)
        yield [
            _row(timestamp=stamp, git_commit=commit, model=a, model_sha=f"{sha_a}|{sha_b}",
                 eval_set=set_name, imgsz=imgsz_a, metric=f"delta/{k}", value=d, ci_lo=lo,
                 ci_hi=hi, n_images=len(im), n_scenes=im.scene_id.nunique(),
                 manifest_hash=mhash, set_hash=set_hash_of(im), config_hash=chash, notes=notes)
            for k, (d, lo, hi) in res.items()
        ]


def model_thresholds(
    model: str, set_name: str, imgsz: int, conf: float, *, models_cfg: dict,
    manifest_dir: Path, hash_cache: Path, pred_cache: Path, log: Log = _quiet,
) -> tuple[dict[str, float], str]:
    """(per-class F1-optimal score thresholds of `model` on `set_name` (use VD-tune), the
    detector's cache_tag). The tag pins the scores the thresholds apply to."""
    images, boxes, _ = load_checked_manifest([set_name], manifest_dir, hash_cache, log)
    im = _set_images(images, boxes, [set_name])[0][1]  # before any inference
    det, _ = open_detector(model, models_cfg, imgsz, conf, images, boxes)
    sc = score_set(im, boxes, set_predictions(det, model, im, set_name, pred_cache, log))
    th = {CLASSES[c]: t for c, t in per_class_thresholds(sc.tables, sc.class_ids).items()}
    return th, det.cache_tag
