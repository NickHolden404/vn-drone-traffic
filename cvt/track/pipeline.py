"""Tracking pipeline: cached detections -> tracks per set -> MOT metrics with sequence-bootstrap
CIs, the pre-declared tuning grids on MOT-tune, paired tracker comparisons and line counts
(counting rules 1 and 2).

Detections always come from the prediction cache written by `cvt eval` (`scripts/track.ps1
-Step detect`); nothing here runs or even constructs a detection model. Frames are keyed by
(sequence, frame number) read from the manifest ids, never by list position.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.manifest import SET_ROLES, load_mot
from cvt.data.schema import CLASS_ID, CLASSES
from cvt.eval.bootstrap import bootstrap_ci, paired_bootstrap
from cvt.eval.mot import MOT_COLS, SeqStats, sequence_stats, summarize
from cvt.eval.run import load_checked_manifest, open_detector, set_hash_of, set_predictions
from cvt.track.count import (
    count_errors,
    count_table,
    crossings,
    first_crossings,
    load_scene,
    near_border,
)
from cvt.track.run import DET_COLS, TRACK_COLS, gmc_warps, load_tracker_cfg, track_sequence
from cvt.track.stabilize import reference_homographies, stabilize_tracks
from cvt.utils import git_commit, stable_hash, write_parquet_atomic

Log = Callable[[str], None]

IMGSZ = 1280  # tracking detector resolution (plan: full frame, imgsz 1280, conf 0.01)
DET_CONF = 0.01
# rule 3: evaluated classes per set; other detections are dropped before tracking
SET_CLASSES: dict[str, tuple[int, ...]] = {
    "MOT-tune": tuple(range(len(CLASSES))),
    "T-MOT": tuple(range(len(CLASSES))),
    "T-UAVDT": tuple(sorted(CLASS_ID[c] for c in ("car", "truck", "bus"))),
}
SLICES = ("daylight", "night", "fog", "low_alt", "medium_alt", "high_alt")
METRICS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDsw")
COMPARE_METRICS = ("HOTA", "MOTA", "IDF1")
IGNORE_MIN_INSIDE = 0.5  # rule 4: tracker boxes >= 50% of their area inside a region are dropped
MASK_SCORE = 0.25  # cached detections at least this confident are masked out of stabilization
# rule 6: the tuning grid (54 points) and the values fixed for every point
GRID: dict[str, tuple] = {
    "track_high_thresh": (0.2, 0.3, 0.4),
    "new_track_thresh": (0.3, 0.4, 0.5),
    "match_thresh": (0.7, 0.8, 0.9),
    "track_buffer": (30, 60),
}
GRID_FIXED = {"track_low_thresh": 0.1, "fuse_score": True, "frame_rate": 30}
GRID_FIXED_BOTSORT = {"gmc_method": "sparseOptFlow", "with_reid": False}
# grid 3b: the wider grid (192 points), same fixed values; tuned configs are `<type>-tuned3b`
GRID3B: dict[str, tuple] = {
    "track_high_thresh": (0.2, 0.3, 0.4, 0.5),
    "new_track_thresh": (0.4, 0.5, 0.6, 0.7),
    "match_thresh": (0.7, 0.8, 0.9, 0.95),
    "track_buffer": (30, 60, 90),
}
GRIDS: dict[str, tuple[dict[str, tuple], str]] = {"plan3": (GRID, "tuned"), "3b": (GRID3B, "tuned3b")}
COUNT_RULES = {1: "count", 2: "count2"}  # counting rule -> metric prefix (count_MAE, count2_MAE)
DETECT_HINT = (r"run `scripts\track.ps1 -Step detect` (the m@1536 engine: "
               r"`scripts\track3b.ps1 -Step detect`) first (GPU, caches the detections)")
NAN = float("nan")


def _quiet(msg: str) -> None:
    pass


def det_imgsz(models_cfg: Mapping, model: str) -> int:
    """The detector's resolution for result rows: its registry `imgsz` (generated engines, e.g.
    1536), else IMGSZ."""
    return int((models_cfg.get(model) or {}).get("imgsz") or IMGSZ)


def set_classes(set_name: str) -> tuple[int, ...]:
    if set_name not in SET_CLASSES:
        raise ValueError(f"no tracking class rule for set {set_name!r} (known: {list(SET_CLASSES)})")
    return SET_CLASSES[set_name]


# --- sets ----------------------------------------------------------------------------------

@dataclass
class SetData:
    """One MOT set: images in manifest order (prediction cache key), frames sorted by
    (seq_id, frame), GT tracks of the evaluated classes, and per-frame ignore regions."""

    name: str
    im: pd.DataFrame
    frames: pd.DataFrame  # image_id, path, width, height, seq_id, frame
    gt: pd.DataFrame  # MOT_GT_COLS
    ignore: pd.DataFrame  # seq_id, frame, x1, y1, x2, y2
    classes: tuple[int, ...]

    @property
    def sequences(self) -> list[str]:
        return sorted(self.frames.seq_id.unique())

    def frames_per_seq(self) -> dict[str, int]:
        return self.frames.groupby("seq_id").size().to_dict()


def load_set(
    set_name: str, images: pd.DataFrame, boxes: pd.DataFrame, mot_gt: pd.DataFrame
) -> SetData:
    """Sequence = scene_id after its source prefix; frame = the number ending the image id
    (the adapters' naming). Cross-checked against the MOT GT table."""
    classes = set_classes(set_name)
    im = images[images.set_name == set_name].reset_index(drop=True)
    if im.empty:
        raise ValueError(f"set {set_name!r} is empty or missing from the manifest")
    seq = im.scene_id.astype(str).str.split("/", n=1).str[1]
    num = pd.to_numeric(im.image_id.astype(str).str.rsplit("/", n=1).str[-1], errors="coerce")
    bad = seq.isna() | num.isna()
    if bad.any():
        r = im[bad].iloc[0]
        raise ValueError(f"{set_name}: cannot read sequence/frame from image {r.image_id!r} "
                         f"(scene {r.scene_id!r})")
    frames = im[["image_id", "path", "width", "height"]].assign(seq_id=seq,
                                                                frame=num.astype("int64"))
    dup = frames.duplicated(["seq_id", "frame"], keep=False)
    if dup.any():
        r = frames[dup].iloc[0]
        raise ValueError(f"{set_name}: frame {r.frame} of sequence {r.seq_id} appears twice")
    gt = mot_gt[mot_gt.image_id.isin(im.image_id)]
    chk = gt.merge(frames, on="image_id", suffixes=("", "_img"))
    off = (chk.seq_id != chk.seq_id_img) | (chk.frame != chk.frame_img)
    if off.any():
        r = chk[off].iloc[0]
        raise ValueError(f"{set_name}: MOT GT says image {r.image_id!r} is {r.seq_id}/{r.frame}, "
                         f"its id says {r.seq_id_img}/{r.frame_img}")
    gt = gt[gt.label.isin(classes)].reset_index(drop=True)
    ign = boxes[boxes.is_ignore.astype(bool) & boxes.image_id.isin(im.image_id)]
    ignore = ign.merge(frames[["image_id", "seq_id", "frame"]], on="image_id")[
        ["seq_id", "frame", "x1", "y1", "x2", "y2"]].reset_index(drop=True)
    frames = frames.sort_values(["seq_id", "frame"]).reset_index(drop=True)
    return SetData(set_name, im, frames, gt, ignore, classes)


# --- cached detections ---------------------------------------------------------------------

class _CacheOnly:
    """A detector stand-in for `set_predictions`: its predictions must already be cached."""

    def __init__(self, name: str, cache_tag: str):
        self.name, self.cache_tag = name, cache_tag

    def predict_paths(self, paths):
        raise RuntimeError(f"{self.name}: cached detections disappeared; {DETECT_HINT}")


def _cache_tag(model: str, cfg: dict, s: SetData, pred_cache: Path) -> str:
    """The detector cache_tag, read from the one matching prediction cache file name.

    An Ultralytics detector's tag hashes the checkpoint's class names, which are only known
    after loading the model; reading the tag from the cache avoids that (and any GPU use)."""
    set_hash = set_hash_of(s.im)
    pat = re.compile(rf"{re.escape(model)}_([0-9a-f]+)_{re.escape(s.name)}_{set_hash}\.parquet")
    hits = sorted(p for p in Path(pred_cache).glob(f"*_{s.name}_{set_hash}.parquet")
                  if pat.fullmatch(p.name))
    if not hits:
        raise RuntimeError(f"no cached detections of {model} on {s.name} (set hash {set_hash}) "
                           f"in {pred_cache}; {DETECT_HINT}")
    if len(hits) > 1:
        raise RuntimeError(f"several cached detection files of {model} on {s.name}: "
                           f"{[h.name for h in hits]}; delete the stale ones (earlier weights or "
                           "--imgsz/--conf) and keep the one from `scripts\\track.ps1 -Step detect`")
    weights = cfg.get("weights")
    if weights and Path(weights).exists() and Path(weights).stat().st_mtime > hits[0].stat().st_mtime:
        raise RuntimeError(f"{weights} is newer than the cached detections {hits[0].name}: "
                           f"delete that file and {DETECT_HINT}")
    return pat.fullmatch(hits[0].name).group(1)


def cached_detections(
    model: str, models_cfg: dict, s: SetData, images: pd.DataFrame, boxes: pd.DataFrame,
    pred_cache: Path, log: Log = _quiet,
) -> tuple[pd.DataFrame, str]:
    """(detections with seq_id + DET_COLS, detector cache_tag). Never runs a model: the oracle
    reads GT, every other detector must have a prediction cache for this set."""
    if model not in models_cfg:
        raise KeyError(f"model {model!r} not in models config ({sorted(models_cfg)})")
    cfg = models_cfg[model]
    if cfg.get("arch") == "oracle":
        det, _ = open_detector(model, models_cfg, IMGSZ, DET_CONF, images, boxes)
    else:
        det = _CacheOnly(model, _cache_tag(model, cfg, s, pred_cache))
    dt = set_predictions(det, model, s.im, s.name, pred_cache, log)
    unknown = set(dt.image_id) - set(s.frames.image_id)
    if unknown:
        raise ValueError(f"cached detections name images outside {s.name}: {sorted(unknown)[:3]}")
    d = dt.merge(s.frames[["image_id", "seq_id", "frame"]], on="image_id")
    d = d.astype({"label": "int64", "score": "float64"})
    return d[["seq_id", *DET_COLS]].reset_index(drop=True), det.cache_tag


# --- tracking ------------------------------------------------------------------------------

def _uses_gmc(cfg: dict) -> bool:
    return cfg["tracker_type"] == "botsort" and str(cfg.get("gmc_method")).lower() != "none"


def _save_atomic(path: Path, write: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def warp_path(seq: str, fr: pd.DataFrame, cfg: dict, warp_cache: Path) -> Path | None:
    """Cache file of one sequence's GMC warps (None when the tracker uses no GMC)."""
    if not _uses_gmc(cfg):
        return None
    key = stable_hash([int(f) for f in fr.frame])[:12]
    return Path(warp_cache) / f"{seq}_{cfg['gmc_method']}_{key}.npy"


def _warps(seq: str, fr: pd.DataFrame, cfg: dict, warp_cache: Path, log: Log) -> np.ndarray | None:
    """BoT-SORT camera warps of one sequence, cached per (sequence, method, frame list)."""
    path = warp_path(seq, fr, cfg, warp_cache)
    if path is None:
        return None
    method = cfg["gmc_method"]
    if path.exists():
        w = np.load(path)
        if w.shape != (len(fr), 2, 3):
            raise ValueError(f"{path}: warps of shape {w.shape} for {len(fr)} frames; delete it")
        return w
    log(f"[{seq}] GMC {method}: reading {len(fr)} frames")
    w = gmc_warps(fr.path.tolist(), method=method)
    _save_atomic(path, lambda f: np.save(f, w))
    return w


def track_set(
    s: SetData, dets: pd.DataFrame, cfg: dict, warp_cache: Path, log: Log = _quiet
) -> pd.DataFrame:
    """Run one tracker over every sequence of the set (empty frames included)."""
    by_seq = {k: g for k, g in dets.groupby("seq_id")}
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in DET_COLS})
    parts = []
    for seq, fr in s.frames.groupby("seq_id", sort=True):
        warps = _warps(seq, fr, cfg, warp_cache, log)
        d = by_seq[seq][DET_COLS] if seq in by_seq else empty
        parts.append(track_sequence(d, fr.frame.tolist(), cfg, seq_id=seq, warps=warps,
                                    classes=s.classes))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in TRACK_COLS})
    return pd.concat(parts, ignore_index=True)[TRACK_COLS]


def tracks_path(out_root: Path, name: str, set_name: str) -> Path:
    return Path(out_root) / name / f"{set_name}.parquet"


def run_tracking(
    model: str, tracker_cfg: Path, set_name: str, *, models_cfg: dict, manifest_dir: Path,
    hash_cache: Path, pred_cache: Path, out_root: Path = Path("data/tracks"),
    warp_cache: Path = Path("data/cache/gmc"), log: Log = _quiet,
) -> Path:
    """Track one set on cached detections -> `out_root/<model>+<tracker name>/<set>.parquet`
    (TRACK_COLS) plus a json sidecar (tracker cfg hash, detection cache_tag, manifest hash)."""
    cfg = load_tracker_cfg(tracker_cfg)
    images, boxes, mhash = load_checked_manifest([set_name], manifest_dir, hash_cache, log)
    s = load_set(set_name, images, boxes, load_mot(manifest_dir)[0])
    dets, tag = cached_detections(model, models_cfg, s, images, boxes, pred_cache, log)
    log(f"[{set_name}] {cfg['name']}: tracking {len(s.sequences)} sequences, {len(dets)} detections")
    tracks = track_set(s, dets, cfg, warp_cache, log)
    out = tracks_path(out_root, f"{model}+{cfg['name']}", set_name)
    meta_path = out.with_suffix(".json")
    meta_path.unlink(missing_ok=True)  # written last: no sidecar = incomplete run
    write_parquet_atomic(tracks, out)
    meta = {"model": model, "tracker": cfg["name"], "tracker_cfg": cfg,
            "tracker_cfg_hash": stable_hash(cfg), "detections_cache_tag": tag,
            "manifest_hash": mhash, "set_name": set_name, "set_hash": set_hash_of(s.im),
            "classes": list(s.classes), "imgsz": det_imgsz(models_cfg, model),
            "git_commit": git_commit()}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                         newline="\n")
    return out


def load_tracks(
    out_root: Path, name: str, set_name: str, cfg: dict | None, manifest_hash: str
) -> tuple[pd.DataFrame, dict]:
    """Tracks written by `run_tracking`, checked against the tracker cfg and the manifest."""
    path = tracks_path(out_root, name, set_name)
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise ValueError(f"no tracks for {name} on {set_name} at {path}: run `cvt track run` "
                         "first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cfg is not None and meta["tracker_cfg_hash"] != stable_hash(cfg):
        raise ValueError(f"{path} was made with a different tracker config; re-run `cvt track run`")
    if meta["manifest_hash"] != manifest_hash:
        raise ValueError(f"{path} was made against manifest {meta['manifest_hash']}, current is "
                         f"{manifest_hash}; re-run `cvt track run`")
    return pd.read_parquet(path), meta


# --- metrics -------------------------------------------------------------------------------

def _ignored(tracks: pd.DataFrame, ignore: pd.DataFrame) -> np.ndarray:
    """Per tracker row: True if >= IGNORE_MIN_INSIDE of its area lies inside one ignore region
    of the same (sequence, frame)."""
    out = np.zeros(len(tracks), bool)
    if tracks.empty or ignore.empty:
        return out
    key = {"seq_id": str, "frame": "int64"}
    t = tracks[["seq_id", "frame", "x1", "y1", "x2", "y2"]].astype(key).reset_index(drop=True)
    t["_row"] = np.arange(len(t))
    m = t.merge(ignore[["seq_id", "frame", "x1", "y1", "x2", "y2"]].astype(key),
                on=["seq_id", "frame"], suffixes=("", "_ig"))
    iw = (np.minimum(m.x2, m.x2_ig) - np.maximum(m.x1, m.x1_ig)).clip(lower=0)
    ih = (np.minimum(m.y2, m.y2_ig) - np.maximum(m.y1, m.y1_ig)).clip(lower=0)
    area = (m.x2 - m.x1) * (m.y2 - m.y1)
    inside = (area > 0) & (iw * ih >= IGNORE_MIN_INSIDE * area)
    out[m.loc[inside, "_row"].to_numpy()] = True
    return out


def _mot(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in MOT_COLS})
    return df.rename(columns={"track_id": "id"})[MOT_COLS]


def sequence_class_stats(
    tracks: pd.DataFrame, gt: pd.DataFrame, ignore: pd.DataFrame, classes: Sequence[int],
    sequences: Sequence[str],
) -> list[dict[int, SeqStats]]:
    """Per sequence (in order), per class: SeqStats after dropping ignored tracker boxes."""
    tr = tracks[~_ignored(tracks, ignore)]
    tr_by = {k: g for k, g in tr.groupby(["seq_id", "label"])} if len(tr) else {}
    gt_by = {k: g for k, g in gt.groupby(["seq_id", "label"])} if len(gt) else {}
    return [{int(c): sequence_stats(_mot(gt_by.get((seq, c))), _mot(tr_by.get((seq, c))))
             for c in classes} for seq in sequences]


def _combined(per_seq: list[dict[int, SeqStats]]) -> list[SeqStats]:
    return [sum(st.values(), SeqStats.zero()) for st in per_seq]


def _weighted_sum(stats: list[SeqStats], w: np.ndarray | None) -> SeqStats:
    """Sum over sequences; `w` = bootstrap multiplicities (non-negative integers)."""
    tot = SeqStats.zero()
    for i, s in enumerate(stats):
        k = 1 if w is None else int(w[i])
        if k:
            tot = tot + (s if k == 1 else s.scaled(k))
    return tot


def _metric_fn(per_seq: list[dict[int, SeqStats]], per_class: Sequence[int],
               metrics: Sequence[str] = METRICS):
    comb = _combined(per_seq)
    by_class = {c: [st[c] for st in per_seq] for c in per_class}

    def fn(w: np.ndarray | None) -> dict[str, float]:
        head = summarize(_weighted_sum(comb, w))
        out = {m: head[m] for m in metrics}
        for c, stats in by_class.items():
            out[f"HOTA/{CLASSES[c]}"] = summarize(_weighted_sum(stats, w))["HOTA"]
        return out

    return fn


def slice_sequences(seq_attrs: pd.DataFrame | None, sequences: Sequence[str]) -> dict[str, list[int]]:
    """Slice name -> indices into `sequences` of the sequences with that attribute flag set."""
    if seq_attrs is None or seq_attrs.empty:
        return {}
    if seq_attrs.seq_id.duplicated().any():
        raise ValueError("seq_attrs lists a sequence twice")
    missing = [c for c in SLICES if c not in seq_attrs.columns]
    if missing:
        raise ValueError(f"seq_attrs is missing slice columns {missing}")
    attrs = seq_attrs.set_index("seq_id")
    return {name: [i for i, s in enumerate(sequences) if s in attrs.index and bool(attrs.at[s, name])]
            for name in SLICES}


def _ci(fn, n_units: int, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """Sequence bootstrap; no CI with fewer than 2 sequences."""
    if n_units < 2 or n_boot <= 0:
        return {}
    return bootstrap_ci(fn, np.arange(n_units), n_boot=n_boot, seed=seed)


def evaluate_tracking(
    tracks: pd.DataFrame, gt: pd.DataFrame, ignore: pd.DataFrame, classes: Sequence[int],
    seq_attrs: pd.DataFrame | None, n_boot: int = 1000, seed: int = 0, *,
    sequences: Sequence[str] | None = None, frames_per_seq: Mapping[str, int] | None = None,
) -> list[dict]:
    """Rows {slice, metric, value, ci_lo, ci_hi, n_scenes[, n_images]}: class-combined METRICS
    and per-class `HOTA/<class>` (classes with GT) for slice `all`, plus class-combined METRICS
    per non-empty attribute slice. CIs: cluster bootstrap over whole sequences of the summed
    SeqStats (NaN with < 2 sequences). `sequences` defaults to those in gt or tracks."""
    classes = [int(c) for c in classes]
    if sequences is None:
        sequences = sorted(set(gt.seq_id) | set(tracks.seq_id))
    seqs = list(sequences)
    per_seq = sequence_class_stats(tracks, gt, ignore, classes, seqs)
    with_gt = [c for c in classes if sum(st[c].n_gt for st in per_seq) > 0]
    groups = [("all", list(range(len(seqs))))]
    groups += [(k, idx) for k, idx in slice_sequences(seq_attrs, seqs).items() if idx]
    rows = []
    for name, idx in groups:
        fn = _metric_fn([per_seq[i] for i in idx], with_gt if name == "all" else [])
        ci = _ci(fn, len(idx), n_boot, seed)
        extra = {"n_scenes": len(idx)}
        if frames_per_seq is not None:
            extra["n_images"] = int(sum(frames_per_seq.get(seqs[i], 0) for i in idx))
        for metric, value in fn(None).items():
            lo, hi = ci.get(metric, (NAN, NAN))
            rows.append({"slice": name, "metric": metric, "value": value, "ci_lo": lo,
                         "ci_hi": hi, **extra})
    return rows


def compare_tracking(
    tracks_a: pd.DataFrame, tracks_b: pd.DataFrame, gt: pd.DataFrame, ignore: pd.DataFrame,
    classes: Sequence[int], n_boot: int = 1000, seed: int = 0, *,
    sequences: Sequence[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """`delta/<metric>` -> (a - b, lo, hi), class-combined, paired bootstrap over sequences."""
    classes = [int(c) for c in classes]
    if sequences is None:
        sequences = sorted(set(gt.seq_id) | set(tracks_a.seq_id) | set(tracks_b.seq_id))
    seqs = list(sequences)
    fa = _metric_fn(sequence_class_stats(tracks_a, gt, ignore, classes, seqs), [], COMPARE_METRICS)
    fb = _metric_fn(sequence_class_stats(tracks_b, gt, ignore, classes, seqs), [], COMPARE_METRICS)
    if len(seqs) < 2 or n_boot <= 0:
        pa, pb = fa(None), fb(None)
        res = {k: (pa[k] - pb[k], NAN, NAN) for k in COMPARE_METRICS}
    else:
        res = paired_bootstrap(fa, fb, np.arange(len(seqs)), n_boot=n_boot, seed=seed)
    return {f"delta/{k}": tuple(float(x) for x in res[k]) for k in COMPARE_METRICS}


# --- tuning --------------------------------------------------------------------------------

def _grid(grid: str) -> tuple[dict[str, tuple], str]:
    if grid not in GRIDS:
        raise ValueError(f"unknown tracker grid {grid!r} (known: {list(GRIDS)})")
    return GRIDS[grid]


def tracker_grid(tracker_type: str, grid: str = "plan3") -> list[dict]:
    """Grid points (rule 6 = `plan3`, or `3b`) in declaration order (last key varies fastest)."""
    if tracker_type not in ("bytetrack", "botsort"):
        raise ValueError(f"tracker type must be bytetrack or botsort, got {tracker_type!r}")
    values = _grid(grid)[0]
    keys = list(values)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*values.values())]


def tuned_cfg(base: dict, point: dict, suffix: str = "tuned") -> dict:
    cfg = {**base, **GRID_FIXED, **point, "name": f"{base['tracker_type']}-{suffix}"}
    if base["tracker_type"] == "botsort":
        cfg.update(GRID_FIXED_BOTSORT)
    return cfg


def grid_report_name(tracker_type: str, grid: str = "plan3", model: str = "") -> str:
    """`tracker_grid_<type>.csv` (grid plan3, unchanged); `tracker_grid_<type>_<grid>_<detector>.csv`
    for any other grid, which is tuned per detector."""
    _grid(grid)
    return (f"tracker_grid_{tracker_type}.csv" if grid == "plan3"
            else f"tracker_grid_{tracker_type}_{grid}_{model}.csv")


def tune_tracker(
    model: str, tracker_type: str, set_name: str = "MOT-tune", *, models_cfg: dict,
    manifest_dir: Path, hash_cache: Path, pred_cache: Path, base_cfg: Path | None = None,
    grid: list[dict] | None = None, grid_name: str = "plan3", reports_dir: Path = Path("reports"),
    warp_cache: Path = Path("data/cache/gmc"), log: Log = _quiet,
) -> tuple[dict, pd.DataFrame]:
    """Run the grid `grid_name` (or explicit `grid` points) on `set_name` (never a test set);
    best = max class-combined HOTA, ties -> first in grid order, named `<type>-<suffix>` of the
    grid. Writes `reports_dir/<grid_report_name(...)>`."""
    role = SET_ROLES.get(set_name)
    if role != "tune":
        raise ValueError(f"{set_name} has role {role!r}: trackers are tuned only on a tune set "
                         "(MOT-tune), never on a test set")
    suffix = _grid(grid_name)[1]
    # also validates the type
    grid = tracker_grid(tracker_type, grid_name) if grid is None else grid
    base_cfg = Path(base_cfg or Path("configs/trackers") / f"{tracker_type}.yaml")
    base = load_tracker_cfg(base_cfg)
    if base["tracker_type"] != tracker_type:
        raise ValueError(f"{base_cfg} is a {base['tracker_type']} config, not {tracker_type}")
    images, boxes, _ = load_checked_manifest([set_name], manifest_dir, hash_cache, log)
    s = load_set(set_name, images, boxes, load_mot(manifest_dir)[0])
    dets, _ = cached_detections(model, models_cfg, s, images, boxes, pred_cache, log)
    rows = []
    for i, point in enumerate(grid):
        tracks = track_set(s, dets, tuned_cfg(base, point, suffix), warp_cache, log)
        per_seq = sequence_class_stats(tracks, s.gt, s.ignore, s.classes, s.sequences)
        res = _metric_fn(per_seq, [])(None)
        log(f"[{set_name}] {tracker_type} {i + 1}/{len(grid)} {point}: HOTA {res['HOTA']:.4f}")
        rows.append({**point, **res})
    table = pd.DataFrame(rows)
    out = Path(reports_dir) / grid_report_name(tracker_type, grid_name, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    hota = table["HOTA"].to_numpy(float)
    if np.isnan(hota).all():
        raise ValueError(f"HOTA is undefined for every grid point on {set_name} (no GT, no tracks)")
    return tuned_cfg(base, grid[int(np.nanargmax(hota))], suffix), table


# --- counting ------------------------------------------------------------------------------

CELL_INDEX = ["line", "label", "direction"]


def _anchors(
    df: pd.DataFrame, H: np.ndarray, valid: np.ndarray, frames: Sequence[int],
    sizes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Counting points: box centres in reference-frame pixels; non-finite ones are invalid. With
    `sizes` ((W, H) per frame, aligned with `frames`; rule 2) so are boxes at the image border."""
    st = stabilize_tracks(df, H, valid, frames)
    x, y = st["cx_ref"].to_numpy(float), st["cy_ref"].to_numpy(float)
    ok = st["valid"].to_numpy(bool) & np.isfinite(x) & np.isfinite(y)
    if sizes is not None and len(st):
        index = {int(f): i for i, f in enumerate(frames)}
        wh = np.asarray(sizes, float)[[index[int(f)] for f in st["frame"]]]
        ok &= ~near_border(st["x1"], st["y1"], st["x2"], st["y2"], wh[:, 0], wh[:, 1])
    return pd.DataFrame({"track_id": st["track_id"].to_numpy(), "frame": st["frame"].to_numpy(),
                         "x": x, "y": y, "label": st["label"].to_numpy(), "valid": ok})


def _line_counts(points: pd.DataFrame, lines, first_only: bool = False) -> pd.DataFrame:
    """Counts per (line, label, direction); `first_only` (rule 2): one event per track per line."""
    keep = first_crossings if first_only else (lambda ev: ev)
    parts = [count_table(keep(crossings(points, line))).reset_index().assign(line=line.name)
             for line in lines]
    parts = [p for p in parts if len(p)]
    if not parts:
        idx = pd.MultiIndex.from_arrays([[], [], []], names=CELL_INDEX)
        return pd.DataFrame({"n": pd.Series([], dtype="int64")}, index=idx)
    return pd.concat(parts, ignore_index=True).set_index(CELL_INDEX)[["n"]]


def count_scene(
    tracks: pd.DataFrame, gt: pd.DataFrame, H: np.ndarray, valid: np.ndarray, scene: dict, *,
    frames: Sequence[int], classes: Sequence[int] | None = None,
    ignore: pd.DataFrame | None = None, rule: int = 1, sizes: np.ndarray | None = None,
) -> dict:
    """Predicted vs GT line counts of one scene (its sequence's rows only). `H`, `valid` (and
    `sizes`, (W, H) per frame) are aligned with `frames`. Tracker rows >= IGNORE_MIN_INSIDE inside
    an `ignore` region of their (sequence, frame) are dropped first (rule 4, as for the metrics).
    Counting `rule` 1 counts every crossing; rule 2 (needs `sizes`) also drops points whose box
    is at the image border and counts only a track's first crossing of each line, for GT and
    tracks alike. Returns {"errors": count_errors(...), "cells": DataFrame indexed (line, label,
    direction) with columns gt, pred}."""
    if rule not in COUNT_RULES:
        raise ValueError(f"counting rule must be one of {list(COUNT_RULES)}, got {rule!r}")
    if rule == 2 and (sizes is None or len(sizes) != len(frames)):
        raise ValueError("counting rule 2 needs the image sizes (W, H) of every frame")
    border = sizes if rule == 2 else None
    if ignore is not None:
        tracks = tracks[~_ignored(tracks, ignore)]
    seq = scene["seq_id"]
    tr = tracks[tracks.seq_id == seq] if "seq_id" in tracks else tracks
    g = gt[gt.seq_id == seq] if "seq_id" in gt else gt
    if classes is not None:
        tr, g = tr[tr.label.isin(list(classes))], g[g.label.isin(list(classes))]
    pc = _line_counts(_anchors(tr, H, valid, frames, border), scene["lines"], rule == 2)
    gc = _line_counts(_anchors(g, H, valid, frames, border), scene["lines"], rule == 2)
    gn, pn = gc["n"].align(pc["n"], join="outer", fill_value=0)
    cells = pd.DataFrame({"gt": gn.astype("int64"), "pred": pn.astype("int64")})
    return {"errors": count_errors(gc, pc), "cells": cells}


def load_scenes(scenes_dir: Path, sets: Sequence[str]) -> list[dict]:
    """Scene files (`*.yaml`) in `scenes_dir` whose set_name is one of `sets`."""
    scenes_dir = Path(scenes_dir)
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"scenes directory {scenes_dir} not found")
    scenes = [load_scene(p) for p in sorted(scenes_dir.glob("*.yaml"))]
    return [sc for sc in scenes if sc["set_name"] in sets]


def scene_homographies(
    s: SetData, scene: dict, dets: pd.DataFrame, det_tag: str, stab_cache: Path,
    log: Log = _quiet,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(H, valid, frame numbers) of the scene's sequence, reference = frame index `ref_frame`;
    cached detections (score >= MASK_SCORE) are masked out of the keypoints."""
    seq = scene["seq_id"]
    fr = s.frames[s.frames.seq_id == seq]
    if fr.empty:
        raise ValueError(f"scene {seq}: no such sequence in {s.name}")
    ref = int(scene["ref_frame"])
    if not 0 <= ref < len(fr):
        raise ValueError(f"scene {seq}: ref_frame {ref} outside 0..{len(fr) - 1}")
    frames = fr.frame.tolist()
    key = stable_hash({"frames": frames, "ref": ref, "detections": det_tag,
                       "mask_score": MASK_SCORE})[:12]
    path = Path(stab_cache) / f"{seq}_{key}.npz"
    if path.exists():
        z = np.load(path)
        return z["H"], z["valid"], frames
    d = dets[(dets.seq_id == seq) & (dets.score >= MASK_SCORE)]
    by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in d.groupby("frame")}
    masks = [by_frame.get(f, np.zeros((0, 4))) for f in frames]
    log(f"[{seq}] stabilizing {len(frames)} frames to frame index {ref}")
    H, valid = reference_homographies(fr.path.tolist(), boxes=masks, ref=ref)
    _save_atomic(path, lambda f: np.savez(f, H=H, valid=valid))
    return H, valid, frames


# --- result rows ---------------------------------------------------------------------------

def _notes(classes: Sequence[int]) -> str:
    return "classes: " + ",".join(CLASSES[c] for c in classes)


def _frame_sizes(s: SetData, seq: str, frames: Sequence[int]) -> np.ndarray:
    """(W, H) of each frame of a sequence, aligned with `frames` (manifest images table)."""
    fr = s.frames[s.frames.seq_id == seq]
    missing = [c for c in ("width", "height") if c not in fr]
    if missing:
        raise ValueError(f"{s.name}: frames have no {missing} (counting rule 2 needs image sizes)")
    return fr.set_index("frame")[["width", "height"]].loc[list(frames)].to_numpy(float)


def _scene_counts(
    s: SetData, tracks: pd.DataFrame, scenes: list[dict], dets: pd.DataFrame, det_tag: str,
    stab_cache: Path, log: Log, rules: Sequence[int] = (1,),
) -> Iterator[tuple[dict, dict[int, dict], int]]:
    """(scene, {rule: count_scene result}, number of frames) per scene, in order."""
    for sc in scenes:
        H, valid, frames = scene_homographies(s, sc, dets, det_tag, stab_cache, log)
        sizes = _frame_sizes(s, sc["seq_id"], frames) if 2 in rules else None
        res = {r: count_scene(tracks, s.gt, H, valid, sc, frames=frames, classes=s.classes,
                              ignore=s.ignore, rule=r, sizes=sizes) for r in rules}
        yield sc, res, len(frames)


def _count_rows(
    s: SetData, tracks: pd.DataFrame, scenes: list[dict], dets: pd.DataFrame, det_tag: str,
    stab_cache: Path, log: Log,
) -> list[dict]:
    """Rule-1 rows (count_MAE, count_pct_err per scene, then `all`), then the same rows for
    rule 2 (count2_MAE, count2_pct_err)."""
    per_scene = list(_scene_counts(s, tracks, scenes, dets, det_tag, stab_cache, log,
                                   rules=tuple(COUNT_RULES)))
    n_frames = sum(n for _, _, n in per_scene)
    rows = []
    for rule, prefix in COUNT_RULES.items():
        cells = {}
        for sc, res, n in per_scene:
            cells[sc["seq_id"]] = res[rule]["cells"]
            for m in ("count_MAE", "count_pct_err"):
                rows.append({"slice": sc["seq_id"], "metric": m.replace("count", prefix, 1),
                             "value": res[rule]["errors"][m], "ci_lo": NAN, "ci_hi": NAN,
                             "n_images": n, "n_scenes": 1})
        if cells:
            allc = pd.concat(cells, names=["scene"])
            err = count_errors(allc[["gt"]].rename(columns={"gt": "n"}),
                               allc[["pred"]].rename(columns={"pred": "n"}))
            for m in ("count_MAE", "count_pct_err"):
                rows.append({"slice": "all", "metric": m.replace("count", prefix, 1),
                             "value": err[m], "ci_lo": NAN, "ci_hi": NAN, "n_images": n_frames,
                             "n_scenes": len(cells)})
    return rows


def _track_detections(
    model: str, models_cfg: dict, s: SetData, images: pd.DataFrame, boxes: pd.DataFrame,
    pred_cache: Path, meta: dict, name: str, log: Log,
) -> tuple[pd.DataFrame, str]:
    """The cached detections the tracks were made from (their cache_tag must match)."""
    dets, tag = cached_detections(model, models_cfg, s, images, boxes, pred_cache, log)
    if tag != meta["detections_cache_tag"]:
        raise ValueError(f"tracks of {name} on {s.name} were made from detections "
                         f"{meta['detections_cache_tag']}, the cache now holds {tag}; "
                         "re-run `cvt track run`")
    return dets, tag


COUNT_CELL_COLS = ["eval_set", "scene", "line", "label", "class", "direction", "gt", "pred"]


def count_cells_sets(
    model: str, tracker_cfg: Path, sets: list[str], scenes_dir: Path, *, models_cfg: dict,
    manifest_dir: Path, hash_cache: Path, pred_cache: Path, out_root: Path = Path("data/tracks"),
    stab_cache: Path = Path("data/cache/stab"), rules: Sequence[int] = (1,), log: Log = _quiet,
) -> dict[int, pd.DataFrame]:
    """Per counting rule: per (scene, line, label, direction), GT and predicted line counts
    (COUNT_CELL_COLS) of tracks written by `run_tracking`, computed exactly as the count rows of
    `eval_tracking_sets` (same scenes, ignore rule and cached stabilization). Writes nothing to
    results."""
    bad = [r for r in rules if r not in COUNT_RULES]
    if bad:
        raise ValueError(f"counting rules must be among {list(COUNT_RULES)}, got {bad}")
    cfg = load_tracker_cfg(tracker_cfg)
    name = f"{model}+{cfg['name']}"
    scenes = load_scenes(scenes_dir, sets)
    if not scenes:
        raise ValueError(f"no scene files for {sets} in {scenes_dir}")
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    mot_gt, _ = load_mot(manifest_dir)
    parts: dict[int, list] = {r: [] for r in rules}
    for set_name in sets:
        set_scenes = [sc for sc in scenes if sc["set_name"] == set_name]
        if not set_scenes:
            log(f"[{set_name}] no scene files in {scenes_dir}: no count cells")
            continue
        s = load_set(set_name, images, boxes, mot_gt)
        tracks, meta = load_tracks(out_root, name, set_name, cfg, mhash)
        dets, tag = _track_detections(model, models_cfg, s, images, boxes, pred_cache, meta,
                                      name, log)
        for sc, res, _ in _scene_counts(s, tracks, set_scenes, dets, tag, stab_cache, log,
                                        rules=rules):
            for r in rules:
                c = res[r]["cells"].reset_index()
                parts[r].append(c.assign(eval_set=set_name, scene=sc["seq_id"],
                                         **{"class": [CLASSES[int(x)] for x in c["label"]]}))
    return {r: _cell_table(p) for r, p in parts.items()}


def _cell_table(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in COUNT_CELL_COLS})
    out = pd.concat(parts, ignore_index=True)[COUNT_CELL_COLS]
    return out.astype({"label": "int64", "gt": "int64", "pred": "int64"})


def eval_tracking_sets(
    model: str, tracker_cfg: Path, sets: list[str], *, models_cfg: dict, manifest_dir: Path,
    hash_cache: Path, pred_cache: Path, out_root: Path = Path("data/tracks"),
    scenes_dir: Path | None = None, stab_cache: Path = Path("data/cache/stab"),
    n_boot: int = 1000, seed: int = 0, log: Log = _quiet,
) -> Iterator[list[dict]]:
    """Result rows (RESULT_COLS) per set for tracks written by `run_tracking`."""
    cfg = load_tracker_cfg(tracker_cfg)
    name = f"{model}+{cfg['name']}"
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    mot_gt, seq_attrs = load_mot(manifest_dir)
    scenes = load_scenes(scenes_dir, sets) if scenes_dir is not None else []
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_commit()
    for set_name in sets:
        s = load_set(set_name, images, boxes, mot_gt)
        tracks, meta = load_tracks(out_root, name, set_name, cfg, mhash)
        log(f"[{set_name}] {name}: metrics + bootstrap ({n_boot}) over {len(s.sequences)} sequences")
        rows = evaluate_tracking(tracks, s.gt, s.ignore, s.classes, seq_attrs, n_boot, seed,
                                 sequences=s.sequences, frames_per_seq=s.frames_per_seq())
        set_scenes = [sc for sc in scenes if sc["set_name"] == set_name]
        if set_scenes:
            dets, tag = _track_detections(model, models_cfg, s, images, boxes, pred_cache, meta,
                                          name, log)
            rows += _count_rows(s, tracks, set_scenes, dets, tag, stab_cache, log)
        elif scenes_dir is not None:
            log(f"[{set_name}] no scene files in {scenes_dir}: no counting rows")
        chash = stable_hash({"tracker": cfg, "detections": meta["detections_cache_tag"],
                             "classes": list(s.classes), "n_boot": n_boot, "seed": seed})
        base = dict(timestamp=stamp, git_commit=commit, model=name,
                    model_sha=meta["detections_cache_tag"], eval_set=set_name,
                    imgsz=det_imgsz(models_cfg, model),
                    manifest_hash=mhash, set_hash=set_hash_of(s.im), config_hash=chash,
                    notes=_notes(s.classes))
        yield [{**base, **r} for r in rows]


def compare_tracking_sets(
    a: str, b: str, sets: list[str], *, manifest_dir: Path, hash_cache: Path,
    out_root: Path = Path("data/tracks"), n_boot: int = 1000, seed: int = 0,
    cross_detector: bool = False, log: Log = _quiet,
) -> Iterator[list[dict]]:
    """Paired sequence-bootstrap rows `delta/HOTA|MOTA|IDF1` (a - b) per set; a, b are
    `<model>+<tracker name>` directories under `out_root`. Both must be tracked on the same
    detections, unless `cross_detector` (a detector+tracker comparison: a and b come from
    different detectors; one detector's runs on different detections are still refused).
    Rows' imgsz is a's detector resolution (tracks sidecar; IMGSZ for runs made before 3b)."""
    if a == b:
        raise ValueError("comparing a tracker run with itself")
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    mot_gt, _ = load_mot(manifest_dir)
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_commit()
    for set_name in sets:
        s = load_set(set_name, images, boxes, mot_gt)
        ta, ma = load_tracks(out_root, a, set_name, None, mhash)
        tb, mb = load_tracks(out_root, b, set_name, None, mhash)
        other_det = ma["model"] != mb["model"]
        if ma["detections_cache_tag"] != mb["detections_cache_tag"] and not (
                cross_detector and other_det):
            hint = (" or pass --cross-detector to compare detector+tracker pipelines"
                    if other_det else "")
            raise ValueError(f"{set_name}: {a} was tracked on detections "
                             f"{ma['detections_cache_tag']}, {b} on "
                             f"{mb['detections_cache_tag']}; a paired tracker comparison needs "
                             f"the same detections (re-run `cvt track run` for one of them){hint}")
        log(f"[{set_name}] paired bootstrap ({n_boot}) {a} vs {b} over {len(s.sequences)} sequences")
        res = compare_tracking(ta, tb, s.gt, s.ignore, s.classes, n_boot, seed,
                               sequences=s.sequences)
        chash = stable_hash({"a": ma["tracker_cfg_hash"], "b": mb["tracker_cfg_hash"],
                             "detections": [ma["detections_cache_tag"], mb["detections_cache_tag"]],
                             "n_boot": n_boot, "seed": seed})
        yield [dict(timestamp=stamp, git_commit=commit, model=a,
                    model_sha=f"{ma['detections_cache_tag']}|{mb['detections_cache_tag']}",
                    eval_set=set_name, slice="all", imgsz=int(ma.get("imgsz", IMGSZ)), metric=k,
                    value=d, ci_lo=lo,
                    ci_hi=hi, n_images=len(s.frames), n_scenes=len(s.sequences),
                    manifest_hash=mhash, set_hash=set_hash_of(s.im), config_hash=chash,
                    notes=f"vs {b}; {_notes(s.classes)}")
                for k, (d, lo, hi) in res.items()]
