"""Lock tests on our own GT: the pre-declared tuning on MOT-tune and the frozen evaluation.

Every set reuses the re-tuned tracking's cached detections, the BoT-SORT-3b tracks and their GMC warps, plus the
detector features of `cvt follow features`. Lock tests are GT vehicle tracks spanning >= `min_s`
seconds at the set's `fps` (configs/follow.yaml); each is initialised from its GT box in its first
frame and runs to its last frame. Frames are read from disk only by the MIL fallback and the hsv
variant. Nothing here runs a model, except `detect_sot`.

The public benchmarks (UAVDT-S, UAV123-veh, UAV20L-veh; configs/follow_sot.yaml) run end to end:
`detect_sot` (GPU) caches the detector's boxes and features per video, `eval_sot_sets` (CPU)
tracks each sequence with GMC warps and scores one lock test per sequence at the same frozen
params, after a pHash leakage check of UAVDT-S against the training-side images.
"""

from __future__ import annotations

import bisect
import itertools
import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from cvt.data.manifest import SET_ROLES, load_mot
from cvt.eval.run import load_checked_manifest, set_hash_of
from cvt.follow.features import _read_meta, features_path, load_features
from cvt.follow.gimbal import GimbalParams, follow, gimbal_metrics
from cvt.follow.lock import VARIANTS, Feats, LockParams, iou_one, lock_params, run_lock
from cvt.follow.ope import aggregate, lock_targets, ope_scores
from cvt.track import pipeline as tp
from cvt.track.run import load_tracker_cfg
from cvt.utils import file_sha256, git_commit, stable_hash, write_parquet_atomic

Log = Callable[[str], None]

BOX = ["x1", "y1", "x2", "y2"]
FEATURE_IOU = 0.5  # a track box takes the feature of its best-IoU detection at or above this
LOCK_KEYS = ("success_auc", "precision_20", "norm_precision_auc", "lock_kept",
             "wrong_frame_rate", "wrong_switches_per_target")
GIMBAL_KEYS = ("centre_err", "inside", "smooth", "clamped_frac")
TIE_FRAC = 0.05  # gimbal: points within 5% of the best centring error tie -> min smoothness
FULL = "features"  # the full lock variant: the tuning objective and the gimbal-lock input
FEATURES_HINT = r"run `scripts\follow.ps1 -Step features` (GPU) first"
NAN = float("nan")


def _quiet(msg: str) -> None:
    pass


# --- config ---------------------------------------------------------------------------------

def load_follow_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    need = ("detector", "models", "tracker", "fps", "min_s", "hfov_deg", "lock_grid", "gimbal_grid")
    missing = [k for k in need if k not in (cfg or {})]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if list(cfg["lock_grid"]) != ["coast_s", "reid_thr", "gate_growth"]:
        raise ValueError(f"{path}: lock_grid keys must be coast_s, reid_thr, gate_growth")
    if list(cfg["gimbal_grid"]) != ["kp", "kd", "target_frac"]:
        raise ValueError(f"{path}: gimbal_grid keys must be kp, kd, target_frac")
    return cfg


def _grid(values: Mapping[str, list]) -> list[dict]:
    keys = list(values)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*values.values())]


def lock_grid(cfg: dict) -> list[dict]:
    """Lock grid points in declaration order (last key varies fastest)."""
    return _grid(cfg["lock_grid"])


def gimbal_grid(cfg: dict) -> list[dict]:
    return _grid(cfg["gimbal_grid"])


def gimbal_params(cfg: dict, point: Mapping) -> GimbalParams:
    return GimbalParams(kp=float(point["kp"]), kd=float(point["kd"]),
                        target_frac=float(point["target_frac"]), hfov_deg=float(cfg["hfov_deg"]))


def set_fps(cfg: dict, set_name: str) -> float:
    if set_name not in cfg["fps"]:
        raise ValueError(f"no fps for {set_name} in the follow config (fps: {cfg['fps']})")
    return float(cfg["fps"][set_name])


# --- features -------------------------------------------------------------------------------

def seq_feature_fn(sub: pd.DataFrame, feats: np.ndarray) -> Feats:
    """`feats(frame, boxes)` for `run_lock` from one sequence's featured detections `sub` (rows of
    the cached detections) and their stored features (one row each): (N, C) float32
    L2-normalised rows, each the feature of the featured detection of the same frame with the best
    IoU >= FEATURE_IOU with the box (NaN rows where none); None when no box matches. Only this
    slice is converted to float32 and normalised."""
    f = np.asarray(feats)
    f = (f if f.ndim == 2 else f.reshape(len(sub), -1)).astype(np.float32)
    if len(f) != len(sub):
        raise ValueError(f"{len(f)} feature rows for {len(sub)} detections")
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(sub):
        frames = sub["frame"].to_numpy(np.int64)
        boxes = sub[BOX].to_numpy(float)
        for fr in np.unique(frames):
            k = frames == fr
            by_frame[int(fr)] = (boxes[k], f[k])
    dim = f.shape[1]

    def fn(frame: int, boxes: np.ndarray) -> np.ndarray | None:
        hit = by_frame.get(int(frame))
        b = np.asarray(boxes, float).reshape(-1, 4)
        if hit is None or len(b) == 0:
            return None
        dboxes, dfeat = hit
        out = np.full((len(b), dim), np.nan, np.float32)
        any_hit = False
        for i, box in enumerate(b):
            iou = iou_one(box, dboxes)
            j = int(np.argmax(iou))
            if iou[j] >= FEATURE_IOU:
                out[i] = dfeat[j]
                any_hit = True
        return out if any_hit else None

    return fn


def feature_fns(
    dets: pd.DataFrame, det_rows: np.ndarray, feats: np.ndarray, seqs,
) -> dict[str, Feats]:
    """`seq_feature_fn` per sequence of `seqs`: the featured detections are selected once and
    split by sequence once; each sequence converts only its own slice. `det_rows` index the rows
    of `dets` (the frame returned by `cached_detections`)."""
    det_rows = np.asarray(det_rows, np.int64)
    f = np.asarray(feats)
    if f.ndim != 2:
        f = f.reshape(len(det_rows), -1)
    sub = dets.iloc[det_rows]
    pos = sub.groupby("seq_id", sort=False, observed=True).indices if len(sub) else {}
    none = np.zeros(0, np.int64)
    return {seq: seq_feature_fn(sub.iloc[pos.get(seq, none)], f[pos.get(seq, none)])
            for seq in seqs}


def track_feature_fn(
    dets: pd.DataFrame, det_rows: np.ndarray, feats: np.ndarray, seq_id: str,
) -> Feats:
    """`seq_feature_fn` of one sequence (see `feature_fns`)."""
    return feature_fns(dets, det_rows, feats, [seq_id])[seq_id]


# --- one set --------------------------------------------------------------------------------

@dataclass
class FollowSet:
    """One set ready for lock runs: lock targets plus, per sequence with targets, its frame list,
    tracks, GMC warps, frame paths, frame size and feature function."""

    name: str
    pipeline: str  # `<detector>+<tracker name>`
    s: tp.SetData
    targets: pd.DataFrame  # seq_id, track_id, label, first, last
    fps: float
    det_tag: str
    tracks_meta: dict
    features_meta: dict
    manifest_hash: str
    frames: dict[str, list[int]] = field(default_factory=dict)
    tracks: dict[str, pd.DataFrame] = field(default_factory=dict)
    warps: dict[str, np.ndarray | None] = field(default_factory=dict)
    paths: dict[str, dict[int, str]] = field(default_factory=dict)
    size: dict[str, tuple[int, int]] = field(default_factory=dict)
    feats: dict[str, Feats] = field(default_factory=dict)
    gt: dict[str, pd.DataFrame] = field(default_factory=dict)

    def read_frame(self, seq: str) -> Callable[[int], np.ndarray]:
        paths = self.paths[seq]

        def read(frame: int) -> np.ndarray:
            img = cv2.imread(paths[int(frame)])
            if img is None:
                raise FileNotFoundError(f"cannot read frame {frame} of {seq}: {paths[int(frame)]}")
            return img

        return read

    def span(self, seq: str, first: int, last: int) -> tuple[int, int]:
        """Index range [lo, hi] of frames first..last in the sequence's frame list."""
        fr = self.frames[seq]
        return bisect.bisect_left(fr, first), bisect.bisect_right(fr, last) - 1


def _offline() -> None:
    """Ultralytics' offline guards: run_tracking and GMC load Ultralytics."""
    from cvt.train import job

    job.no_network()


def _ensure_tracks(cfg: dict, tcfg: dict, set_name: str, *, models_cfg: dict, manifest_dir: Path,
                   hash_cache: Path, pred_cache: Path, out_root: Path, warp_cache: Path,
                   log: Log) -> None:
    """Track the set with the frozen tracker when its tracks are missing (CPU, cached dets)."""
    path = tp.tracks_path(out_root, f"{cfg['detector']}+{tcfg['name']}", set_name)
    if path.exists() and path.with_suffix(".json").exists():
        return
    log(f"[{set_name}] no {tcfg['name']} tracks at {path}: tracking now")
    _offline()
    tp.run_tracking(cfg["detector"], Path(cfg["tracker"]), set_name, models_cfg=models_cfg,
                    manifest_dir=manifest_dir, hash_cache=hash_cache, pred_cache=pred_cache,
                    out_root=out_root, warp_cache=warp_cache, log=log)


def prepare_set(
    cfg: dict, set_name: str, *, models_cfg: dict, manifest_dir: Path, hash_cache: Path,
    pred_cache: Path, out_root: Path = Path("data/tracks"),
    warp_cache: Path = Path("data/cache/gmc"), feats_dir: Path = Path("data/cache/feats"),
    loaded: tuple[pd.DataFrame, pd.DataFrame, str] | None = None, log: Log = _quiet,
) -> FollowSet:
    """Load everything one set's lock runs need; tracks are made first if missing."""
    det = cfg["detector"]
    tcfg = load_tracker_cfg(Path(cfg["tracker"]))
    name = f"{det}+{tcfg['name']}"
    fps = set_fps(cfg, set_name)
    images, boxes, mhash = loaded or load_checked_manifest([set_name], manifest_dir, hash_cache, log)
    s = tp.load_set(set_name, images, boxes, load_mot(manifest_dir)[0])
    _ensure_tracks(cfg, tcfg, set_name, models_cfg=models_cfg, manifest_dir=manifest_dir,
                   hash_cache=hash_cache, pred_cache=pred_cache, out_root=out_root,
                   warp_cache=warp_cache, log=log)
    tracks, meta = tp.load_tracks(out_root, name, set_name, tcfg, mhash)
    dets, tag = tp.cached_detections(det, models_cfg, s, images, boxes, pred_cache, log)
    if tag != meta["detections_cache_tag"]:
        raise ValueError(f"tracks of {name} on {set_name} were made from detections "
                         f"{meta['detections_cache_tag']}, the cache now holds {tag}; delete "
                         f"{tp.tracks_path(out_root, name, set_name)} and re-run")
    fpath = features_path(feats_dir, det, tag, set_name)
    fmeta = _read_meta(fpath) if fpath.exists() else None
    if fmeta is None:
        raise ValueError(f"no detector features of {det} on {set_name} at {fpath}; {FEATURES_HINT}")
    if fmeta.get("det_tag") != tag or fmeta.get("set_hash") != set_hash_of(s.im):
        raise ValueError(f"{fpath} was made for other detections or another image list; delete it "
                         f"and {FEATURES_HINT}")
    det_rows, feat = load_features(fpath)
    if len(det_rows) and (det_rows.min() < 0 or det_rows.max() >= len(dets)):
        raise ValueError(f"{fpath} names detection rows outside the {len(dets)} cached detections")

    targets = lock_targets(s.gt, fps, float(cfg["min_s"]))
    fs = FollowSet(set_name, name, s, targets, fps, tag, meta, fmeta, mhash)
    tr_by = {k: g for k, g in tracks.groupby("seq_id")} if len(tracks) else {}
    gt_by = {k: g for k, g in s.gt.groupby("seq_id")}
    seqs = sorted(targets["seq_id"].unique())
    fs.feats = feature_fns(dets, det_rows, feat, seqs)
    for seq in seqs:
        fr = s.frames[s.frames.seq_id == seq]
        fs.frames[seq] = [int(f) for f in fr.frame]
        fs.paths[seq] = dict(zip(fs.frames[seq], fr.path.astype(str), strict=True))
        fs.size[seq] = (int(fr.width.iloc[0]), int(fr.height.iloc[0]))
        wpath = tp.warp_path(seq, fr, tcfg, warp_cache)
        if wpath is not None and not wpath.exists():
            _offline()  # GMC (Ultralytics) runs on the frames
        fs.warps[seq] = tp._warps(seq, fr, tcfg, warp_cache, log)
        fs.tracks[seq] = tr_by.get(seq, pd.DataFrame(columns=tracks.columns))
        fs.gt[seq] = gt_by[seq]
    log(f"[{set_name}] {len(targets)} lock tests in {targets['seq_id'].nunique()} sequences "
        f"(vehicle GT tracks >= {cfg['min_s']} s at {fps:g} fps)")
    return fs


def _target_gt(fs: FollowSet, seq: str, tid: int, first: int, last: int):
    g = fs.gt[seq]
    g = g[(g.frame >= first) & (g.frame <= last)]
    own = g.track_id == tid
    return g[own].sort_values("frame"), g[~own]


def _check_unique_targets(targets: pd.DataFrame) -> None:
    """Lock tests, per-target rows and lock outputs are keyed (seq_id, track_id)."""
    dup = targets.duplicated(["seq_id", "track_id"], keep=False)
    if dup.any():
        keys = sorted({(str(a), int(b)) for a, b in
                       targets.loc[dup, ["seq_id", "track_id"]].itertuples(index=False)})
        raise ValueError(f"lock targets are not unique per (seq_id, track_id): {keys[:5]}")


def variant_of(params: LockParams) -> str:
    """The `VARIANTS` name of `params` ("custom" when none matches)."""
    for name, v in VARIANTS.items():
        if all(getattr(params, k) == x for k, x in v.items()):
            return name
    return "custom"


PROGRESS_EVERY = 50  # targets between progress lines inside one variant run


def run_locks(
    fs: FollowSet, params: LockParams, log: Log = _quiet,
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    """Every lock test of the set: per-target OPE scores (seq_id, track_id, label, ope_scores
    keys) and the lock outputs keyed (seq_id, track_id)."""
    _check_unique_targets(fs.targets)
    rows, outs = [], {}
    n, variant = len(fs.targets), variant_of(params)
    for k, t in enumerate(fs.targets.itertuples(index=False), 1):
        seq, tid, first, last = t.seq_id, int(t.track_id), int(t.first), int(t.last)
        lo, hi = fs.span(seq, first, last)
        frames = fs.frames[seq][lo:hi + 1]
        warps = fs.warps[seq]
        tr = fs.tracks[seq]
        tr = tr[(tr.frame >= first) & (tr.frame <= last)]
        gt_track, others = _target_gt(fs, seq, tid, first, last)
        init_box = gt_track[BOX].iloc[0].to_numpy(float)
        out = run_lock(tr, fs.feats[seq], None if warps is None else warps[lo:hi + 1], frames,
                       first, init_box, params, fs.fps, fs.read_frame(seq))
        if (seq, tid) in outs:
            raise ValueError(f"duplicate lock target ({seq}, {tid})")
        outs[(seq, tid)] = out
        rows.append({"seq_id": seq, "track_id": tid, "label": int(t.label),
                     **ope_scores(out, gt_track, others)})
        if k % PROGRESS_EVERY == 0:
            log(f"[{fs.name}] lock-{variant}: target {k}/{n}")
    log(f"[{fs.name}] lock {params}: {len(rows)} targets")
    return pd.DataFrame(rows), outs


def run_gimbal(
    fs: FollowSet, params: GimbalParams,
    outs: Mapping[tuple[str, int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Per-target gimbal metrics, driven by the GT box (`outs` None) or by the lock outputs."""
    _check_unique_targets(fs.targets)
    rows = []
    for t in fs.targets.itertuples(index=False):
        seq, tid, first, last = t.seq_id, int(t.track_id), int(t.first), int(t.last)
        lo, hi = fs.span(seq, first, last)
        gt_track, _ = _target_gt(fs, seq, tid, first, last)
        if outs is None:  # one row per frame of the run; NaN where the GT box is missing
            frames = pd.DataFrame({"frame": fs.frames[seq][lo:hi + 1]})
            boxes = frames.merge(gt_track[["frame", *BOX]], on="frame", how="left")
        else:
            boxes = outs[(seq, tid)][["frame", *BOX]]
        w, h = fs.size[seq]
        fdf = follow(boxes, w, h, fs.fps, params)
        rows.append({"seq_id": seq, "track_id": tid, "n_frames": float(len(gt_track)),
                     **gimbal_metrics(fdf, gt_track, w, h)})
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame, keys, n_boot: int = 1000,
              seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """(value, lo, hi) per key via `ope.aggregate` (frame-weighted, sequence bootstrap); no CI
    with fewer than 2 sequences or n_boot <= 0. A NaN per-target metric raises ValueError."""
    df = per.copy()
    for c in ("wrong_frames", "wrong_switches"):  # gimbal tables have no wrong-target counts
        if c not in df:
            df[c] = 0.0
    ci = df["seq_id"].nunique() >= 2 and n_boot > 0
    agg = aggregate(df, n_boot=n_boot if ci else 1, seed=seed)
    return {k: agg[k] if ci else (agg[k][0], NAN, NAN) for k in keys}


# --- tuning ---------------------------------------------------------------------------------

def pick_gimbal(table: pd.DataFrame) -> int:
    """Row of the min centring error; rows within TIE_FRAC of it -> min smoothness, then first."""
    ce = table["centre_err"].to_numpy(float)
    if np.isnan(ce).all():
        raise ValueError("centring error is undefined for every gimbal grid point")
    best = np.nanmin(ce)
    near = np.flatnonzero(ce <= best * (1 + TIE_FRAC) + 1e-12)
    sm = table["smooth"].to_numpy(float)[near]
    return int(near[0] if np.isnan(sm).all() else near[int(np.nanargmin(sm))])


def tune_follow(
    cfg: dict, set_name: str = "MOT-tune", *, models_cfg: dict, manifest_dir: Path,
    hash_cache: Path, pred_cache: Path, out_root: Path = Path("data/tracks"),
    warp_cache: Path = Path("data/cache/gmc"), feats_dir: Path = Path("data/cache/feats"),
    reports_dir: Path = Path("reports"), log: Log = _quiet,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The pre-declared lock and gimbal grids on `set_name` (a tune set only). Writes
    `reports_dir/follow_grid_{lock,gimbal}.csv`; returns (tuned params, lock grid, gimbal grid).

    Lock: max frame-weighted success AUC of the `features` variant, ties -> first in grid order.
    Gimbal (GT-box input): min centring error, ties within 5% -> min smoothness."""
    role = SET_ROLES.get(set_name)
    if role != "tune":
        raise ValueError(f"{set_name} has role {role!r}: follow parameters are tuned only on a "
                         "tune set (MOT-tune), never on a test set")
    fs = prepare_set(cfg, set_name, models_cfg=models_cfg, manifest_dir=manifest_dir,
                     hash_cache=hash_cache, pred_cache=pred_cache, out_root=out_root,
                     warp_cache=warp_cache, feats_dir=feats_dir, log=log)
    if fs.targets.empty:
        raise ValueError(f"{set_name} has no lock tests (vehicle GT tracks >= {cfg['min_s']} s)")
    n_targets, n_frames = len(fs.targets), None
    lock_pts, lock_rows = lock_grid(cfg), []
    for i, point in enumerate(lock_pts):
        per, _ = run_locks(fs, lock_params(FULL, **point), log)
        n_frames = float(per["n_frames"].sum())
        res = {k: v[0] for k, v in summarize(per, LOCK_KEYS, n_boot=0).items()}
        log(f"[{set_name}] lock {i + 1}/{len(lock_pts)} {point}: success AUC "
            f"{res['success_auc']:.4f}")
        lock_rows.append({**point, **res, "n_targets": n_targets, "n_frames": n_frames})
    lock_table = pd.DataFrame(lock_rows)
    auc = lock_table["success_auc"].to_numpy(float)
    if np.isnan(auc).all():
        raise ValueError(f"success AUC is undefined for every lock grid point on {set_name}")
    best_lock = int(np.nanargmax(auc))

    gim_pts, gim_rows = gimbal_grid(cfg), []
    for i, point in enumerate(gim_pts):
        per = run_gimbal(fs, gimbal_params(cfg, point))
        res = {k: v[0] for k, v in summarize(per, GIMBAL_KEYS, n_boot=0).items()}
        log(f"[{set_name}] gimbal {i + 1}/{len(gim_pts)} {point}: centre err "
            f"{res['centre_err']:.4f}, smooth {res['smooth']:.4f}")
        gim_rows.append({**point, **res, "n_targets": n_targets, "n_frames": n_frames})
    gim_table = pd.DataFrame(gim_rows)
    best_gim = pick_gimbal(gim_table)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    lock_table.to_csv(reports_dir / "follow_grid_lock.csv", index=False, lineterminator="\n")
    gim_table.to_csv(reports_dir / "follow_grid_gimbal.csv", index=False, lineterminator="\n")
    tuned = {
        "set": set_name, "detector": cfg["detector"], "tracker": fs.pipeline.rsplit("+", 1)[1],
        "lock": dict(lock_pts[best_lock]), "gimbal": dict(gim_pts[best_gim]),
        "objective": {"lock_success_auc": float(auc[best_lock]),
                      "gimbal_centre_err": float(gim_table["centre_err"].iloc[best_gim]),
                      "gimbal_smooth": float(gim_table["smooth"].iloc[best_gim]),
                      "n_targets": n_targets},
    }
    return tuned, lock_table, gim_table


# --- evaluation -----------------------------------------------------------------------------

def check_tuned(cfg: dict, tuned: dict) -> None:
    """The frozen params must come from `cvt follow tune` on a tune set for this pipeline."""
    tname = load_tracker_cfg(Path(cfg["tracker"]))["name"]
    for k in ("set", "detector", "tracker", "lock", "gimbal"):
        if k not in tuned:
            raise ValueError(f"tuned follow params lack {k!r}: re-run `cvt follow tune`")
    if SET_ROLES.get(tuned["set"]) != "tune":
        raise ValueError(f"follow params were tuned on {tuned['set']}, not a tune set: re-run "
                         "`cvt follow tune`")
    if (tuned["detector"], tuned["tracker"]) != (cfg["detector"], tname):
        raise ValueError(f"follow params were tuned for {tuned['detector']}+{tuned['tracker']}, "
                         f"the config uses {cfg['detector']}+{tname}: re-run `cvt follow tune`")


def eval_follow_sets(
    cfg: dict, tuned: dict, sets: list[str], *, models_cfg: dict, manifest_dir: Path,
    hash_cache: Path, pred_cache: Path, out_root: Path = Path("data/tracks"),
    warp_cache: Path = Path("data/cache/gmc"), feats_dir: Path = Path("data/cache/feats"),
    n_boot: int = 1000, seed: int = 0, log: Log = _quiet,
) -> Iterator[list[dict]]:
    """Result rows (RESULT_COLS) per set: the 5 lock variants (`<pipeline>+lock-<variant>`,
    metrics lock/<LOCK_KEYS>) and the gimbal driven by the GT box and by the full lock
    (`<pipeline>+gimbal-gt|lock`, metrics gimbal/<GIMBAL_KEYS>), all at the frozen params."""
    check_tuned(cfg, tuned)
    loaded = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_commit()
    gparams = gimbal_params(cfg, tuned["gimbal"])
    for set_name in sets:
        fs = prepare_set(cfg, set_name, models_cfg=models_cfg, manifest_dir=manifest_dir,
                         hash_cache=hash_cache, pred_cache=pred_cache, out_root=out_root,
                         warp_cache=warp_cache, feats_dir=feats_dir, loaded=loaded, log=log)
        if fs.targets.empty:
            log(f"[{set_name}] no lock tests: no rows")
            continue
        base = dict(timestamp=stamp, git_commit=commit, model_sha=fs.det_tag, eval_set=set_name,
                    slice="all", imgsz=tp.det_imgsz(models_cfg, cfg["detector"]),
                    manifest_hash=fs.manifest_hash, set_hash=set_hash_of(fs.s.im))
        common = {"pipeline": fs.pipeline, "tracker_cfg_hash": fs.tracks_meta["tracker_cfg_hash"],
                  "detections": fs.det_tag, "features": stable_hash(fs.features_meta),
                  "fps": fs.fps, "min_s": cfg["min_s"], "n_boot": n_boot, "seed": seed}
        tests = (f"{len(fs.targets)} lock tests (vehicle GT tracks >= {cfg['min_s']} s at "
                 f"{fs.fps:g} fps); params tuned on {tuned['set']}")
        yield _set_rows(fs, tuned, gparams, base, common, tests, LOCK_KEYS, n_boot, seed, log)


def _set_rows(fs: FollowSet, tuned: dict, gparams: GimbalParams, base: dict, common: dict,
              tests: str, lock_keys, n_boot: int, seed: int, log: Log) -> list[dict]:
    """The 5 lock variants at the frozen lock point, then the gimbal driven by the GT box and by
    the full lock, at the frozen gimbal point."""
    rows: list[dict] = []
    full_outs = None
    for v in VARIANTS:
        per, outs = run_locks(fs, lock_params(v, **tuned["lock"]), log)
        if v == FULL:
            full_outs = outs
        rows += _rows(base, f"{fs.pipeline}+lock-{v}", "lock", per, lock_keys, n_boot, seed,
                      stable_hash({**common, "variant": v, "lock": tuned["lock"]}), tests)
    for mode, outs in (("gt", None), ("lock", full_outs)):
        per = run_gimbal(fs, gparams, outs)
        src = "the GT box" if mode == "gt" else f"lock-{FULL}"
        chash = stable_hash({**common, "gimbal": tuned["gimbal"], "mode": mode,
                             "lock": tuned["lock"] if mode == "lock" else None})
        rows += _rows(base, f"{fs.pipeline}+gimbal-{mode}", "gimbal", per, GIMBAL_KEYS,
                      n_boot, seed, chash, f"driven by {src}; {tests}")
    return rows


def _rows(base: dict, model: str, prefix: str, per: pd.DataFrame, keys, n_boot: int, seed: int,
          chash: str, notes: str) -> list[dict]:
    """One result row per metric: n_images = scored frames, n_scenes = sequences with tests."""
    extra = {"model": model, "n_images": int(per["n_frames"].sum()),
             "n_scenes": int(per["seq_id"].nunique()), "config_hash": chash, "notes": notes}
    return [{**base, **extra, "metric": f"{prefix}/{k}", "value": v, "ci_lo": lo, "ci_hi": hi}
            for k, (v, lo, hi) in summarize(per, keys, n_boot, seed).items()]


# --- public single-object benchmarks (UAVDT-S, UAV123 / UAV20L vehicles) ----------------------

SOT_LOCK_KEYS = LOCK_KEYS[:4]  # one annotated target per sequence: no wrong-target metrics
SOT_DETECT_HINT = r"run `scripts\follow.ps1 -Step sot-detect` (GPU) first"
SOT_LABEL = -1  # benchmark GT carries no class; the lock takes the class of the track at t0


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sot_det_path(cache_dir: Path, det_model: str, video: str) -> Path:
    """Detections of one benchmark video: `<cache>/<det model>/<source>/<folder>.parquet`, with a
    `.json` sidecar (written last) and the detector features in `<folder>.feats.npz`."""
    return Path(cache_dir) / det_model / f"{video}.parquet"


def _feats_path(det_path: Path) -> Path:
    return det_path.with_name(det_path.stem + ".feats.npz")


def _numbered(template: str, i: int) -> str:
    """`template` (a frame path ending in a zero-padded image number) renumbered to `i`."""
    p = Path(template)
    m = re.fullmatch(r"(.*?)(\d+)", p.stem)
    if m is None:
        raise ValueError(f"frame name {p.name} does not end in an image number")
    return str(p.with_name(f"{m.group(1)}{i:0{len(m.group(2))}d}{p.suffix}"))


def sot_videos(seqs) -> dict[str, tuple[int, int, list[str]]]:
    """(first, last, frame paths first..last) per video over every sequence cut from it, so each
    frame is detected once (UAV123 pieces and UAV20L share their videos). Frames between declared
    pieces (e.g. an excluded piece) are included; every path must exist."""
    span: dict[str, tuple[int, int, str]] = {}
    for s in seqs:
        lo, hi, tpl = span.get(s.video, (s.start, s.end, s.frame_paths[0]))
        span[s.video] = (min(lo, s.start), max(hi, s.end), tpl)
    out = {}
    for video, (lo, hi, tpl) in span.items():
        paths = [_numbered(tpl, i) for i in range(lo, hi + 1)]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise ValueError(f"{video}: {len(missing)} frames of {lo}..{hi} are missing, e.g. "
                             f"{missing[0]}")
        out[video] = (lo, hi, paths)
    return out


def _open_sot_detector(model: str, models_cfg: dict, imgsz: int, conf: float):
    """The registry detector (TensorRT engine), Ultralytics offline."""
    from cvt.eval.run import open_detector
    from cvt.train import job

    job.no_network()
    return open_detector(model, models_cfg, imgsz, conf, pd.DataFrame(), pd.DataFrame())[0]


def _open_extractor(weights: str, imgsz: int):
    from cvt.follow.features import _P3Extractor

    return _P3Extractor(weights, imgsz)  # calls no_network() before loading Ultralytics


def _det_fresh(meta: dict | None, path: Path, det_model: str, models_cfg: dict, imgsz: int,
               conf: float, lo: int, hi: int) -> bool:
    if not meta or not path.exists():
        return False
    if (meta.get("det_model"), meta.get("imgsz"), meta.get("conf")) != (det_model, imgsz, conf):
        return False
    if meta.get("first", 1 << 30) > lo or meta.get("last", -1) < hi:
        return False
    w = (models_cfg.get(det_model) or {}).get("weights")
    return not (w and Path(w).exists() and Path(w).stat().st_mtime > path.stat().st_mtime)


class _SotFrames:
    """The slice of `tp.SetData` that `features.extract_set` reads: name and frames."""

    def __init__(self, name: str, frames: pd.DataFrame):
        self.name, self.frames = name, frames


def detect_sot(
    cfg: dict, sot_cfg: dict, sets: list[str], *, models_cfg: dict, weights_model: str,
    imgsz: int, conf: float, min_score: float, cache_dir: Path, log: Log = _quiet,
) -> list[str]:
    """GPU step: per benchmark video, the detector's boxes on every frame (cached parquet, frame =
    image number), then the P3 features of its vehicle detections with score >= min_score
    (`cvt.follow.features`, cached npz). Resumable: up-to-date videos are skipped. Detection runs
    over all videos before the feature model is loaded, so one model is on the GPU at a time."""
    from cvt.follow import features as ft
    from cvt.follow import sot_data as sd

    sd.check_sets(sets)
    det_model = cfg["detector"]
    for m in (det_model, weights_model):
        if m not in models_cfg:
            raise KeyError(f"model {m!r} not in models config ({sorted(models_cfg)})")
    weights = models_cfg[weights_model].get("weights")
    if not weights or not Path(weights).exists():
        raise FileNotFoundError(f"{weights_model}: weights not found: {weights}")
    seqs = [s for set_name in sets for s in sd.sot_sequences(sot_cfg, set_name, cfg, log)]
    videos = sot_videos(seqs)
    log(f"{len(seqs)} sequences of {sets} in {len(videos)} videos")

    detector = None
    for k, (video, (lo, hi, paths)) in enumerate(sorted(videos.items()), 1):
        path = sot_det_path(cache_dir, det_model, video)
        if _det_fresh(_read_json(path.with_suffix(".json")), path, det_model, models_cfg, imgsz,
                      conf, lo, hi):
            continue
        if detector is None:
            detector = _open_sot_detector(det_model, models_cfg, imgsz, conf)
        nums = list(range(lo, hi + 1))
        rows = []
        for f, d in zip(nums, detector.predict_paths(paths), strict=True):
            for (x1, y1, x2, y2), s, lab in zip(d.boxes, d.scores, d.labels, strict=True):
                rows.append((f, float(x1), float(y1), float(x2), float(y2), float(s), int(lab)))
        dets = pd.DataFrame(rows, columns=tp.DET_COLS).astype(
            {"frame": "int64", "label": "int64", **{c: "float64" for c in (*BOX, "score")}})
        path.with_suffix(".json").unlink(missing_ok=True)  # sidecar last: none = incomplete
        _feats_path(path).unlink(missing_ok=True)
        write_parquet_atomic(dets, path)
        _write_json(path.with_suffix(".json"), {
            "det_model": det_model, "det_tag": detector.cache_tag, "imgsz": imgsz, "conf": conf,
            "video": video, "first": lo, "last": hi,
            "frames_hash": stable_hash([Path(p).name for p in paths])})
        log(f"[{video}] {len(dets)} detections on {len(nums)} frames ({k}/{len(videos)})")
    del detector

    extractor, sha = None, file_sha256(weights)[:12]
    for video, (_, _, declared) in sorted(videos.items()):
        path = sot_det_path(cache_dir, det_model, video)
        meta = _read_json(path.with_suffix(".json"))
        want = {"weights_model": weights_model, "weights_sha": sha, "det_model": det_model,
                "det_tag": meta["det_tag"], "imgsz": imgsz, "layer": ft.LAYER,
                "min_score": min_score, "first": meta["first"], "last": meta["last"]}
        fpath = _feats_path(path)
        if fpath.exists() and _read_meta(fpath) == want:
            continue
        nums = range(meta["first"], meta["last"] + 1)  # the cache may span more than declared
        paths = [_numbered(declared[0], i) for i in nums]
        if extractor is None:
            extractor = _open_extractor(weights, imgsz)
        dets = pd.read_parquet(path).assign(seq_id=video)
        s = _SotFrames(video, pd.DataFrame({"seq_id": video, "frame": list(nums), "path": paths}))
        row, feat = ft.extract_set(extractor, s, dets, min_score, log)
        ft._save(fpath, row, feat, want)
        log(f"[{video}] {len(row)} features -> {fpath}")
    return sorted(videos)


def load_sot_dets(cache_dir: Path, det_model: str, video: str) -> tuple[pd.DataFrame, dict]:
    path = sot_det_path(cache_dir, det_model, video)
    meta = _read_json(path.with_suffix(".json"))
    if meta is None or not path.exists() or meta.get("det_model") != det_model:
        raise ValueError(f"no detections of {det_model} on {video} at {path}; {SOT_DETECT_HINT}")
    return pd.read_parquet(path), meta


def load_sot_feats(cache_dir: Path, det_model: str, video: str,
                   det_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    fpath = _feats_path(sot_det_path(cache_dir, det_model, video))
    fmeta = _read_meta(fpath) if fpath.exists() else None
    if fmeta is None:
        raise ValueError(f"no detector features on {video} at {fpath}; {SOT_DETECT_HINT}")
    if (fmeta.get("det_tag"), fmeta.get("first"), fmeta.get("last")) != (
            det_meta["det_tag"], det_meta["first"], det_meta["last"]):
        raise ValueError(f"{fpath} was made for other detections; {SOT_DETECT_HINT}")
    return load_features(fpath)


def training_hashes(manifest_dir: Path, hash_cache: Path, log: Log = _quiet) -> pd.DataFrame:
    """image_id, set_name, phash of every training/tuning-side image of the manifest
    (`TRAINING_SIDE_ROLES`: VD-train, VD-tune, UD-pool, P-train, P-val, MOT-tune, ...): the
    reference of the benchmark leakage check. Test sets (T-UAVDT, T-MOT, ...) are never included:
    overlap with a test set is not leakage."""
    from cvt.data.leakage import TRAINING_SIDE_ROLES, load_or_compute_hashes
    from cvt.data.manifest import load_manifest

    images = load_manifest(manifest_dir)[0]
    tr = images[images.set_name.map(SET_ROLES).isin(TRAINING_SIDE_ROLES)]
    tr = tr.drop_duplicates("image_id").reset_index(drop=True)
    h = load_or_compute_hashes(tr, hash_cache, log=log)
    return pd.DataFrame({"image_id": tr.image_id.to_numpy(), "set_name": tr.set_name.to_numpy(),
                         "phash": h.to_numpy(np.uint64)})


def prepare_sot_set(
    cfg: dict, sot_cfg: dict, set_name: str, seqs, *, models_cfg: dict, cache_dir: Path,
    warp_cache: Path = Path("data/cache/gmc"), log: Log = _quiet,
) -> FollowSet:
    """One benchmark set ready for `run_locks` / `run_gimbal`: per sequence (local frames
    0..n-1), the cached detections and features of its video slice, BoT-SORT-3b tracks with GMC
    warps (computed on the frames and cached when missing), and one lock target (track_id 0) from
    its first to its last annotated frame."""
    from PIL import Image

    from cvt.data.schema import CLASS_ID
    from cvt.track.run import track_sequence

    det = cfg["detector"]
    tcfg = load_tracker_cfg(Path(cfg["tracker"]))
    classes = sorted(CLASS_ID[c] for c in sot_cfg["classes"])
    fps = set_fps(cfg, set_name)
    for s in seqs:  # every cache first: fail before any GMC or tracking
        _, meta = load_sot_dets(cache_dir, det, s.video)
        if meta["first"] > s.start or meta["last"] < s.end:
            raise ValueError(f"{s.video}: cached frames {meta['first']}..{meta['last']} do not "
                             f"cover {s.name} ({s.start}..{s.end}); {SOT_DETECT_HINT}")
        load_sot_feats(cache_dir, det, s.video, meta)
    fs = FollowSet(set_name, f"{det}+{tcfg['name']}", None, pd.DataFrame(), fps, "",
                   {"tracker_cfg_hash": stable_hash(tcfg)}, {}, "")
    targets, tags = [], set()
    _offline()  # tracking (and GMC) load Ultralytics
    for k, s in enumerate(seqs, 1):
        n = len(s.frame_paths)
        valid = np.flatnonzero(~np.isnan(s.gt).any(1))
        if not len(valid):
            log(f"[{set_name}] {s.name}: no annotated frame, no lock test")
            continue
        dets, meta = load_sot_dets(cache_dir, det, s.video)
        tags.add(meta["det_tag"])
        rows, feat = load_sot_feats(cache_dir, det, s.video, meta)
        fs.features_meta[s.video] = _read_meta(_feats_path(sot_det_path(cache_dir, det, s.video)))
        local = dets[(dets.frame >= s.start) & (dets.frame <= s.end)]
        local = local.assign(frame=local.frame - s.start)[tp.DET_COLS]
        fdets = dets.iloc[rows]
        keep = ((fdets.frame >= s.start) & (fdets.frame <= s.end)).to_numpy()
        fsub = fdets[keep].assign(frame=fdets.frame[keep] - s.start)
        frames = list(range(n))
        fr = pd.DataFrame({"frame": frames, "path": s.frame_paths})
        warps = tp._warps(f"{s.video.replace('/', '_')}_{s.start}", fr, tcfg, warp_cache, log)
        fs.tracks[s.name] = track_sequence(local, frames, tcfg, seq_id=s.name, fps=fps,
                                           warps=warps, classes=classes)
        fs.feats[s.name] = seq_feature_fn(fsub, np.asarray(feat)[keep])
        fs.frames[s.name], fs.warps[s.name] = frames, warps
        fs.paths[s.name] = dict(zip(frames, s.frame_paths, strict=True))
        with Image.open(s.frame_paths[0]) as im:
            fs.size[s.name] = im.size
        g = s.gt[valid]
        fs.gt[s.name] = pd.DataFrame({"seq_id": s.name, "frame": valid, "track_id": 0,
                                      "label": SOT_LABEL, **{c: g[:, i] for i, c in enumerate(BOX)}})
        targets.append({"seq_id": s.name, "track_id": 0, "label": SOT_LABEL,
                        "first": int(valid[0]), "last": int(valid[-1])})
        log(f"[{set_name}] {s.name}: {len(fs.tracks[s.name])} track boxes ({k}/{len(seqs)})")
    if len(tags) > 1:
        raise ValueError(f"{set_name}: videos were detected by different engines {sorted(tags)}; "
                         f"delete {Path(cache_dir) / det} and {SOT_DETECT_HINT}")
    fs.targets = pd.DataFrame(targets, columns=["seq_id", "track_id", "label", "first", "last"])
    fs.det_tag = next(iter(tags), "")
    return fs


def eval_sot_sets(
    cfg: dict, tuned: dict, sot_cfg: dict, sets: list[str], *, models_cfg: dict,
    cache_dir: Path, train_hashes: pd.DataFrame, sot_hash_cache: Path,
    warp_cache: Path = Path("data/cache/gmc"), reports_dir: Path = Path("reports"),
    n_boot: int = 1000, seed: int = 0, log: Log = _quiet,
) -> Iterator[list[dict]]:
    """Result rows per benchmark set (eval_set UAVDT-S, UAV123-veh, UAV20L-veh): the 5 lock
    variants (metrics lock/<SOT_LOCK_KEYS>) and the gimbal driven by the GT box and by the full
    lock, all at the frozen params. UAVDT-S first passes the pHash leakage check against the
    training-side images: matched sequences are dropped and the pairs written to
    `reports_dir/follow_sot_leakage.csv`."""
    from cvt.follow import sot_data as sd

    check_tuned(cfg, tuned)  # before reading any benchmark file
    sd.check_sets(sets)
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_commit()
    gparams = gimbal_params(cfg, tuned["gimbal"])
    for set_name in sets:
        seqs = sd.sot_sequences(sot_cfg, set_name, cfg, log)
        dropped: list[str] = []
        if set_name == "UAVDT-S":
            report, dropped = sd.sot_leakage(seqs, train_hashes, sot_hash_cache, log=log)
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            report.to_csv(Path(reports_dir) / "follow_sot_leakage.csv", index=False,
                          lineterminator="\n")
            log(f"[{set_name}] leakage: {len(report)} near-duplicate pairs; dropped {dropped}")
            seqs = [s for s in seqs if s.name not in dropped]
        fs = prepare_sot_set(cfg, sot_cfg, set_name, seqs, models_cfg=models_cfg,
                             cache_dir=cache_dir, warp_cache=warp_cache, log=log)
        if fs.targets.empty:
            log(f"[{set_name}] no lock tests: no rows")
            continue
        sec = sot_cfg[sd.SOT_SETS[set_name]]
        decl = stable_hash({"set": set_name, "sequences": sec["sequences"],
                            "classes": sot_cfg["classes"], "dropped": dropped})
        base = dict(timestamp=stamp, git_commit=commit, model_sha=fs.det_tag, eval_set=set_name,
                    slice="all", imgsz=tp.det_imgsz(models_cfg, cfg["detector"]),
                    manifest_hash=decl[:12],
                    set_hash=stable_hash([[s.name, s.video, s.start, len(s.frame_paths)]
                                          for s in seqs])[:8])
        common = {"pipeline": fs.pipeline, "tracker_cfg_hash": fs.tracks_meta["tracker_cfg_hash"],
                  "detections": fs.det_tag, "features": stable_hash(fs.features_meta),
                  "fps": fs.fps, "sot": decl, "n_boot": n_boot, "seed": seed}
        drop = f"; dropped by the leakage check: {', '.join(dropped)}" if dropped else ""
        tests = (f"{len(fs.targets)} lock tests (one annotated target per sequence, run end to end "
                 f"from its first annotated box at {fs.fps:g} fps{drop}); params tuned on "
                 f"{tuned['set']}")
        yield _set_rows(fs, tuned, gparams, base, common, tests, SOT_LOCK_KEYS, n_boot, seed, log)
