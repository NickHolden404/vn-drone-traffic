"""Per-clip traffic-analytics pipeline: detect (GPU) -> track (CPU) -> analyze (CPU), resumable via sidecars.

Frame k is the k-th decoded frame (0-based). Every data file is written before its sidecar, so a
clip whose sidecar exists and matches its inputs is complete; anything else is redone.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from cvt.analytics.anomaly import align, auc, pooled_auc
from cvt.analytics.events import (
    VEHICLE_IDS,
    event_table,
    flow_field,
    frame_scores,
    stopped_score,
    wrong_way_score,
)
from cvt.analytics.kinematics import KINEMATIC_COLS, ground_tracks, kinematics, noise_floor
from cvt.analytics.maps import grid_maps
from cvt.analytics.od import endpoints, find_arms, od_summary, od_table
from cvt.analytics.telemetry import clip_scale
from cvt.data.manifest import load_mot
from cvt.data.schema import CLASS_ID, CLASSES
from cvt.eval.run import load_checked_manifest, set_hash_of
from cvt.track.count import count_errors
from cvt.track.pipeline import (
    _ignored,
    _save_atomic,
    _uses_gmc,
    det_imgsz,
    load_scenes,
    load_set,
    load_tracks,
)
from cvt.track.run import DET_COLS, gmc_warps, load_tracker_cfg, track_sequence
from cvt.track.stabilize import reference_homographies
from cvt.utils import stable_hash, write_parquet_atomic

Log = Callable[[str], None]

VIDEO_EXTS = (".mp4", ".mov", ".mkv")
CHUNK = 16  # frames per detector call
LOG_EVERY = 500  # detect progress line every this many frames
JPEG_QUALITY = 90
MASK_SCORE = 0.3  # detections at least this confident are masked out of stabilization
SEQ_FPS = 30.0  # frame rate of the T-MOT / T-UAVDT sequences (as the follow tier assumes)
REQUIRED = (
    "detector", "models", "tracker", "imgsz", "conf", "set_name", "clips_dir", "labels_dir",
    "label_step", "label_alias", "held_out", "cameras", "car_length_m", "ref_height_tol_m",
    "stab_every", "smooth_s", "grid_m", "od", "events", "noise_floor", "od_sets", "scenes_dir",
    "readme_clips", "n_boot", "seed",
)  # fmt: skip
HINT = r"run `scripts\analytics.ps1 -Step detect` / `-Step track` first"


def load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return cfg


def clips(cfg: dict, only: list[str] | None = None) -> list[Path]:
    d = Path(cfg["clips_dir"])
    found = sorted((f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTS), key=lambda f: f.name)
    if only:
        unknown = [c for c in only if c not in {f.stem for f in found}]
        if unknown:
            raise ValueError(f"unknown clip(s) {unknown} in {d}")
        found = [f for f in found if f.stem in set(only)]
    return found


def paths(cache: Path, cfg: dict, clip: str) -> dict[str, Path]:
    cache = Path(cache)
    det = cache / str(cfg["detector"])
    trk = cache / f"{cfg['detector']}+{load_tracker_cfg(Path(cfg['tracker']))['name']}"
    return {
        "dets": det / f"{clip}.parquet",
        "dets_meta": det / f"{clip}.json",
        "tracks": trk / f"{clip}.parquet",
        "stab": trk / f"{clip}_stab.npz",
        "tracks_meta": trk / f"{clip}.json",
        "ref_jpg": trk / f"{clip}_ref.jpg",
        "kin": trk / f"{clip}_kin.parquet",
        "frames_dir": cache / "frames" / clip,
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, obj: dict) -> None:
    _save_atomic(path, lambda f: f.write(json.dumps(obj, sort_keys=True).encode("utf-8")))


def _frames(video: Path):
    """(fps, width, height, generator of BGR frames in decode order)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def gen():
        try:
            while True:
                ok, im = cap.read()
                if not ok:
                    return
                yield im
        finally:
            cap.release()

    return fps, w, h, gen()


def detect_clip(video: Path, cfg: dict, detector, cache: Path, log: Log) -> pd.DataFrame:
    video = Path(video)
    p = paths(cache, cfg, video.stem)
    want = {"clip": video.stem, "detector": cfg["detector"], "imgsz": cfg["imgsz"],
            "conf": cfg["conf"]}
    meta = _read_json(p["dets_meta"])
    if meta and p["dets"].exists() and all(meta.get(k) == v for k, v in want.items()):
        log(f"{video.stem}: detections cached")
        return pd.read_parquet(p["dets"])
    fps, w, h, frames = _frames(video)
    rows: list[np.ndarray] = []
    n = 0
    batch: list[np.ndarray] = []

    def flush() -> None:
        nonlocal n
        for d in detector.predict_arrays(batch):
            k = len(d.boxes)
            rows.append(np.column_stack([np.full(k, n), np.asarray(d.boxes).reshape(k, 4),
                                         np.asarray(d.scores), np.asarray(d.labels)]))
            n += 1
        batch.clear()

    for im in frames:
        batch.append(im)
        if len(batch) == CHUNK:
            before = n
            flush()
            if n // LOG_EVERY != before // LOG_EVERY:
                log(f"{video.stem}: {n} frames")
    if batch:
        flush()
    log(f"{video.stem}: {n} frames")
    arr = np.concatenate(rows) if rows else np.zeros((0, len(DET_COLS)))
    dets = pd.DataFrame(arr, columns=DET_COLS).astype(
        {"frame": "int64", "label": "int64", "x1": "float64", "y1": "float64", "x2": "float64",
         "y2": "float64", "score": "float64"})
    write_parquet_atomic(dets, p["dets"])
    _write_json(p["dets_meta"], {**want, "n_frames": n, "fps": fps, "width": w, "height": h})
    return dets


def interp_homographies(Hk: np.ndarray, validk: np.ndarray, key_frames: np.ndarray,
                        n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame H (each / H[2, 2]) by linear interpolation between key frames; a frame is valid
    when both neighbouring key frames are; the last key frame holds to the end."""
    Hk = np.asarray(Hk, float)
    Hk = Hk / Hk[:, 2:3, 2:3]
    validk = np.asarray(validk, bool)
    key = np.asarray(key_frames)
    H = np.empty((n_frames, 3, 3))
    valid = np.zeros(n_frames, bool)
    for f in range(n_frames):
        j = min(max(int(np.searchsorted(key, f, side="right")) - 1, 0), len(key) - 1)
        if f <= key[j] or j == len(key) - 1:
            H[f], valid[f] = Hk[j], validk[j]
        else:
            t = (f - key[j]) / (key[j + 1] - key[j])
            H[f] = (1 - t) * Hk[j] + t * Hk[j + 1]
            valid[f] = validk[j] and validk[j + 1]
    return H, valid


def _tracks_want(cfg: dict, tcfg: dict, meta: dict) -> dict:
    return {"dets": meta, "tracker_hash": stable_hash(tcfg), "stab_every": cfg["stab_every"]}


def _need(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}: {HINT}")


def track_clip(video: Path, cfg: dict, cache: Path, log: Log) -> None:
    video = Path(video)
    clip = video.stem
    p = paths(cache, cfg, clip)
    _need(p["dets_meta"])
    _need(p["dets"])
    meta = _read_json(p["dets_meta"])
    tcfg = load_tracker_cfg(Path(cfg["tracker"]))
    want = _tracks_want(cfg, tcfg, meta)
    done = _read_json(p["tracks_meta"])
    if (done and p["tracks"].exists() and p["stab"].exists()
            and all(done.get(k) == v for k, v in want.items())):
        log(f"{clip}: tracks cached")
        return
    dets = pd.read_parquet(p["dets"])
    n, fps = int(meta["n_frames"]), float(meta["fps"])
    fdir = p["frames_dir"]
    shutil.rmtree(fdir.parent, ignore_errors=True)  # also frames left by a killed run of any clip
    fdir.mkdir(parents=True)
    try:
        names = []
        for k, im in enumerate(_frames(video)[3]):
            name = str(fdir / f"{k:06d}.jpg")
            if not cv2.imwrite(name, im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise OSError(f"cannot write frame: {name}")
            names.append(name)
        if len(names) != n:
            raise ValueError(f"{clip}: decoded {len(names)} frames, detections have {n}")
        warps = None
        if _uses_gmc(tcfg):
            log(f"{clip}: GMC {tcfg['gmc_method']} over {n} frames")
            warps = gmc_warps(names, method=tcfg["gmc_method"])
        tracks = track_sequence(dets, list(range(n)), tcfg, seq_id=clip, fps=fps, warps=warps)
        key = np.unique(np.append(np.arange(0, n, int(cfg["stab_every"])), n - 1))
        ref_s = clip_scale(video, dets, int(meta["width"]), cfg)["ref_s"]
        ref = int(np.argmin(np.abs(key - ref_s * fps)))
        sure = dets[dets["score"] >= MASK_SCORE]
        by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(float)
                    for f, g in sure.groupby("frame")}
        log(f"{clip}: stabilizing {len(key)} key frames to frame {key[ref]}")
        Hk, validk = reference_homographies(
            [names[k] for k in key], boxes=[by_frame.get(int(k), np.zeros((0, 4))) for k in key],
            ref=ref)
        H, valid = interp_homographies(Hk, validk, key, n)
        write_parquet_atomic(tracks, p["tracks"])
        _save_atomic(p["ref_jpg"], lambda f: f.write(Path(names[int(key[ref])]).read_bytes()))
        _save_atomic(p["stab"], lambda f: np.savez(f, H=H, valid=valid, ref_frame=int(key[ref])))
        _write_json(p["tracks_meta"], {**want, "n_tracks": int(tracks["track_id"].nunique())})
    finally:
        shutil.rmtree(fdir, ignore_errors=True)


@dataclass
class ClipResult:
    clip: str
    meta: dict
    scale: dict
    kin: pd.DataFrame
    stopped: np.ndarray
    wrong: np.ndarray
    scores: dict[str, np.ndarray]
    events: pd.DataFrame
    endpoints: pd.DataFrame
    arms: np.ndarray
    od: pd.DataFrame
    od_summary: dict
    maps: dict[str, np.ndarray]
    noise: dict
    ref_frame: int


def analyze_clip(video: Path, cfg: dict, cache: Path) -> ClipResult:
    video = Path(video)
    clip = video.stem
    p = paths(cache, cfg, clip)
    for k in ("dets_meta", "dets", "tracks", "stab", "ref_jpg", "tracks_meta"):
        _need(p[k])
    meta = _read_json(p["dets_meta"])
    done = _read_json(p["tracks_meta"])
    want = _tracks_want(cfg, load_tracker_cfg(Path(cfg["tracker"])), meta)
    if not done or any(done.get(k) != v for k, v in want.items()):
        raise FileNotFoundError(f"{p['tracks_meta']} does not match the current detections or "
                                f"tracker config: {HINT}")
    dets = pd.read_parquet(p["dets"])
    tracks = pd.read_parquet(p["tracks"])
    with np.load(p["stab"], allow_pickle=False) as z:
        H, valid, ref_frame = z["H"], z["valid"], int(z["ref_frame"])
    n, fps = int(meta["n_frames"]), float(meta["fps"])
    size = (int(meta["width"]), int(meta["height"]))
    scale = clip_scale(video, dets, size[0], cfg)
    mpp = scale["mpp"]
    ev, od = cfg["events"], cfg["od"]

    g = ground_tracks(tracks, H, valid, list(range(n)), mpp)
    if np.isfinite(mpp):
        kin = kinematics(g, fps, cfg["smooth_s"])
    else:  # no scale: keep the rows and boxes, every metric column undefined
        kin = g.assign(t_s=g["frame"] / fps, **dict.fromkeys(KINEMATIC_COLS, np.nan))
    noise = noise_floor(kin, n, **cfg["noise_floor"])
    ep = endpoints(tracks, size, 0, n - 1, od["border_frac"], round(od["min_track_s"] * fps))
    pos = np.concatenate([ep["entry"].to_numpy(float), ep["exit"].to_numpy(float)])
    arms = find_arms(pos, od["bandwidth"], od["min_sep"], od["min_tracks"])
    table = od_table(ep, arms, od["max_dist"])
    summary = od_summary(ep, arms, od["max_dist"])

    stopped = stopped_score(kin, fps, ev["v_stop_kmh"], ev["t_stop_s"], ev["t_sat_s"],
                            gap_s=ev["gap_s"])
    field = flow_field(kin, cfg["grid_m"], ev["v_move_kmh"], ev["min_passes"],
                       ev["min_concentration"])
    wrong = wrong_way_score(kin, field, fps, ev["v_move_kmh"], ev["angle_deg"], ev["t_wrong_s"],
                            gap_s=ev["gap_s"])
    fs, fw = frame_scores(kin, stopped, n), frame_scores(kin, wrong, n)
    scores = {"stopped": fs, "wrong": fw, "both": np.maximum(fs, fw)}
    events = event_table(kin, stopped, wrong, fps, gap_s=ev["gap_s"], v_stop_kmh=ev["v_stop_kmh"])
    maps = {}
    if np.isfinite(mpp):
        maps = grid_maps(kin, mpp, size, fps, cfg["grid_m"], ev["v_stop_kmh"])
    return ClipResult(clip, meta, scale, kin, stopped, wrong, scores, events, ep, arms, table,
                      summary, maps, noise, ref_frame)


# --- evaluation rows and the clip summary --------------------------------------------------

NAN = float("nan")
ANOMALY_NOTES = ("frame-level ROC AUC, pooled over clips; flags stopped and wrong-way vehicles "
                 "only (labels cover 10 violation types); 16 of the dataset's 51 test videos: "
                 "not comparable with published AUC")
SCORE_KINDS = ("stopped", "wrong", "both")
SPEED_CLASSES = ("car", "motorbike", "truck", "bus")


def _group(clip: str, cfg: dict) -> str:
    return "held-out" if clip in cfg["held_out"] else "trained"


def _speed_stats(speed: np.ndarray, suffix: str = "") -> dict:
    return {f"speed_median_kmh{suffix}": float(np.median(speed)) if len(speed) else NAN,
            f"speed_p85_kmh{suffix}": float(np.percentile(speed, 85)) if len(speed) else NAN}


def clip_summary(r: ClipResult, labels: np.ndarray | None, cfg: dict) -> dict:
    """One row of the per-clip summary table (NaN where undefined). Speed statistics are over
    vehicle samples moving at least `events.v_move_kmh`, overall and per vehicle class; the
    `vehicles_<class>` columns count tracks."""
    sc, od = r.scale, r.od_summary
    speed = r.kin["speed_kmh"].to_numpy(float)
    label = r.kin["label"].to_numpy()
    with np.errstate(invalid="ignore"):
        moving = (np.isin(label, VEHICLE_IDS) & np.isfinite(speed)
                  & (speed >= cfg["events"]["v_move_kmh"]))
    per_class = r.endpoints.groupby("label").size() if len(r.endpoints) else pd.Series(dtype=int)
    row = {"clip": r.clip, "group": _group(r.clip, cfg),
           "duration_s": r.meta["n_frames"] / r.meta["fps"], "camera": sc.get("camera"),
           "height_m": sc.get("height_med", NAN), "mpp_telemetry": sc.get("mpp_telemetry", NAN),
           "mpp_car": sc.get("mpp_car", NAN), "scale_source": sc.get("source")}
    row |= {f"vehicles_{c}": int(per_class.get(i, 0)) for i, c in enumerate(CLASSES)}
    kinds = r.events["kind"] if len(r.events) else pd.Series(dtype=object)
    row |= {"tracks": od["tracks"], "complete_frac": od["complete_frac"], "arms": od["arms"],
            **_speed_stats(speed[moving]),
            "noise_median_kmh": r.noise["median_kmh"], "noise_p95_kmh": r.noise["p95_kmh"],
            "stopped_events": int((kinds == "stopped").sum()),
            "wrong_way_events": int((kinds == "wrong_way").sum()),
            "abnormal_frac": NAN, "auc_both": NAN}
    for c in SPEED_CLASSES:
        row |= _speed_stats(speed[moving & (label == CLASS_ID[c])], f"_{c}")
    if labels is not None:
        s, lab = align(r.scores["both"], labels, cfg["label_step"], r.clip)
        row["abnormal_frac"] = float(np.mean(labels))
        row["auc_both"] = auc(s, lab)
    return row


def anomaly_rows(results: list[ClipResult], cfg: dict, labels: dict[str, np.ndarray],
                 base: dict) -> list[dict]:
    """`anomaly/auc` rows per score kind and slice (all / trained / held-out), pooled over the
    clips that have labels; clips without labels are left out."""
    tname = load_tracker_cfg(Path(cfg["tracker"]))["name"]
    scored = [r for r in results if r.clip in labels]
    rows = []
    for kind in SCORE_KINDS:
        per = {r.clip: align(r.scores[kind], labels[r.clip], cfg["label_step"], r.clip)
               for r in scored}
        for sl in ("all", "trained", "held-out"):
            names = [c for c in per if sl == "all" or _group(c, cfg) == sl]
            if not names:
                continue
            value, lo, hi = pooled_auc([per[c] for c in names], int(cfg["n_boot"]),
                                       int(cfg["seed"]))
            rows.append({**base, "eval_set": cfg["set_name"], "slice": sl,
                         "model": f"{cfg['detector']}+{tname}+events-{kind}",
                         "metric": "anomaly/auc", "value": value, "ci_lo": lo, "ci_hi": hi,
                         "n_images": int(sum(len(per[c][1]) for c in names)),
                         "n_scenes": len(names), "notes": ANOMALY_NOTES})
    return rows


def od_eval(gt: pd.DataFrame, pred: pd.DataFrame, sizes: pd.DataFrame, cfg: dict) -> dict:
    """Turning-movement counts of one sequence from GT and predicted tracks, both assigned to the
    arms found in the GT endpoints; tracks shorter than `od.min_track_s` are not counted."""
    od = cfg["od"]
    first, last = int(sizes["frame"].min()), int(sizes["frame"].max())
    min_frames = round(od["min_track_s"] * SEQ_FPS)
    ep_gt = endpoints(gt, sizes, first, last, od["border_frac"], min_frames)
    ep_pred = endpoints(pred, sizes, first, last, od["border_frac"], min_frames)
    pos = np.concatenate([ep_gt["entry"].to_numpy(float), ep_gt["exit"].to_numpy(float)])
    arms = find_arms(pos, od["bandwidth"], od["min_sep"], od["min_tracks"])
    cells = pd.concat([od_table(ep_gt, arms, od["max_dist"])["n"].rename("gt"),
                       od_table(ep_pred, arms, od["max_dist"])["n"].rename("pred")], axis=1)
    return {"cells": cells.fillna(0).astype("int64"), "arms": len(arms),
            "complete_gt": int(ep_gt["complete"].sum()),
            "complete_pred": int(ep_pred["complete"].sum())}


def _stacked_errors(cells: list[pd.DataFrame]) -> dict:
    allc = pd.concat(cells, keys=range(len(cells)))
    return count_errors(allc[["gt"]].rename(columns={"gt": "n"}),
                        allc[["pred"]].rename(columns={"pred": "n"}))


def _boot_od(cells: list[pd.DataFrame], n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% intervals of the stacked count errors, resampling sequences with replacement."""
    out = {m: (NAN, NAN) for m in ("count_pct_err", "count_MAE")}
    if len(cells) < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = [_stacked_errors([cells[i] for i in rng.integers(0, len(cells), len(cells))])
             for _ in range(n_boot)]
    for m in out:
        v = [b[m] for b in boots if not np.isnan(b[m])]
        if v:
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[m] = (float(lo), float(hi))
    return out


def od_rows(cfg: dict, sets: list[str], *, models_cfg: dict, manifest_dir: Path,
            hash_cache: Path, out_root: Path, base: dict, log: Log) -> list[dict]:
    """Origin-destination count rows per set over the sequences that have a scene file (every
    sequence when the set has none), from the tracks written by `cvt track run`."""
    tcfg = load_tracker_cfg(Path(cfg["tracker"]))
    name = f"{cfg['detector']}+{tcfg['name']}"
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    mot_gt, _ = load_mot(manifest_dir)
    scenes = load_scenes(cfg["scenes_dir"], sets)
    rows = []
    for set_name in sets:
        s = load_set(set_name, images, boxes, mot_gt)
        tracks, meta = load_tracks(out_root, name, set_name, tcfg, mhash)
        tracks = tracks[~_ignored(tracks, s.ignore)]
        have = {sc["seq_id"] for sc in scenes if sc["set_name"] == set_name}
        missing = sorted(have - set(s.sequences))
        if missing:
            raise ValueError(f"scene {', '.join(missing)}: no such sequence in {set_name}")
        seqs = [q for q in s.sequences if q in have] if have else s.sequences
        log(f"[{set_name}] OD counts over {len(seqs)} sequences")
        res = []
        for q in seqs:
            sizes = s.frames[s.frames.seq_id == q][["frame", "width", "height"]]
            res.append(od_eval(s.gt[s.gt.seq_id == q], tracks[tracks.seq_id == q], sizes, cfg))
        cells = [r["cells"] for r in res]
        err = _stacked_errors(cells) if cells else {"count_pct_err": NAN, "count_MAE": NAN}
        ci = _boot_od(cells, int(cfg["n_boot"]), int(cfg["seed"]))
        cgt, cpred = sum(r["complete_gt"] for r in res), sum(r["complete_pred"] for r in res)
        notes = f"{len(seqs)} sequences, {sum(r['arms'] for r in res)} arms, {cgt} complete GT tracks"
        n_images = int(s.frames.seq_id.isin(seqs).sum())
        common = {**base, "eval_set": set_name, "slice": "all", "model": name,
                  "model_sha": meta["detections_cache_tag"],
                  "imgsz": det_imgsz(models_cfg, cfg["detector"]), "manifest_hash": mhash,
                  "set_hash": set_hash_of(s.im), "n_images": n_images, "n_scenes": len(seqs),
                  "notes": notes,
                  "config_hash": stable_hash({"od": cfg["od"], "tracker": stable_hash(tcfg),
                                              "sequences": seqs, "n_boot": cfg["n_boot"],
                                              "seed": cfg["seed"]})}
        for m in ("count_pct_err", "count_MAE"):
            rows.append({**common, "metric": f"od/{m}", "value": err[m], "ci_lo": ci[m][0],
                         "ci_hi": ci[m][1]})
        rows.append({**common, "metric": "od/complete_ratio",
                     "value": cpred / cgt if cgt else NAN, "ci_lo": NAN, "ci_hi": NAN})
    return rows


def result_base(results: list[ClipResult], cfg: dict, stamp: str, commit: str) -> dict:
    """The RESULT_COLS shared by the A-LONG rows (no manifest: manifest_hash is empty)."""
    return {"timestamp": stamp, "git_commit": commit,
            "model_sha": stable_hash([r.meta for r in sorted(results, key=lambda r: r.clip)]),
            "imgsz": cfg["imgsz"], "manifest_hash": "",
            "set_hash": stable_hash(sorted(r.clip for r in results)),
            "config_hash": stable_hash(cfg)}
