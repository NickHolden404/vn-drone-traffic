"""Markdown tables from results.csv, injected into README between markers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import pandas as pd

START, END = "<!-- results:start -->", "<!-- results:end -->"
KEY = ["model", "imgsz", "eval_set", "slice", "metric"]


def _fmt(v: float, lo: float, hi: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{100 * v:.1f}"
    if lo is not None and not (isinstance(lo, float) and math.isnan(lo)):
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def _class_footnotes(d: pd.DataFrame) -> list[str]:
    """One line per eval set naming the classes averaged into its mAP (from `notes`)."""
    lines = []
    for s, g in d.groupby("eval_set", sort=True):
        notes = g.notes.fillna("mAP classes: unknown").astype(str)
        classes = notes.str.removeprefix("mAP classes: ")
        distinct = list(dict.fromkeys(classes))
        if len(distinct) == 1:
            lines.append(f"- {s}: mAP over {distinct[0]}")
        else:
            parts = [
                f"mAP over {c} ({', '.join(sorted(set(g.model[classes == c].astype(str))))})"
                for c in distinct
            ]
            lines.append(f"- {s}: " + "; ".join(parts))
    return lines


DET_SETS = ("T-VD", "T-VN", "VD-tune")  # the README's main detection table
TRACK_DET_SETS = ("MOT-tune", "T-MOT", "T-UAVDT")  # detection mAP shown in the tracking block


def results_table(
    df: pd.DataFrame, metrics: Sequence[str] = ("mAP50", "mAP50_95"),
    sets: Sequence[str] | None = None,
) -> str:
    """Detection metrics per (model, imgsz); `sets` limits the eval sets (default: all)."""
    df = df.assign(slice=df["slice"].fillna("all") if "slice" in df else "all")
    df = df[(df["slice"] == "all") & df.metric.isin(metrics)]
    if sets is not None:
        df = df[df.eval_set.isin(list(sets))]
    d = df.sort_values("timestamp", kind="stable").groupby(KEY).tail(1)
    cols = [(s, m) for s in sorted(d.eval_set.unique()) for m in metrics]
    header = "| model | imgsz | " + " | ".join(f"{s} {m}" for s, m in cols) + " |"
    lines = [header, "|" + "---|" * (2 + len(cols))]
    for (model, imgsz), g in d.groupby(["model", "imgsz"], sort=True):
        cells = []
        for s, m in cols:
            hit = g[(g.eval_set == s) & (g.metric == m)]
            cells.append("–" if hit.empty else _fmt(*hit[["value", "ci_lo", "ci_hi"]].iloc[0]))
        lines.append(f"| {model} | {imgsz} | " + " | ".join(cells) + " |")
    note = (
        "mAP in %, 95% CI in brackets from a cluster bootstrap over whole clips/sequences. "
        "With few clips (see n_scenes in reports/results.csv) the CIs are approximate."
    )
    return "\n".join(lines) + "\n\n" + "\n".join(_class_footnotes(d)) + "\n\n" + note


def inject_readme(text: str, table: str, name: str = "results") -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise ValueError(f"README is missing the {name} markers")
    return pattern.sub(lambda _: f"{start}\n{table}\n{end}", text)


COMPARE_METRICS = ("delta/mAP50_95", "delta/AP50_95/motorbike", "delta/AP50_95/pedestrian",
                   "delta/AP50_95/bicycle")


def _fmt_delta(v: float, lo: float, hi: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{100 * v:+.1f}"
    if lo is not None and not (isinstance(lo, float) and math.isnan(lo)):
        s += f" [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return s


def compare_table(df: pd.DataFrame, metrics: Sequence[str] = COMPARE_METRICS) -> str:
    """Paired deltas (a − b, points) with 95% paired cluster-bootstrap CIs."""
    d = df[df.metric.isin(metrics)].copy()
    if d.empty:
        return "(no paired comparisons yet)"
    d["slice"] = d["slice"].fillna("all") if "slice" in d else "all"
    d = d[d["slice"] == "all"]
    d["vs"] = d.notes.fillna("vs ?").str.split(";").str[0]
    d = d.sort_values("timestamp", kind="stable").groupby(
        ["model", "imgsz", "vs", "eval_set", "metric"]).tail(1)
    cols = [(s, m) for s in sorted(d.eval_set.unique()) for m in metrics if m in set(d.metric)]
    head = "| a | b | " + " | ".join(f"{s} {m.removeprefix('delta/')}" for s, m in cols) + " |"
    lines = [head, "|" + "---|" * (2 + len(cols))]
    for (model, imgsz, vs), g in d.groupby(["model", "imgsz", "vs"], sort=True):
        cells = []
        for s, m in cols:
            hit = g[(g.eval_set == s) & (g.metric == m)]
            cells.append("–" if hit.empty else _fmt_delta(*hit[["value", "ci_lo", "ci_hi"]].iloc[0]))
        lines.append(f"| {model}@{imgsz} | {vs} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\nΔ in points (a − b); 95% paired cluster-bootstrap CI over clips."


TRACK_SETS = ("T-MOT", "T-UAVDT")
TRACK_HEAD = ("HOTA", "MOTA", "IDF1")
TRACK_SLICES = ("daylight", "night", "fog", "low_alt", "medium_alt", "high_alt")


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_count(v: float, lo: float, hi: float) -> str:
    return "–" if _missing(v) else f"{v:.1f}"


def _fmt_pct(v: float, lo: float, hi: float) -> str:
    return "–" if _missing(v) else f"{100 * v:.1f}%"


def _cells(g: pd.DataFrame, cols, fmt_of) -> list[str]:
    out = []
    for _, s, sl, m in cols:
        hit = g[(g.eval_set == s) & (g["slice"] == sl) & (g.metric == m)]
        out.append("–" if hit.empty else fmt_of(m)(*hit[["value", "ci_lo", "ci_hi"]].iloc[0]))
    return out


def _distinct(values) -> str:
    """Distinct integer values joined by '/' ('' without any), e.g. '14' or '2/3'."""
    n = sorted({int(v) for v in pd.to_numeric(pd.Series(values), errors="coerce").dropna()})
    return "/".join(map(str, n))


def _n_seq(d: pd.DataFrame, eval_set: str, slice_: str) -> str:
    """' (n seq)' from the slice's HOTA rows' n_scenes ('' without any; '2/3' if they differ)."""
    if "n_scenes" not in d:
        return ""
    hit = d[(d.eval_set == eval_set) & (d["slice"] == slice_) & (d.metric == "HOTA")]
    n = _distinct(hit.n_scenes)
    return f" ({n} seq)" if n else ""


def tracking_table(df: pd.DataFrame) -> str:
    """Tracking results (models named `<detector>+<tracker>`): HOTA/MOTA/IDF1 per test set,
    T-UAVDT HOTA per slice, counting error, paired tracker comparisons, then detection mAP on
    the tracking sets (kept out of the main detection table)."""
    det = df[df.metric.isin(["mAP50", "mAP50_95"]) & df.eval_set.isin(TRACK_DET_SETS)]
    det_part = ("Detection mAP on tracking sets\n\n" + results_table(det, sets=TRACK_DET_SETS)
                if len(det) else "")
    track = _tracking_part(df)
    if not track and not det_part:
        return "(no tracking results yet)"
    return "\n\n".join(p for p in (track, det_part) if p)


def _tracking_part(df: pd.DataFrame) -> str:
    d = df.assign(slice=df["slice"].fillna("all") if "slice" in df else "all")
    d = d[d.model.astype(str).str.contains("+", regex=False)]
    # `<detector>+<tracker>` models only: tiled detectors (`...+tiles640-fp16`) also contain "+"
    trackers = set(d.model[d.metric.isin(TRACK_HEAD) | d.metric.str.startswith("delta/")])
    d = d[d.model.isin(trackers)]
    if d.empty:
        return ""
    # a model compared with several b's keeps one delta row per b (the `vs` part of notes)
    d = d.assign(vs=d.notes.fillna("").astype(str).str.split(";").str[0])
    d = d.sort_values("timestamp", kind="stable").groupby([*KEY, "vs"]).tail(1)
    cols = [(f"{s} {m}", s, "all", m) for s in TRACK_SETS for m in TRACK_HEAD]
    cols += [(f"T-UAVDT HOTA {sl}{_n_seq(d, 'T-UAVDT', sl)}", "T-UAVDT", sl, "HOTA")
             for sl in TRACK_SLICES]
    cols += [c for s in TRACK_SETS for c in ((f"{s} count MAE", s, "all", "count_MAE"),
                                             (f"{s} count % err", s, "all", "count_pct_err"))]
    rule2 = bool((d.metric == "count2_pct_err").any())  # counting rule 2; without rows, no columns
    if rule2:
        cols += [(f"{s} count2 % err", s, "all", "count2_pct_err") for s in TRACK_SETS]
    fmts = {"count_MAE": _fmt_count, "count_pct_err": _fmt_pct, "count2_pct_err": _fmt_pct}
    main = d[~d.metric.str.startswith("delta/")]
    parts = []
    if not main.empty:
        lines = ["| model+tracker | " + " | ".join(c[0] for c in cols) + " |",
                 "|" + "---|" * (1 + len(cols))]
        for model, g in main.groupby("model", sort=True):
            lines.append(f"| {model} | " + " | ".join(_cells(g, cols, lambda m: fmts.get(m, _fmt)))
                         + " |")
        parts.append("\n".join(lines))
    delta = d[d.metric.isin([f"delta/{m}" for m in TRACK_HEAD]) & (d["slice"] == "all")]
    if not delta.empty:
        delta = delta.assign(vs=delta.vs.replace("", "vs ?"))
        dcols = [(f"{s} Δ{m}", s, "all", f"delta/{m}") for s in TRACK_SETS for m in TRACK_HEAD]
        lines = ["| a | b | " + " | ".join(c[0] for c in dcols) + " |",
                 "|" + "---|" * (2 + len(dcols))]
        for (model, vs), g in delta.groupby(["model", "vs"], sort=True):
            lines.append(f"| {model} | {vs} | " + " | ".join(_cells(g, dcols, lambda m: _fmt_delta))
                         + " |")
        parts.append("\n".join(lines))
    note = (
        "HOTA / MOTA / IDF1 in %, class-combined (statistics summed over the evaluated classes: "
        "T-MOT all six, T-UAVDT car/truck/bus); 95% CI in brackets from a cluster bootstrap over "
        "whole sequences (none for slices with fewer than 2 sequences; CIs from 3 or fewer "
        "sequences are not meaningful). Counting: MAE over "
        "(scene, line, class, direction) cells and total % error Σ|pred − GT| / Σ GT on the "
        "pre-declared lines in configs/scenes. Δ in points (a − b), paired bootstrap over sequences."
    )
    if rule2:
        note += (
            " count2: counting rule 2, applied alike to GT and tracks: points whose box lies "
            "within max(2 px, 1% of min(W, H)) of the image border are dropped, and only a "
            "track's first crossing of each line counts."
        )
    return "\n\n".join(parts) + "\n\n" + note


FOLLOW_PREFIXES = ("lock/", "gimbal/")
FOLLOW_SETS = ("T-MOT", "T-UAVDT", "UAVDT-S", "UAV123-veh", "UAV20L-veh")  # column order
SOT_SETS = ("UAVDT-S", "UAV123-veh", "UAV20L-veh")  # public single-object benchmarks
LOCK_VARIANTS = ("id_only", "mil", "reacq", "features", "hsv")  # ablation order
LOCK_COLS = (("lock/success_auc", "success AUC"), ("lock/precision_20", "prec@20px"),
             ("lock/norm_precision_auc", "norm prec AUC"), ("lock/lock_kept", "lock kept"),
             ("lock/wrong_frame_rate", "wrong-target frames"),
             ("lock/wrong_switches_per_target", "wrong switches/target"))
GIMBAL_COLS = (("gimbal/centre_err", "centre err"), ("gimbal/inside", "inside"),
               ("gimbal/smooth", "smoothness"), ("gimbal/clamped_frac", "clamped"))


def is_follow(df: pd.DataFrame) -> pd.Series:
    """Lock / gimbal rows (README block `follow` only)."""
    return df.metric.astype(str).str.startswith(FOLLOW_PREFIXES)


def _fmt_num(digits: int):
    def fmt(v: float, lo: float, hi: float) -> str:
        if _missing(v):
            return "–"
        s = f"{v:.{digits}f}"
        if not _missing(lo):
            s += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
        return s
    return fmt


def _variant_key(model: str) -> tuple:
    head, _, v = model.rpartition("+")
    v = v.removeprefix("lock-").removeprefix("gimbal-")
    order = (*LOCK_VARIANTS, "gt", "lock")
    return (head, order.index(v) if v in order else len(order), v)


def _n_follow(d: pd.DataFrame, eval_set: str) -> str:
    """' (n targets, m seq)' for one set: n from the rows' notes ('N lock tests ...', written by
    `cvt follow eval`), m from n_scenes; '' without either, '2/3' where rows differ."""
    g = d[d.eval_set == eval_set]
    nt = (_distinct(g.notes.astype(str).str.extract(r"(\d+) lock tests", expand=False))
          if "notes" in g else "")
    ns = _distinct(g.n_scenes) if "n_scenes" in g else ""
    parts = [p for p in (f"{nt} targets" if nt else "", f"{ns} seq" if ns else "") if p]
    return f" ({', '.join(parts)})" if parts else ""


def _variant_name(model: str) -> str:
    """'…+lock-features' -> 'features'; '…+gimbal-gt' -> 'GT box'; '…+gimbal-lock' -> 'lock'."""
    v = model.rpartition("+")[2]
    return {"gimbal-gt": "GT box", "gimbal-lock": "lock"}.get(v, v.removeprefix("lock-"))


def _follow_part(d: pd.DataFrame, cols, label: str, sets: list[str], fmt_of) -> str:
    """Lock or gimbal table: one row per set and variant, metrics as columns. A set's first row
    names its lock tests and sequences; variants drop the shared pipeline prefix when there is
    only one pipeline (named above the table)."""
    heads = {m.rpartition("+")[0] for m in d.model.unique()}
    short = len(heads) == 1
    lines = [f"| set | {label} | " + " | ".join(name for _, name in cols) + " |",
             "|" + "---|" * (2 + len(cols))]
    for s in sets:
        g_set = d[d.eval_set == s]
        for i, model in enumerate(sorted(g_set.model.unique(), key=_variant_key)):
            g = g_set[g_set.model == model]
            name = _variant_name(model) if short else model
            first = f"{s}{_n_follow(d, s)}" if i == 0 else ""
            cells = _cells(g, [(None, s, "all", m) for m, _ in cols], fmt_of)
            lines.append(f"| {first} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def follow_table(df: pd.DataFrame) -> str:
    """Target lock on our own GT lock tests: one table of lock variants x sets (OPE metrics) and one
    of the virtual gimbal driven by the GT box and by the lock."""
    d = df.assign(slice=df["slice"].fillna("all") if "slice" in df else "all")
    d = d[is_follow(d) & (d["slice"] == "all")]
    if d.empty:
        return "(no follow results yet)"
    d = d.sort_values("timestamp", kind="stable").groupby(KEY).tail(1)
    present = set(d.eval_set)
    sets = [s for s in FOLLOW_SETS if s in present] + sorted(present - set(FOLLOW_SETS))
    lock = d[d.metric.str.startswith("lock/")]
    gim = d[d.metric.str.startswith("gimbal/")]
    count = {"lock/wrong_switches_per_target": _fmt_num(2), "gimbal/smooth": _fmt_num(3)}
    parts = []
    if not lock.empty:
        heads = sorted({m.rpartition("+")[0] for m in d.model.unique()})
        if len(heads) == 1:
            parts.append(f"Pipeline `{heads[0]}`; lock (OPE metrics)")
        parts.append(_follow_part(lock, LOCK_COLS, "lock variant", sets,
                                  lambda m: count.get(m, _fmt)))
    if not gim.empty:
        parts.append("Virtual gimbal (crop driven by the GT box, or end to end by lock-features)"
                     "\n\n" + _follow_part(gim, GIMBAL_COLS, "gimbal input", sets,
                                           lambda m: count.get(m, _fmt)))
    note = (
        "Lock tests: every GT vehicle track (car, truck, bus, motorbike) lasting at least 5 s, "
        "initialised from its GT box in its first frame and run to its last; only frames with "
        "the target's GT box are scored. Success AUC (IoU thresholds 0..1), precision at 20 px "
        "centre error, normalized precision AUC, lock kept (IoU > 0.5) and wrong-target frames "
        "(IoU >= 0.5 with another GT track) in %, frame-weighted; wrong switches = runs of "
        "wrong-target frames per target. Variants: tracker ID only; + MIL fallback; + motion and "
        "geometry re-acquisition; + detector-feature appearance (features) or an HSV colour "
        "histogram (hsv). Gimbal: centring error (target centre offset, % of the crop), target "
        "fully inside the crop (%), smoothness (RMS crop acceleration, frame fractions / s²) and "
        "frames clamped at the frame edge (%). Lock and gimbal parameters were tuned on MOT-tune "
        "only (configs/follow.tuned.yaml) and frozen; 95% CI in brackets from a cluster "
        "bootstrap over sequences."
    )
    if present & set(SOT_SETS):
        note += (
            " Public benchmarks (UAVDT-S; UAV123-veh and UAV20L-veh, the real car and truck "
            "sequences; lists and exclusions in configs/follow_sot.yaml): one annotated target "
            "per sequence, run end to end (m@1536 detection, BoT-SORT-3b, lock) from its first "
            "annotated box to its last, same OPE metrics; with no other GT tracks the "
            "wrong-target metrics are not defined (–). A detector-based lock is a different kind "
            "of tracker from the generic single-object trackers of these leaderboards, and the "
            "detector was trained on other data: not leaderboard-comparable."
        )
    return "\n\n".join(parts) + "\n\n" + note


ANALYTICS_PREFIXES = ("od/", "anomaly/")
ANALYTICS_SLICES = ("all", "trained", "held-out")
ANALYTICS_EVENTS = (("stopped", "stopped"), ("wrong", "wrong-way"), ("both", "both"))
OD_COLS = (("od/count_pct_err", "total count error %", _fmt),
           ("od/count_MAE", "count MAE", _fmt_num(2)),
           ("od/complete_ratio", "complete tracks, ours / GT %", _fmt))
VEHICLE_COLS = ("vehicles_car", "vehicles_truck", "vehicles_bus", "vehicles_motorbike")
CLIP_HEAD = ("clip", "group", "height (m)", "cm/px telemetry", "cm/px car length", "tracks",
             "complete tracks %", "median km/h", "85th pct km/h", "noise floor km/h (median)",
             "events stopped / wrong-way", "abnormal %", "AUC both %")
ANALYTICS_NOTE = (
    "Speeds are unvalidated: no measured speeds exist for these clips, and the scale uncertainty "
    "is about 10-15 % (telemetry height against the car-length cross-check). Speed statistics "
    "are over vehicles (car, motorbike, truck, bus) moving at 5 km/h or faster, with per-class "
    "values in reports/analytics_clips.csv; the noise floor is the apparent speed of parked "
    "vehicles. \"tracks\" counts distinct vehicle tracks, not vehicles (a vehicle whose track "
    "breaks counts twice); the csv's vehicles_<class> columns count tracks too. The labels cover "
    "ten violation types, while we flag stopped and wrong-way vehicles only, so the AUC rewards "
    "only part of what the labels mark as abnormal; a frame scores its most stopped vehicle, so "
    "a long-parked vehicle keeps the stopped score high for the whole clip. The detector was "
    "trained on pseudo-labels from 11 of the 16 clips (the 5 location-A clips are held out). The "
    "AUC covers 16 of the dataset's 51 test videos and is not comparable with published AUC. The "
    "origin-destination check runs on short test sequences with few complete tracks and ignores "
    "tracks shorter than 1 s; it is weak: on MOT-tune, whose moving-camera sequences make "
    "entering and leaving the frame no junction arm, the counts were off by more than 100 %. "
    "\"Complete tracks, ours / GT\" is the number of our tracks that cross the frame border to "
    "border over the GT number and can exceed 100 % (broken or spurious tracks). Maps are JPG "
    "to keep the repository small. 95% CI in brackets from a cluster bootstrap."
)


def is_analytics(df: pd.DataFrame) -> pd.Series:
    """Anomaly / origin-destination rows (README block `analytics` only)."""
    return df.metric.astype(str).str.startswith(ANALYTICS_PREFIXES)


def _latest(d: pd.DataFrame) -> pd.DataFrame:
    return d.sort_values("timestamp", kind="stable").groupby(KEY).tail(1)


def _anomaly_part(d: pd.DataFrame) -> str:
    d = d[d.metric == "anomaly/auc"]
    if d.empty:
        return ""
    ev = d.model.astype(str).str.rpartition("+events-")[2]
    n = {sl: _distinct(d[d["slice"] == sl].n_scenes) for sl in ANALYTICS_SLICES}
    head = [sl + (f" ({n[sl]} clips)" if n[sl] else "") for sl in ANALYTICS_SLICES]
    lines = ["Event AUC (frame-level, A-LONG)\n",
             "| events flagged | " + " | ".join(head) + " |", "|" + "---|" * 4]
    for key, name in ANALYTICS_EVENTS:
        g = d[ev == key]
        cells = []
        for sl in ANALYTICS_SLICES:
            hit = g[g["slice"] == sl]
            cells.append("–" if hit.empty else _fmt(*hit[["value", "ci_lo", "ci_hi"]].iloc[0]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _od_part(d: pd.DataFrame) -> str:
    d = d[d.metric.str.startswith("od/")]
    if d.empty:
        return ""
    lines = ["Origin-destination counts vs ground truth\n",
             "| set | " + " | ".join(c[1] for c in OD_COLS) + " |", "|" + "---|" * 4]
    for s in sorted(d.eval_set.unique()):
        g = d[d.eval_set == s]
        notes = g.notes.dropna().astype(str)
        notes = notes[notes != ""]
        label = f"{s} ({notes.iloc[0]})" if len(notes) else s
        cells = []
        for m, _, fmt in OD_COLS:
            hit = g[g.metric == m]
            cells.append("–" if hit.empty else fmt(*hit[["value", "ci_lo", "ci_hi"]].iloc[0]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _clip_cell(row, col: str, fmt) -> str:
    v = row.get(col)
    return "–" if _missing(v) or pd.isna(v) else fmt(v)


def _clips_part(clips: pd.DataFrame) -> str:
    one = "{:.1f}".format

    def pct(v):  # fraction -> percent; metres per pixel -> cm per pixel
        return f"{100 * v:.1f}"

    lines = ["Per clip\n", "| " + " | ".join(CLIP_HEAD) + " |", "|" + "---|" * len(CLIP_HEAD)]
    for _, r in clips.iterrows():
        have = [c for c in VEHICLE_COLS if c in r.index and not pd.isna(r[c])]
        veh = "–" if not have else str(int(sum(r[c] for c in have)))
        ev = [_clip_cell(r, c, lambda v: str(int(v))) for c in ("stopped_events", "wrong_way_events")]
        cells = [str(r.get("clip", "–")), _clip_cell(r, "group", str),
                 _clip_cell(r, "height_m", one), _clip_cell(r, "mpp_telemetry", pct),
                 _clip_cell(r, "mpp_car", pct), veh, _clip_cell(r, "complete_frac", pct),
                 _clip_cell(r, "speed_median_kmh", one), _clip_cell(r, "speed_p85_kmh", one),
                 _clip_cell(r, "noise_median_kmh", one), " / ".join(ev),
                 _clip_cell(r, "abnormal_frac", pct), _clip_cell(r, "auc_both", pct)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def analytics_table(df: pd.DataFrame, clips: pd.DataFrame | None) -> str:
    """Traffic analytics: event AUC, origin-destination accuracy, per-clip table and the honesty note."""
    d = df.assign(slice=df["slice"].fillna("all") if "slice" in df else "all")
    d = d[is_analytics(d)]
    if d.empty:
        return "(no analytics results yet)"
    d = _latest(d)
    parts = [_anomaly_part(d), _od_part(d[d["slice"] == "all"])]
    if clips is not None and len(clips):
        parts.append(_clips_part(clips))
    return "\n\n".join(p for p in parts if p) + "\n\n" + ANALYTICS_NOTE
