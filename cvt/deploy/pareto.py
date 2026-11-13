"""Pareto report of the deployment grid: accuracy (T-VN / T-VD mAP50-95 from results.csv) joined
with end-to-end latency (latency.csv), the frontier, the plot and the README block.

Frontier: a point is on it unless another point has T-VN mAP50-95 >= and median latency <= with
at least one strictly better (so equal latency keeps the higher mAP, and points equal on both are
all kept). A point missing accuracy or latency is never on the frontier and never dominates.

Stale accuracy: for a full-frame point, an accuracy row whose `model_sha` (file_sha256 of the
engine / checkpoint, first 12 hex) differs from the latest latency row's `weights_sha[:12]` was
measured on other weights (e.g. an engine rebuilt after `cvt eval`); it is blanked and the point
flagged `accuracy_stale`. Tiled points' model_sha is a detector cache tag and is not checked."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.deploy.engines import PRECISIONS, grid_points, grid_tile, registry_name
from cvt.eval.report import _fmt

PLACEHOLDER = "(no deployment results yet)"
METRIC = "mAP50_95"
SETS = ("T-VN", "T-VD")
PARETO_COLS = [
    "model", "student", "imgsz", "precision", "tiled",
    "T-VN_mAP50_95", "T-VN_ci_lo", "T-VN_ci_hi", "T-VD_mAP50_95", "T-VD_ci_lo", "T-VD_ci_hi",
    "median_ms", "p90_ms", "fps", "gpu_mem_mb", "accuracy_stale", "frontier",
]
# first slots of the validated categorical palette (valid all-pairs, so usable in a scatter)
STUDENT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb"
LAT_NUM = ("median_ms", "p90_ms", "fps", "gpu_mem_mb")
MARKERS = {"fp32": "s", "fp16": "o", "int8": "^"}


def frontier_mask(acc, ms) -> np.ndarray:
    """Boolean mask of the non-dominated points (accuracy up, latency down); see module doc."""
    a, m = np.asarray(acc, dtype=float), np.asarray(ms, dtype=float)
    if a.shape != m.shape or a.ndim != 1:
        raise ValueError(f"acc and ms must be 1-D arrays of the same length, got {a.shape} "
                         f"and {m.shape}")
    ok = np.isfinite(a) & np.isfinite(m)
    out = np.zeros(len(a), dtype=bool)
    for i in np.flatnonzero(ok):
        better = ok & (a >= a[i]) & (m <= m[i]) & ((a > a[i]) | (m < m[i]))
        out[i] = not better.any()
    return out


# --- table --------------------------------------------------------------------------------------

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_true(v) -> bool:
    return bool(v) if isinstance(v, bool | np.bool_) else str(v).strip().lower() == "true"


def _latest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Last row per key (by timestamp, ties: last appended), as the results report does."""
    return df.sort_values("timestamp", kind="stable").groupby(keys).tail(1)


def _accuracy(results: pd.DataFrame
              ) -> dict[tuple[str, int, str], tuple[tuple[float, float, float], str]]:
    """((value, ci_lo, ci_hi), model_sha) of the last row per (model, imgsz, eval_set): rows
    evaluated at another input size never stand in for a point."""
    d = results
    if "slice" in d:
        d = d[d["slice"].fillna("all") == "all"]
    d = d[(d.metric == METRIC) & d.eval_set.isin(SETS)]
    d = d.assign(imgsz=_num(d.imgsz), value=_num(d.value), ci_lo=_num(d.ci_lo),
                 ci_hi=_num(d.ci_hi)).dropna(subset=["imgsz"])
    d = _latest(d, ["model", "imgsz", "eval_set", "metric"])
    sha = d.model_sha.fillna("").astype(str) if "model_sha" in d else pd.Series("", index=d.index)
    return {(r.model, int(r.imgsz), r.eval_set): ((float(r.value), float(r.ci_lo),
                                                   float(r.ci_hi)), s)
            for r, s in zip(d.itertuples(), sha, strict=True)}


def _points(latest: pd.DataFrame, cfg: dict | None) -> list[tuple[str, str, int, str, bool]]:
    """(registry name, student, imgsz, precision, tiled): the grid when `cfg` is given, else
    the points of the latency rows."""
    if cfg is not None:
        tile = grid_tile(cfg)
        return [(registry_name(p, tile), p.model, p.imgsz, p.precision, p.tiled)
                for p in grid_points(cfg)]
    return [(r.model, r.model.split("@")[0], int(float(r.imgsz)), str(r.precision),
             _is_true(r.tiled)) for r in latest.itertuples()]


def pareto_table(results: pd.DataFrame, latency: pd.DataFrame, cfg: dict | None = None
                 ) -> pd.DataFrame:
    """One row per grid point (from `cfg`, else from the latency rows): last accuracy row per
    (model, imgsz, eval_set) and last latency row per model, plus the `frontier` flag. Sorted by
    student, tiled, imgsz, precision."""
    lat = _latest(latency, ["model"]) if len(latency) else latency
    lat = lat.assign(**{c: _num(lat[c]) for c in LAT_NUM}).set_index("model")
    acc = _accuracy(results) if len(results) else {}
    rows = []
    for name, student, imgsz, precision, tiled in _points(lat.reset_index(), cfg):
        row = {"model": name, "student": student, "imgsz": imgsz, "precision": precision,
               "tiled": tiled, "accuracy_stale": False}
        wsha = ""
        if not tiled and name in lat.index and "weights_sha" in lat:
            wsha = str(lat.at[name, "weights_sha"])
            wsha = "" if wsha in ("", "nan") else wsha[:12]
        for s in SETS:
            vals, sha = acc.get((name, imgsz, s), ((np.nan,) * 3, ""))
            if wsha and not np.isnan(vals[0]) and sha != wsha:  # measured on other weights
                vals, row["accuracy_stale"] = (np.nan,) * 3, True
            row.update(zip((f"{s}_mAP50_95", f"{s}_ci_lo", f"{s}_ci_hi"), vals, strict=True))
        for c in LAT_NUM:
            row[c] = float(lat.at[name, c]) if name in lat.index else np.nan
        rows.append(row)
    table = pd.DataFrame(rows, columns=PARETO_COLS[:-1])
    if table.empty:
        return pd.DataFrame(columns=PARETO_COLS).astype(
            {"frontier": bool, "tiled": bool, "accuracy_stale": bool})
    order = {p: i for i, p in enumerate(PRECISIONS)}
    table = table.sort_values(
        ["student", "tiled", "imgsz", "precision"], kind="stable",
        key=lambda c: c.map(order) if c.name == "precision" else c).reset_index(drop=True)
    table["accuracy_stale"] = table["accuracy_stale"].astype(bool)
    table["frontier"] = frontier_mask(table["T-VN_mAP50_95"], table["median_ms"])
    return table[PARETO_COLS]


# --- plot ---------------------------------------------------------------------------------------

def _short(name: str) -> str:
    return re.sub(r"^[^@]*?-(yolo\w+?)-s\d+(?=@)", r"\1", name)


def plot_pareto(table: pd.DataFrame, out: Path) -> Path:
    """T-VN mAP50-95 (%) against median latency (ms, log x): 95% CI bars, colour per student,
    marker per precision (hollow = tiled), frontier line and labelled frontier points."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    d = table.dropna(subset=["T-VN_mAP50_95", "median_ms"])
    if d.empty:
        raise ValueError("no point with both T-VN mAP50-95 and median latency to plot")
    students = sorted(table.student.unique())
    if len(students) > len(STUDENT_COLORS):
        raise ValueError(f"the plot colours at most {len(STUDENT_COLORS)} students, got "
                         f"{len(students)}")
    color = dict(zip(students, STUDENT_COLORS, strict=False))
    fig = Figure(figsize=(10, 6), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    f = d[d.frontier].sort_values("median_ms")
    if len(f) > 1:
        ax.plot(f.median_ms, 100 * f["T-VN_mAP50_95"], color=INK_MUTED, linewidth=1.5,
                zorder=2)
    for r in d.itertuples(index=False):
        v, lo, hi = r[5], r[6], r[7]
        c, mk = color[r.student], MARKERS[r.precision]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(r.median_ms, 100 * v, yerr=[[100 * max(v - lo, 0)], [100 * max(hi - v, 0)]],
                        fmt="none", ecolor=c, elinewidth=1, alpha=0.55, capsize=0, zorder=3)
        ax.scatter([r.median_ms], [100 * v], s=60, marker=mk, zorder=4, linewidths=1.5,
                   facecolors="none" if r.tiled else c, edgecolors=c)
    for i, r in enumerate(f.itertuples(index=False)):
        ax.annotate(_short(r.model), (r.median_ms, 100 * r[5]), fontsize=7, color=INK,
                    textcoords="offset points", xytext=(6, 7 if i % 2 == 0 else -12), zorder=5)
    ax.set_xlabel("Median end-to-end latency per frame (ms, log scale)", color=INK)
    ax.set_ylabel("T-VN mAP50-95 (%)", color=INK)
    handles = [Line2D([], [], marker="o", linestyle="", color=color[s], label=s)
               for s in students]
    handles += [Line2D([], [], marker=MARKERS[p], linestyle="", color=INK_MUTED, label=p)
                for p in PRECISIONS if p in set(table.precision)]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                       color=INK_MUTED, label="tiled"),
                Line2D([], [], color=INK_MUTED, linewidth=1.5, label="frontier")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


# --- README block -------------------------------------------------------------------------------

def _distinct(df: pd.DataFrame, col: str) -> str:
    vals = [str(v) for v in dict.fromkeys(df[col]) if str(v).strip() and str(v) != "nan"]
    return "/".join(vals) if vals else "?"


def _frames_phrase(latency: pd.DataFrame) -> str:
    """`199 fixed T-VN frames (187 at 3840x2160, 12 at 1920x1080)` from the rows' frame-size
    mix; without a single recorded mix just the frame count."""
    head = f"{_distinct(latency, 'n_frames')} fixed T-VN frames"
    mixes = [m for m in dict.fromkeys(latency.get("frame_sizes", pd.Series(dtype=str)).astype(str))
             if m.strip() and m != "nan"]
    if len(mixes) != 1:
        return head
    parts = [kv.rsplit(":", 1) for kv in mixes[0].split(";")]
    if len(parts) == 1:
        return f"{head} (all {parts[0][0]})"
    return f"{head} ({', '.join(f'{c} at {s}' for s, c in parts)})"


def _decode_sentence(latency: pd.DataFrame) -> str:
    ms = _num(latency.get("decode_ms_per_frame", pd.Series(dtype=str))).dropna()
    if ms.empty:
        return ""
    return (f" Decoding a frame (cv2.imread, excluded from the timings) took "
            f"{float(np.median(ms)):.1f} ms.")


def _stale_sentence(table: pd.DataFrame) -> str:
    names = table.model[table.accuracy_stale.astype(bool)].tolist()
    if not names:
        return ""
    return (f" Accuracy stale: re-run eval for {', '.join(names)} (engine or weights changed "
            "since their accuracy rows, so their accuracy is left blank).")


def _caption(latency: pd.DataFrame, table: pd.DataFrame) -> str:
    return (
        f"Latency: {_distinct(latency, 'gpu')} desktop GPU (driver {_distinct(latency, 'driver')}, "
        f"TensorRT {_distinct(latency, 'tensorrt')}, Ultralytics "
        f"{_distinct(latency, 'ultralytics')}, PyTorch {_distinct(latency, 'torch')}); a "
        "Jetson-class device would be several times slower and was not measured. Protocol: batch 1, "
        "end-to-end per frame (letterbox, inference, NMS, class mapping; tiles for tiled "
        f"points) on {_frames_phrase(latency)} decoded before timing, after "
        f"{_distinct(latency, 'warmup')} warm-up calls; median and p90 over the frames, "
        f"fps = 1000 / median.{_decode_sentence(latency)} Accuracy: mAP50-95 in %, 95% CI in "
        "brackets (cluster bootstrap over clips). Models are trained at 1280, so inputs other "
        "than 1280 are inference-only changes. All points use a square letterbox at their input "
        "size (imgsz x imgsz, also the PyTorch FP32 ones). gpu_mem_mb is a device-level "
        "steady-state footprint (free device memory before the detector is built minus after "
        "the timed loop, in MB), not a per-process figure. Frontier (T-VN mAP50-95 vs median "
        "latency): no other point is at least as accurate and at least as fast, and better on "
        f"one.{_stale_sentence(table)}"
    )


def _md_table(t: pd.DataFrame, with_frontier: bool) -> str:
    cols = ["point", "imgsz", "precision", "tiled", "T-VN mAP50-95", "T-VD mAP50-95",
            "median ms", "p90 ms", "fps", "gpu_mem_mb"] + (["frontier"] if with_frontier else [])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def num(v: float, spec: str) -> str:
        return "–" if pd.isna(v) else format(v, spec)

    for r in t.itertuples(index=False):
        cells = [r.model, str(r.imgsz), r.precision, "yes" if r.tiled else "no",
                 _fmt(r[5], r[6], r[7]), _fmt(r[8], r[9], r[10]), num(r.median_ms, ".2f"),
                 num(r.p90_ms, ".2f"), num(r.fps, ".1f"), num(r.gpu_mem_mb, ".0f")]
        if with_frontier:
            cells.append("yes" if r.frontier else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pareto_block(table: pd.DataFrame, latency: pd.DataFrame, png: str | Path | None = None
                 ) -> str:
    """Markdown for the README: frontier table, full table, caption, plot. `latency` supplies the
    hardware and software versions of the caption (only the rows of the table's models)."""
    if table.empty or table.median_ms.isna().all():
        return PLACEHOLDER
    lat = _latest(latency[latency.model.isin(table.model)], ["model"])
    front = table[table.frontier].sort_values("median_ms", kind="stable")
    parts = [
        "Frontier points (not dominated on T-VN mAP50-95 vs median latency), fastest first:",
        _md_table(front, with_frontier=False),
        "All grid points:",
        _md_table(table, with_frontier=True),
        _caption(lat, table),
    ]
    if png is not None:
        parts.append(f"![Pareto]({Path(png).as_posix()})")
    return "\n\n".join(parts)
