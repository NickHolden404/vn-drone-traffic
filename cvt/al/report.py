"""Active-learning report: the accuracy-vs-budget curve table and plot, the paired deltas,
the seed spread at 400, the headline engine table and sentence (README block `al`).

Accuracy is the last results row per (model, imgsz 1280, eval_set, metric), slice `all`. Budget 0
is the starting student (`students.<size>.plan2_job` of configs/al.yaml); all-pool runs have budget
POOL_FRAMES. Deltas are never computed here: they are the `delta/...` rows `cvt compare` wrote
(model = a, notes `vs <b>@<imgsz>`). The headline sentence uses only the pre-declared primary
run's engine (`headline` at `engines.imgsz` / `engines.precision`); a class "reaches 90" iff its
T-VN AP50 point estimate, as printed with one decimal (in %), is >= 90.0, always with its
95% CI. The `fallbacks` points (pre-declared before any T-VN evaluation of an AL run) are reported
as secondary table rows and a secondary sentence; they never change the headline verdict."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.al.prepare import _parse_runs
from cvt.deploy.engines import engine_name
from cvt.deploy.pareto import GRID, INK, INK_MUTED, STUDENT_COLORS, SURFACE
from cvt.eval.report import _fmt, _fmt_delta

PLACEHOLDER = "(no active-learning results yet)"
POOL_FRAMES = 9352  # UD-pool frames (spec): the budget of the all-pool runs
IMGSZ = 1280
CLASSES6 = ("car", "motorbike", "bus", "truck", "pedestrian", "bicycle")
METRICS = (("T-VN", "mAP50_95"), *(("T-VN", f"AP50/{c}") for c in CLASSES6), ("T-VD", "mAP50_95"))
CURVE_STRATEGIES = ("random", "disagree", "variety")
ORDER = {s: i for i, s in enumerate(("start", *CURVE_STRATEGIES, "all"))}
COLORS = dict(zip(CURVE_STRATEGIES, STUDENT_COLORS, strict=True))
TARGET = 90.0  # "reaches 90": one-decimal point estimate of AP50 (in %) >= 90.0
X0 = 50  # where budget 0 (the starting student) sits on the log2 budget axis


def metric_col(eval_set: str, metric: str) -> str:
    return f"{eval_set}_{metric.replace('/', '_')}"


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rows_at(results: pd.DataFrame, imgsz: int = IMGSZ) -> pd.DataFrame:
    d = results
    if d.empty:
        return d
    sl = d["slice"].fillna("all").astype(str).replace("", "all") if "slice" in d else "all"
    d = d[sl == "all"]
    d = d.assign(imgsz=_num(d.imgsz), value=_num(d.value), ci_lo=_num(d.ci_lo),
                 ci_hi=_num(d.ci_hi))
    return d[d.imgsz == imgsz].sort_values("timestamp", kind="stable")


def _latest(results: pd.DataFrame, imgsz: int = IMGSZ
            ) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """(model, eval_set, metric) -> (value, ci_lo, ci_hi) of the last row at `imgsz`."""
    d = _rows_at(results, imgsz)
    if d.empty:
        return {}
    d = d.groupby(["model", "eval_set", "metric"]).tail(1)
    return {(r.model, r.eval_set, r.metric): (r.value, r.ci_lo, r.ci_hi) for r in d.itertuples()}


def _latest_deltas(results: pd.DataFrame
                   ) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    """(a, b, eval_set, metric without `delta/`) -> last `cvt compare` row, both at 1280."""
    d = _rows_at(results)
    if d.empty:
        return {}
    d = d[d.metric.astype(str).str.startswith("delta/")]
    vs = d.notes.fillna("").astype(str).str.split(";").str[0].str.strip()
    d = d.assign(vs=vs)[vs.str.startswith("vs ") & vs.str.endswith(f"@{IMGSZ}")]
    d = d.groupby(["model", "vs", "eval_set", "metric"]).tail(1)
    return {(r.model, r.vs[3:].rsplit("@", 1)[0], r.eval_set, r.metric[6:]):
            (r.value, r.ci_lo, r.ci_hi) for r in d.itertuples()}


# --- curve table --------------------------------------------------------------------------------

def curve_table(results: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per starting student (budget 0) and per inventory run: run info, then value /
    `_lo` / `_hi` columns for T-VN mAP50-95, T-VN AP50 of the six classes and T-VD mAP50-95."""
    rows = [{"run": st["plan2_job"], "group": "start", "student": size, "strategy": "start",
             "budget": 0, "ranking_seed": None, "seed": 0} for size, st in cfg["students"].items()]
    budget = _budget_of(cfg)
    rows += [{"run": s["name"], "group": s["group"], "student": s["student"],
              "strategy": s["strategy"], "budget": budget[s["name"]],
              "ranking_seed": s["ranking_seed"], "seed": s["seed"]} for s in _parse_runs(cfg)]
    acc = _latest(results)
    for r in rows:
        for s, m in METRICS:
            c = metric_col(s, m)
            r[c], r[f"{c}_lo"], r[f"{c}_hi"] = acc.get((r["run"], s, m), (np.nan,) * 3)
    t = pd.DataFrame(rows)
    t["ranking_seed"] = t["ranking_seed"].astype("Int64")
    return t


# --- plot ---------------------------------------------------------------------------------------

PANELS = (("T-VN_mAP50_95", "T-VN mAP50-95 (%)"), ("T-VN_AP50_motorbike", "T-VN motorbike AP50 (%)"),
          ("T-VN_AP50_car", "T-VN car AP50 (%)"))


def plot_curve(table: pd.DataFrame, out: Path) -> Path:
    """s student: T-VN mAP50-95, motorbike and car AP50 against labelled frames (log2 x), one line
    per strategy from the starting student (budget 0), 95% CI bars, the all-pool run dashed."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s = table[table.student == "s"]
    curve = s[s.group.isin(["start", "al_curve"])]
    ref = s[s.strategy == "all"]
    cols = [c for c, _ in PANELS]
    if not np.isfinite(pd.concat([curve, ref])[cols].to_numpy(dtype=float)).any():
        raise ValueError("nothing to plot: no s-student AL run or starting student has T-VN "
                         "results at 1280")
    fig = Figure(figsize=(13, 4.4), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(PANELS))
    start = curve[curve.strategy == "start"]
    for ax, (col, label) in zip(axes, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_MUTED)
        r = ref[np.isfinite(ref[col].astype(float))]
        if len(r):
            v, lo, hi = (100 * float(r[c].iloc[0]) for c in (col, f"{col}_lo", f"{col}_hi"))
            ax.axhline(v, color=INK_MUTED, linestyle="--", linewidth=1.2,
                       label=f"all {POOL_FRAMES:,} pool frames")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axhspan(lo, hi, color=INK_MUTED, alpha=0.08, linewidth=0)
        for k, st in enumerate(CURVE_STRATEGIES):
            d = pd.concat([start, curve[curve.strategy == st]]).sort_values("budget")
            d = d[np.isfinite(d[col].astype(float))]
            if d.empty or (d.budget > 0).sum() == 0:
                continue
            # budget 0 at X0; strategies nudged apart (x 2^+-0.06) so their CI bars don't overlap
            x = np.where(d.budget > 0, d.budget * 2 ** (0.06 * (k - 1)), X0)
            v = 100 * d[col].astype(float).to_numpy()
            err = np.nan_to_num(np.vstack([v - 100 * d[f"{col}_lo"].astype(float),
                                           100 * d[f"{col}_hi"].astype(float) - v])).clip(min=0)
            err[:, (d.budget == 0).to_numpy()] = 0  # the starting student's CI is drawn once
            ax.errorbar(x, v, yerr=err, color=COLORS[st], marker="o", markersize=4,
                        linewidth=1.5, elinewidth=1, alpha=0.9, capsize=0, label=st)
        for _, row in start.iterrows():
            v, lo, hi = (100 * float(row[c]) for c in (col, f"{col}_lo", f"{col}_hi"))
            if np.isfinite(v):
                err = np.nan_to_num([[v - lo], [hi - v]]).clip(min=0)
                ax.errorbar([X0], [v], yerr=err, color=INK, marker="o", markersize=6,
                            elinewidth=1, capsize=0, zorder=5, label="starting student")
        ax.set_xticks([X0, 100, 200, 400, 800])
        ax.set_xticklabels(["0\n(start)", "100", "200", "400", "800"])
        ax.minorticks_off()
        ax.set_xlim(X0 / 1.35, 800 * 1.35)
        ax.set_xlabel("Labelled pool frames (log2 scale)", color=INK)
        ax.set_ylabel(label, color=INK)
    handles, labels = {}, []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    axes[0].legend([handles[lab] for lab in labels], labels, loc="lower right", fontsize=8,
                   frameon=False, labelcolor=INK)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


# --- headline -----------------------------------------------------------------------------------

def _engines(cfg: dict) -> tuple[list[str], int, str]:
    e = cfg["engines"]
    if cfg["headline"] not in e["models"]:
        raise ValueError(f"headline {cfg['headline']!r} is not in engines.models {e['models']}")
    return list(e["models"]), int(e["imgsz"]), str(e["precision"])


def fallback_points(cfg: dict) -> list[dict]:
    """The pre-declared secondary points of the primary run: [{name, imgsz, label}]."""
    return [{"name": str(p["name"]), "imgsz": int(p["imgsz"]), "label": str(p["label"])}
            for p in (cfg.get("fallbacks") or {}).get("points") or []]


def _budget_of(cfg: dict) -> dict[str, int]:
    out = {st["plan2_job"]: 0 for st in cfg["students"].values()}
    out |= {s["name"]: POOL_FRAMES if s["budget"] is None else int(s["budget"])
            for s in _parse_runs(cfg)}
    return out


def headline_sentence(results: pd.DataFrame, cfg: dict) -> str:
    """Car and motorbike T-VN AP50 of the primary run's engine against the 90 target."""
    _, imgsz, precision = _engines(cfg)
    head = cfg["headline"]
    name = engine_name(head, imgsz, precision)
    acc = _latest(results, imgsz)
    vals = {c: acc.get((name, "T-VN", f"AP50/{c}")) for c in ("car", "motorbike")}
    lead = (f"**Headline** ({name}: the pre-declared primary run {head} as a TensorRT "
            f"{precision.upper()} engine at {imgsz} px)")
    if any(v is None or not math.isfinite(v[0]) for v in vals.values()):
        return f"{lead}: the headline engine is not evaluated yet on T-VN."
    n = _budget_of(cfg)[head]
    found = ", ".join(f"{c} AP50 {_fmt(*v)}" for c, v in vals.items())
    verdict = ", ".join(f"{c} {'reaches' if reaches_target(v[0]) else 'does not reach'} 90"
                        for c, v in vals.items())
    out = (f"{lead}: {found} on T-VN with {n} labelled frames — {verdict} (reaching 90 means "
           "the one-decimal point estimate is >= 90; 95% CI in brackets).")
    sec = []
    for fb in fallback_points(cfg):
        acc_fb = _latest(results, fb["imgsz"])
        v = {c: acc_fb.get((fb["name"], "T-VN", f"AP50/{c}")) for c in ("car", "motorbike")}
        if any(x is None or not math.isfinite(x[0]) for x in v.values()):
            continue
        sec.append(f"{fb['label']}: " + ", ".join(
            f"{c} {_fmt(*x)} ({'reaches' if reaches_target(x[0]) else 'does not reach'} 90)"
            for c, x in v.items()))
    if sec:
        out += (" Secondary, pre-declared fallbacks of the same run (never the headline): "
                + "; ".join(sec) + ".")
    return out


def reaches_target(v: float) -> bool:
    """The rule on the printed value (`_fmt`'s one decimal), so verdict and number agree."""
    return float(f"{100 * v:.1f}") >= TARGET


def _latest_latency(latency: pd.DataFrame | None) -> dict[str, float]:
    if latency is None or not len(latency):
        return {}
    d = latency.sort_values("timestamp", kind="stable").groupby("model").tail(1)
    return dict(zip(d.model, _num(d.median_ms), strict=True))


def _headline_table(results: pd.DataFrame, latency: pd.DataFrame | None, cfg: dict) -> str:
    models, imgsz, precision = _engines(cfg)
    start = cfg["students"][_student_of(cfg, cfg["headline"])]["plan2_job"]
    budget = _budget_of(cfg)
    acc, lat = _latest(results, imgsz), _latest_latency(latency)
    head = (["engine", "labelled frames", "T-VN mAP50-95"] + [f"T-VN AP50 {c}" for c in CLASSES6]
            + ["T-VD mAP50-95", "median ms"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = [(engine_name(m, imgsz, precision), budget[m], acc) for m in [*models, start]]
    rows += [(fb["name"], budget[cfg["headline"]], _latest(results, fb["imgsz"]))
             for fb in fallback_points(cfg)]
    for name, n, a in rows:
        cells = [_fmt(*a.get((name, s, k), (np.nan,) * 3)) for s, k in METRICS]
        ms = lat.get(name, np.nan)
        lines.append(f"| {name} | {n} | " + " | ".join(cells)
                     + f" | {'–' if pd.isna(ms) else format(ms, '.2f')} |")
    return "\n".join(lines)


def _student_of(cfg: dict, run: str) -> str:
    return next(s["student"] for s in _parse_runs(cfg) if s["name"] == run)


# --- README block -------------------------------------------------------------------------------


def _seeds(r: pd.Series) -> str:
    if r["strategy"] == "start":
        return "–"
    seeds = f"training {r['seed']}"
    return seeds if pd.isna(r["ranking_seed"]) else f"ranking {r['ranking_seed']}, {seeds}"


def _pt(v: float) -> str:
    return "–" if pd.isna(v) else f"{100 * v:.1f}"


def _cell(r: pd.Series, eval_set: str, metric: str) -> str:
    c = metric_col(eval_set, metric)
    return _fmt(r[c], r[f"{c}_lo"], r[f"{c}_hi"])


def _curve_md(table: pd.DataFrame) -> str:
    """Evaluated runs, per student and strategy by budget: mAP50-95 with CIs, AP50 points."""
    t = table[table[[metric_col(s, m) for s, m in METRICS]].notna().any(axis=1)]
    t = t.assign(_s=t.student.map({"s": 0, "m": 1}), _o=t.strategy.map(ORDER),
                 _r=t.ranking_seed.fillna(-1)).sort_values(
        ["_s", "_o", "budget", "_r", "seed"], kind="stable")
    head = (["run", "strategy", "budget", "student", "seeds", "T-VN mAP50-95"]
            + [f"AP50 {c}" for c in CLASSES6] + ["T-VD mAP50-95"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in t.iterrows():
        cells = ([r["run"], r["strategy"], str(r["budget"]), r["student"], _seeds(r),
                  _cell(r, "T-VN", "mAP50_95")]
                 + [_pt(r[metric_col("T-VN", f"AP50/{c}")]) for c in CLASSES6]
                 + [_cell(r, "T-VD", "mAP50_95")])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


DELTA_COLS = (("T-VN", "mAP50_95"), ("T-VN", "AP50/motorbike"), ("T-VN", "AP50/car"),
              ("T-VD", "mAP50_95"))


def _deltas_md(results: pd.DataFrame, cfg: dict) -> str:
    """The configured comparisons that have `cvt compare` rows (Δ = a − b, points)."""
    deltas = _latest_deltas(results)
    head = ["a", "b"] + [f"{s} Δ{m.replace('_', '-')}" for s, m in DELTA_COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in cfg["comparisons"]:
        cells = [deltas.get((c["a"], c["b"], s, m)) for s, m in DELTA_COLS]
        if all(v is None for v in cells):
            continue
        lines.append(f"| {c['a']} | {c['b']} | " + " | ".join(
            "–" if v is None else _fmt_delta(*v) for v in cells) + " |")
    if len(lines) == 2:
        return "(no paired comparisons yet)"
    return "\n".join(lines)


def _spread(vals: list[float]) -> str:
    v = [x for x in vals if not pd.isna(x)]
    return f"{100 * (max(v) - min(v)):.1f}" if len(v) >= 2 else "–"


def _spread_md(table: pd.DataFrame) -> str:
    """Per variance family (the curve run plus its `al_variance` runs): T-VN mAP50-95 and
    motorbike AP50 per seed and their max − min, in points ('' without variance runs)."""
    t = table.set_index("run", drop=False)
    var = t[t.group == "al_variance"]
    if var.empty:
        return ""
    head = ["runs", "seeds", "T-VN mAP50-95", "spread", "T-VN AP50 motorbike", "spread"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (student, strategy, budget), g in var.groupby(["student", "strategy", "budget"],
                                                       sort=False):
        base = t[(t.group == "al_curve") & (t.student == student) & (t.strategy == strategy)
                 & (t.budget == budget)]
        fam = pd.concat([base, g])
        ranking = fam.ranking_seed.nunique(dropna=True) > 1
        seeds = "/".join(str(x) for x in (fam.ranking_seed if ranking else fam.seed))
        cells = [f"{', '.join(fam.run)}", f"{'ranking' if ranking else 'training'} seeds {seeds}"]
        for col in ("T-VN_mAP50_95", "T-VN_AP50_motorbike"):
            vals = list(fam[col])
            cells += [" / ".join(_pt(v) for v in vals), _spread(vals)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def al_block(results: pd.DataFrame, latency: pd.DataFrame | None, cfg: dict,
             png: str | Path | None = None) -> str:
    """Markdown for the README (`<!-- al:start -->`): headline sentence and engine table, curve
    plot, per-strategy table, paired deltas and seed spread; PLACEHOLDER without AL rows."""
    models, imgsz, precision = _engines(cfg)
    fallbacks = fallback_points(cfg)
    fallback_note = ("; the rows after it are pre-declared secondary fallbacks of the primary run ("
                     + ", ".join(fb["label"] for fb in fallbacks) + "), never the headline")
    names = ({s["name"] for s in _parse_runs(cfg)}
             | {engine_name(m, imgsz, precision) for m in models} | {fb["name"] for fb in fallbacks})
    if results.empty or not names & set(results.model.astype(str)):
        return PLACEHOLDER
    table = curve_table(results, cfg)
    parts = [headline_sentence(results, cfg), _headline_table(results, latency, cfg)]
    if png is not None:
        parts.append(f"![Active-learning curve]({Path(png).as_posix()})")
    parts += [
        "Per run (budget = labelled pool frames; 0 = the starting student, "
        f"{POOL_FRAMES} = the whole pool):",
        _curve_md(table),
        "Paired comparisons (Δ = a − b in points, 95% paired cluster-bootstrap CI over clips, "
        "from `cvt compare`):",
        _deltas_md(results, cfg),
    ]
    spread = _spread_md(table)
    if spread:
        parts += ["Seed spread at the variance budget (points; spread = max − min):", spread]
    parts.append(
        "mAP50-95 and AP50 in %, 95% CI in brackets (cluster bootstrap over clips; per-class AP50 "
        "CIs of every run are in reports/al_curve.csv). AL runs: last weights after 15 fixed "
        f"epochs at {IMGSZ} px, evaluated at {IMGSZ}; engines: TensorRT {precision.upper()} at "
        f"{imgsz} px, median end-to-end latency per frame from reports/latency.csv (`cvt deploy "
        "bench`, same protocol as the deployment block); the row after the AL engines is the "
        "starting student's engine from the deployment grid for reference"
        + (fallback_note if fallbacks else "")
        + ". A class reaches 90 when its one-decimal AP50 point estimate is >= 90.0.")
    return "\n\n".join(parts)
