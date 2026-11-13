"""Static batch-1 vs batched tile engines (configs/tiles.yaml, README block `tiles`).

Each pair has the same weights, tiles and merge; only the tile engine differs (the deployment grid's static
batch-1 640 engine, called once per tile, vs a batch-32 engine taking all of a 4K frame's tiles in
one call). Accuracy is the last T-VN/T-VD row at 1280 (results.csv); latency is the last row per
model of the pair file (`latency`), which scripts/tiles.ps1 times in one session together with the
`reference` full-frame point, so the comparison never mixes machine states."""

from __future__ import annotations

import pandas as pd

from cvt.al.report import _latest, _num
from cvt.eval.report import _fmt

PLACEHOLDER = "(no batched tile results yet)"
IMGSZ = 1280
METRICS = (("T-VN", "mAP50_95"), ("T-VN", "AP50/motorbike"), ("T-VD", "mAP50_95"))


def _ms(latency: pd.DataFrame) -> dict[str, float]:
    if latency is None or not len(latency):
        return {}
    d = latency.sort_values("timestamp", kind="stable").groupby("model").tail(1)
    return dict(zip(d.model, _num(d.median_ms), strict=True))


def tiles_table(results: pd.DataFrame, latency: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per pair: accuracy of both points (value, lo, hi), session latency of both and the
    speed-up (static ms / batched ms)."""
    acc, ms = _latest(results, IMGSZ), _ms(latency)
    nan3 = (float("nan"),) * 3
    rows = []
    for p in cfg["pairs"]:
        r = {"static": p["static"], "batched": p["batched"]}
        for kind in ("static", "batched"):
            for s, m in METRICS:
                r[f"{kind}/{s}/{m}"] = acc.get((p[kind], s, m), nan3)
            r[f"{kind}/ms"] = ms.get(p[kind], float("nan"))
        r["speedup"] = r["static/ms"] / r["batched/ms"]
        rows.append(r)
    return pd.DataFrame(rows)


def _ms_txt(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}"


def _x(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}x"


def tiles_block(table: pd.DataFrame, latency: pd.DataFrame, cfg: dict,
                plan4_latency: pd.DataFrame | None = None) -> str:
    """Markdown for `<!-- tiles:start -->`; PLACEHOLDER until a batched point is timed."""
    if not len(table) or table["batched/ms"].isna().all():
        return PLACEHOLDER
    head = ["static -> batched", "T-VN mAP50-95 static / batched",
            "T-VN AP50 motorbike static / batched", "T-VD mAP50-95 static / batched",
            "ms static", "ms batched", "fps batched", "speed-up"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in table.iterrows():
        cells = [f"{_fmt(*r[f'static/{s}/{m}'])} / {_fmt(*r[f'batched/{s}/{m}'])}"
                 for s, m in METRICS]
        b = r["batched/ms"]
        lines.append(f"| {r['static']} -> `b32` | " + " | ".join(cells)
                     + f" | {_ms_txt(r['static/ms'])} | {_ms_txt(b)} | "
                     + ("–" if pd.isna(b) else f"{1000 / b:.1f}") + f" | {_x(r['speedup'])} |")
    ref = cfg["reference"]
    now, then = _ms(latency).get(ref), _ms(plan4_latency).get(ref)
    note = ("Median end-to-end ms per frame (199 fixed T-VN frames, 187 of them 4K; batch 1 "
            "frame), all timed in one session")
    if now is not None and not pd.isna(now):
        note += (f"; the reference {ref} took {now:.1f} ms in that session"
                 + (f" vs {then:.1f} ms in the deployment benchmark" if then is not None and not pd.isna(then)
                    else ""))
    note += (". Accuracy in %, 95% CI in brackets (cluster bootstrap over clips), evaluated at "
             f"{IMGSZ}; static and batched use the same weights, tiles and merge, so their "
             "accuracy should agree up to engine numerics (INT8 engines are calibrated separately).")
    return "\n".join(lines) + "\n\n" + note
