"""Append-only results store. README tables are generated from this file, never typed."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

# `slice` is "all" for whole-set metrics; sub-set slices (e.g. day/night) may come later.
# `set_hash` identifies the exact image list of the evaluated set (empty for old rows).
RESULT_COLS = [
    "timestamp", "git_commit", "model", "model_sha", "eval_set", "slice", "imgsz", "metric",
    "value", "ci_lo", "ci_hi", "n_images", "n_scenes", "manifest_hash", "set_hash",
    "config_hash", "notes",
]


def append_results(path: Path, rows: list[dict]) -> None:
    """Append `rows` (missing keys -> NaN). Refuses to append under a different header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.exists() and path.stat().st_size > 0
    if has_header:
        with open(path, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != RESULT_COLS:
            raise ValueError(
                f"{path}: header {header} differs from RESULT_COLS {RESULT_COLS}; "
                "migrate the file before appending"
            )
    df = pd.DataFrame(rows, columns=RESULT_COLS)
    df.to_csv(path, mode="a", header=not has_header, index=False, lineterminator="\n")
