"""VisDrone2019-MOT: `frame,id,x,y,w,h,score,category,truncation,occlusion` per line, one .txt
per sequence; frames are `sequences/<seq>/<frame:07d>.jpg`."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.adapters.visdrone_det import _frame_from_rows
from cvt.data.classmaps import IGNORE, VISDRONE_RAW
from cvt.data.schema import FrameAnn
from cvt.utils import image_size

SOURCE = "visdrone_mot"
MOT_GT_COLS = ["seq_id", "frame", "image_id", "track_id", "x1", "y1", "x2", "y2", "label"]
MOT_GT_DTYPES = {"seq_id": "str", "frame": "int64", "image_id": "str", "track_id": "int64",
                 "x1": "float64", "y1": "float64", "x2": "float64", "y2": "float64",
                 "label": "int64"}
N_FIELDS = 10


def empty_mot_gt() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in MOT_GT_DTYPES.items()})


def _parse(path: Path) -> dict[int, np.ndarray]:
    """frame -> (N, 7) rows `id x y w h score category`; fails with `path:line` on bad rows.

    Exact duplicates of a row (frame and these 7 fields) are dropped, order kept; an object row
    (score != 0, not an ignore category) whose (frame, id) is already taken by a different row
    fails. Ignore-region rows may share ids."""
    by_frame: dict[int, list[list[int]]] = defaultdict(list)
    seen: set[tuple[int, ...]] = set()
    first_line: dict[tuple[int, int], int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != N_FIELDS:
            raise ValueError(
                f"{path}:{lineno}: expected {N_FIELDS} comma-separated fields, got {len(parts)}"
            )
        try:
            v = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: non-integer field ({e})") from None
        if v[7] not in VISDRONE_RAW:
            raise ValueError(f"{path}:{lineno}: unknown VisDrone category {v[7]}")
        row = tuple(v[:8])
        if row in seen:
            continue
        seen.add(row)
        if v[6] != 0 and VISDRONE_RAW[v[7]] != IGNORE:
            key = (v[0], v[1])
            if key in first_line:
                raise ValueError(f"{path}:{lineno}: track {v[1]} appears twice in frame {v[0]} "
                                 f"with different boxes (first at line {first_line[key]})")
            first_line[key] = lineno
        by_frame[v[0]].append(v[1:8])
    return {f: np.array(r, np.int64).reshape(-1, 7) for f, r in by_frame.items()}


def load_visdrone_mot(
    root: Path, split: str, rider_merge: bool
) -> tuple[list[FrameAnn], pd.DataFrame]:
    """Load `root/sequences` + `root/annotations`. Every frame on disk becomes a FrameAnn (the
    frame number is the file name, never its position); ignore regions and rider merging follow
    the DET rules. The GT table keeps the non-ignore boxes that survive, with their track ids."""
    root = Path(root)
    frames: list[FrameAnn] = []
    rows: list[tuple] = []
    anns = sorted((root / "annotations").glob("*.txt"))
    seq_dirs = {p.name for p in (root / "sequences").glob("*") if p.is_dir()}
    unlabelled = sorted(seq_dirs - {a.stem for a in anns})
    if unlabelled:
        raise FileNotFoundError(f"sequences without annotations in {root}: {unlabelled[:5]}")
    for ann in anns:
        seq = ann.stem
        seq_dir = root / "sequences" / seq
        if not seq_dir.is_dir():
            raise FileNotFoundError(f"annotation without sequence folder: {seq_dir}")
        images = {int(p.stem): p for p in seq_dir.glob("*.jpg")}
        by_frame = _parse(ann)
        missing = sorted(set(by_frame) - set(images))
        if missing:
            raise FileNotFoundError(
                f"{ann}: annotated frames without an image in {seq_dir}: "
                f"{[f'{f:07d}.jpg' for f in missing[:5]]}"
            )
        for fno in sorted(images):
            img = images[fno]
            w, h = image_size(img)
            a = by_frame.get(fno, np.zeros((0, 7), np.int64))
            boxes, labels, ignore, src = _frame_from_rows(
                a[:, 1:5], a[:, 6], a[:, 5], w, h, rider_merge
            )
            image_id = f"{SOURCE}/{split}/{seq}/{fno:07d}"
            frames.append(
                FrameAnn(
                    image_id=image_id,
                    path=str(img),
                    width=w,
                    height=h,
                    source=SOURCE,
                    scene_id=f"{SOURCE}/{seq}",
                    split=split,
                    boxes=boxes,
                    labels=labels,
                    ignore=ignore,
                )
            )
            for tid, b, lab in zip(a[src, 0], boxes, labels, strict=True):
                rows.append((seq, fno, image_id, int(tid), *map(float, b), int(lab)))
    gt = pd.DataFrame(rows, columns=MOT_GT_COLS) if rows else empty_mot_gt()
    return frames, gt.astype(MOT_GT_DTYPES)
