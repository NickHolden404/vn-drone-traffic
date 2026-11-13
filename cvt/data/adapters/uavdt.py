"""UAVDT (UAV-benchmark-M frames + UAV-benchmark-MOTD_v1.0 GT), checked against the real download.

Frames are `images_dir/<seq>/img<frame:06d>.jpg`. Per sequence, `gt_dir` holds (see the toolkit
README.md):
- `<seq>_gt.txt`: the official MOT ground truth `frame,id,x,y,w,h,score,in_view,occlusion`, score
  1 = evaluated, 0 = ignored; no class column.
- `<seq>_gt_whole.txt`: the DET ground truth `frame,id,x,y,w,h,out_of_view,occlusion,category`
  (1 car, 2 truck, 3 bus); a superset of `_gt.txt` (test sequences add partly out-of-view rows).
- `<seq>_gt_ignore.txt`: ignore regions `frame,id,x,y,w,h,1,-1,-1`.

The MOT GT is the score-1 `_gt.txt` rows labelled by joining `_gt_whole` on (frame, id); score-0
rows, `_gt_whole` rows absent from `_gt.txt` and `_gt_ignore` rows are ignore regions (not
evaluated by the official MOT protocol). The detection boxes are the same MOT GT boxes. There is
no attribute folder: the test split and sequence attributes come from the config (copied from the
official toolkit's CalculateTrakingAcc.m).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.adapters.visdrone_mot import MOT_GT_COLS, MOT_GT_DTYPES, empty_mot_gt
from cvt.data.schema import CLASS_ID, FrameAnn, clip_boxes
from cvt.utils import image_size

SOURCE = "uavdt"
UAVDT_RAW: dict[int, int] = {1: CLASS_ID["car"], 2: CLASS_ID["truck"], 3: CLASS_ID["bus"]}
SEQ_ATTR_COLS = ["seq_id", "daylight", "night", "fog", "low_alt", "medium_alt", "high_alt",
                 "front_view", "side_view", "bird_view"]
SEQ_ATTR_DTYPES = {c: ("str" if c == "seq_id" else "bool") for c in SEQ_ATTR_COLS}
N_FIELDS = 9  # all three GT files
_IMG = re.compile(r"img(\d+)\.jpg$", re.IGNORECASE)


def empty_seq_attrs() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in SEQ_ATTR_DTYPES.items()})


def _rows(path: Path):
    """(lineno, 9 ints) per non-blank line; fails with `path:line` on bad rows."""
    if not path.is_file():
        raise FileNotFoundError(f"missing UAVDT file: {path}")
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
            yield lineno, [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: non-integer field ({e})") from None


def _parse_tracks(path: Path, key_cols: tuple[int, ...]) -> dict[tuple[int, int], tuple]:
    """(frame, id) -> (lineno, row) in file order. Exact duplicates of (frame, id, box and the
    `key_cols` fields) are dropped; a (frame, id) taken by a different row fails."""
    out: dict[tuple[int, int], tuple] = {}
    for lineno, v in _rows(path):
        key = (v[0], v[1])
        if key in out:
            first, w = out[key]
            if [w[i] for i in (*range(6), *key_cols)] != [v[i] for i in (*range(6), *key_cols)]:
                raise ValueError(f"{path}:{lineno}: track {v[1]} appears twice in frame {v[0]} "
                                 f"with different boxes (first at line {first})")
            continue
        out[key] = (lineno, v)
    return out


def _parse_whole(path: Path) -> dict[tuple[int, int], tuple]:
    whole = _parse_tracks(path, (8,))
    for lineno, v in whole.values():
        if v[8] not in UAVDT_RAW:
            raise ValueError(f"{path}:{lineno}: unknown UAVDT category {v[8]}")
    return whole


def _parse_gt(path: Path, whole: dict[tuple[int, int], tuple], whole_path: Path):
    """frame -> rows `id x y w h category` (score 1) and frame -> ignore rows `x y w h` (score 0,
    plus `_gt_whole` rows absent from `_gt.txt`). Every `_gt.txt` row must match a `_gt_whole`
    row with the same (frame, id) and box."""
    objects: dict[int, list[list[int]]] = defaultdict(list)
    ignore: dict[int, list[list[int]]] = defaultdict(list)
    gt = _parse_tracks(path, (6,))
    for key, (lineno, v) in gt.items():
        if v[6] not in (0, 1):
            raise ValueError(f"{path}:{lineno}: score must be 0 or 1, got {v[6]}")
        if key not in whole:
            raise ValueError(f"{path}:{lineno}: frame {v[0]} track {v[1]} has no row in "
                             f"{whole_path}")
        wline, w = whole[key]
        if w[2:6] != v[2:6]:
            raise ValueError(f"{path}:{lineno}: box differs from {whole_path}:{wline} "
                             f"({v[2:6]} vs {w[2:6]})")
        if v[6] == 1:
            objects[v[0]].append([v[1], *v[2:6], w[8]])
        else:
            ignore[v[0]].append(v[2:6])
    for key, (_, w) in whole.items():
        if key not in gt:
            ignore[w[0]].append(w[2:6])
    return objects, ignore


def _xyxy(xywh: np.ndarray) -> np.ndarray:
    return np.stack(
        [xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1
    ).astype(np.float32)


def _attr_table(sequences: list[str], attributes: dict[str, list[str]]) -> pd.DataFrame:
    keys = SEQ_ATTR_COLS[1:]
    unknown = sorted(set(attributes) - set(keys))
    if unknown:
        raise ValueError(f"unknown UAVDT attribute(s) {unknown}; expected {keys}")
    missing = [k for k in keys if k not in attributes]
    if missing:
        raise ValueError(f"missing UAVDT attribute(s) {missing}")
    for k in keys:
        stray = sorted(set(attributes[k]) - set(sequences))
        if stray:
            raise ValueError(f"UAVDT attribute {k}: {stray} not in the sequence list")
    rows = [(s, *(s in set(attributes[k]) for k in keys)) for s in sequences]
    return pd.DataFrame(rows, columns=SEQ_ATTR_COLS).astype(SEQ_ATTR_DTYPES)


def load_uavdt(
    images_dir: Path, gt_dir: Path, sequences: list[str], attributes: dict[str, list[str]],
    split: str,
) -> tuple[list[FrameAnn], pd.DataFrame, pd.DataFrame]:
    """Load `sequences` (in order). Every frame on disk becomes a FrameAnn whose boxes are the MOT
    GT (car/truck/bus); `attributes` maps each SEQ_ATTR_COLS flag to the sequences that have it.
    Returns (frames, MOT GT table, sequence attribute table)."""
    images_dir, gt_dir = Path(images_dir), Path(gt_dir)
    if not sequences:
        raise ValueError("no UAVDT sequences given")
    twice = sorted(s for s, n in Counter(sequences).items() if n > 1)
    if twice:
        raise ValueError(f"UAVDT sequence(s) listed twice: {twice}")
    attrs = _attr_table(list(sequences), attributes)
    frames: list[FrameAnn] = []
    rows: list[tuple] = []
    for seq in sequences:
        seq_dir = images_dir / seq
        images = {int(m.group(1)): p for p in seq_dir.glob("*.jpg") if (m := _IMG.match(p.name))}
        if not images:
            raise FileNotFoundError(f"no img<frame>.jpg frames in {seq_dir}")
        gt_path, whole_path = gt_dir / f"{seq}_gt.txt", gt_dir / f"{seq}_gt_whole.txt"
        ignore_path = gt_dir / f"{seq}_gt_ignore.txt"
        objects, ignore = _parse_gt(gt_path, _parse_whole(whole_path), whole_path)
        for _, v in _rows(ignore_path):
            ignore[v[0]].append(v[2:6])
        for kind, table in (("annotated", objects), ("ignore", ignore)):
            missing = sorted(set(table) - set(images))
            if missing:
                raise FileNotFoundError(
                    f"{gt_dir}/{seq}_gt*.txt: {kind} frames without an image in {seq_dir}: "
                    f"{[f'img{f:06d}.jpg' for f in missing[:5]]}"
                )
        for fno in sorted(images):
            img = images[fno]
            w, h = image_size(img)
            a = np.array(objects.get(fno, []), np.int64).reshape(-1, 6)
            boxes, keep = clip_boxes(_xyxy(a[:, 1:5]), w, h)
            ids, labels = a[keep, 0], np.array([UAVDT_RAW[c] for c in a[keep, 5]], np.int64)
            ign = np.array(ignore.get(fno, []), np.int64).reshape(-1, 4)
            regions, _ = clip_boxes(_xyxy(ign), w, h)
            image_id = f"{SOURCE}/{seq}/{fno:06d}"
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
                    ignore=regions,
                )
            )
            for tid, b, lab in zip(ids, boxes, labels, strict=True):
                rows.append((seq, fno, image_id, int(tid), *map(float, b), int(lab)))
    gt = pd.DataFrame(rows, columns=MOT_GT_COLS) if rows else empty_mot_gt()
    return frames, gt.astype(MOT_GT_DTYPES), attrs
