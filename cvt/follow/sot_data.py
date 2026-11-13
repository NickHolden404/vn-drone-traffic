"""Public single-object benchmarks for the target lock: UAVDT-S and the UAV123 / UAV20L vehicle
subsets, read in their native formats (inspected 2026-11-09).

UAVDT-S (`<root>` = data/external/uavdt_sot, from UAV-benchmark-S.zip and the SOT toolkit
UAV-benchmark-SOT_v1.0.zip):
- frames: `<root>/UAV-benchmark-S/<seq>/img000001.jpg ...` (50 folders S0101..S1702; 1024x540 or
  960x540);
- annotations: `<root>/UAV-benchmark-SOT_v1.0/anno/<seq>_gt.txt`, one `x,y,w,h` line per frame
  (comma-separated integers, top-left corner + size, no NaN rows, no header), as many lines as
  frames for all 50 sequences; per-sequence attribute flags in `anno/att/<seq>_att.txt` (9
  comma-separated 0/1: background clutter, camera motion, object motion, small object, illumination
  variations, object blur, scale variations, long-term tracking, large occlusion, per the
  toolkit's perfPlot.m). The toolkit's configSeqs.m runs every folder from frame 1 to its last.
- a sequence with annotations but no frame folder (or vice versa), or whose frame count differs
  from its annotation rows, is excluded by name (`uavdt_sot_inventory`), never guessed.

UAV123 (`<root>` = data/external/uav123/UAV123, from Dataset_UAV123.zip):
- frames: `<root>/data_seq/UAV123/<video>/000001.jpg ...` (1280x720);
- annotations: `<root>/anno/UAV123/<name>.txt` and `<root>/anno/UAV20L/<video>.txt`, one
  `x,y,w,h` line per frame (comma-separated, floats or integers), `NaN,NaN,NaN,NaN` where the
  target is absent (kept as NaN rows: not scored);
- several UAV123 sequences are pieces of one video (`car1_1`, `car1_2`, `car1_3` share
  `data_seq/UAV123/car1/`, the piece boundaries share a frame). Every UAV123 range (video,
  startFrame, endFrame) comes from the toolkit's `<root>/configSeqs.m` (cell array seqUAV123,
  parsed by `parse_config_seqs`), recorded in configs/follow_sot.yaml and re-checked
  against configSeqs.m on every load. The loader requires annotation rows == end - start + 1
  and every frame of the range on disk. Pieces of a UAV20L video are cross-checked (logged,
  never a source) by aligning their rows against the UAV20L full-length annotation
  (`crosscheck_uav20l`; the boundary rows of car1_2 / car6_1 differ by 1-2 px, the best
  offsets agree).

Boxes are returned as xyxy = (x, y, x + w, y + h), coordinates taken as given (no 1-based shift),
NaN rows for absent targets.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cvt.data.leakage import (
    REPORT_COLS,
    TRAINING_SIDE_ROLES,
    load_or_compute_hashes,
    near_duplicates,
)
from cvt.data.manifest import SET_ROLES
from cvt.follow.ope import VEHICLES

Log = Callable[[str], None]

ALIGN_TOL = 1e-6
LEAK_MAX_DIST = 6  # as `cvt leakage check`
LEAK_COLS = list(REPORT_COLS)
# result eval_set -> config section
SOT_SETS: dict[str, str] = {"UAVDT-S": "uavdt_s", "UAV123-veh": "uav123_veh",
                            "UAV20L-veh": "uav20l_veh"}
VEHICLE_NAME = re.compile(r"^(car|truck)\d+(_\d+)?$")  # UAV123 / UAV20L real vehicle videos
CONFIG_SEQS = "configSeqs.m"  # UAV123 toolkit sequence ranges
UAVDT_SEQ = re.compile(r"^S\d{4}$")


def _quiet(msg: str) -> None:
    pass


@dataclass
class SotSeq:
    """One single-object sequence: its frames in order, the target's GT box per frame (xyxy,
    NaN rows where absent), the frame rate, the video it is cut from (`<source>/<folder>`, the
    detection cache key) and the image number of its first frame in that video."""

    name: str
    frame_paths: list[str]
    gt: np.ndarray  # (n, 4) xyxy
    fps: float
    video: str = ""
    start: int = 1

    @property
    def end(self) -> int:
        return self.start + len(self.frame_paths) - 1


# --- native readers --------------------------------------------------------------------------

def read_xywh(path: Path) -> np.ndarray:
    """(n, 4) float rows of an `x,y,w,h` annotation file (NaN kept); blank lines skipped."""
    rows = []
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        vals = [v for v in re.split(r"[,\s]+", line) if v]
        if len(vals) != 4:
            raise ValueError(f"{path}:{k}: expected x,y,w,h, got {line!r}")
        rows.append([float(v) for v in vals])
    return np.asarray(rows, float).reshape(-1, 4)


def xywh_to_xyxy(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    return np.concatenate([a[:, :2], a[:, :2] + a[:, 2:]], axis=1)


def _uavdt_dirs(root: Path) -> tuple[Path, Path]:
    return Path(root) / "UAV-benchmark-S", Path(root) / "UAV-benchmark-SOT_v1.0" / "anno"


def uavdt_sot_inventory(root: Path) -> tuple[list[str], dict[str, str]]:
    """(usable sequence names, {excluded name: reason}) of a UAVDT-S tree."""
    fdir, adir = _uavdt_dirs(root)
    frames = {p.name for p in fdir.iterdir() if p.is_dir()} if fdir.exists() else set()
    annos = ({p.name.removesuffix("_gt.txt") for p in adir.glob("*_gt.txt")}
             if adir.exists() else set())
    ok, excluded = [], {}
    for name in sorted(frames | annos):
        if name not in frames:
            excluded[name] = "annotation but no frame folder"
            continue
        if name not in annos:
            excluded[name] = "frame folder but no annotation"
            continue
        n = len(list((fdir / name).glob("img*.jpg")))
        rows = len(read_xywh(adir / f"{name}_gt.txt"))
        if n != rows:
            excluded[name] = f"{n} frames, {rows} annotation rows"
        elif not all((fdir / name / f"img{i:06d}.jpg").exists() for i in (1, n)):
            excluded[name] = f"frames are not img000001..img{n:06d}"
        else:
            ok.append(name)
    return ok, excluded


def load_uavdt_sot(root: Path, fps: float = 30.0, log: Log = _quiet) -> list[SotSeq]:
    """Every usable UAVDT-S sequence (see `uavdt_sot_inventory`), frames 1..n."""
    fdir, adir = _uavdt_dirs(root)
    names, excluded = uavdt_sot_inventory(root)
    for name, why in excluded.items():
        log(f"[UAVDT-S] {name} excluded: {why}")
    out = []
    for name in names:
        gt = xywh_to_xyxy(read_xywh(adir / f"{name}_gt.txt"))
        paths = [str(fdir / name / f"img{i:06d}.jpg") for i in range(1, len(gt) + 1)]
        out.append(SotSeq(name, paths, gt, float(fps), f"UAVDT-S/{name}", 1))
    return out


def align_piece(piece: np.ndarray, full: np.ndarray, tol: float = ALIGN_TOL) -> list[int]:
    """Every 0-based offset o where `piece` equals `full[o:o + len(piece)]`: the same NaN rows and
    every other value within `tol`. A piece with no annotated row aligns nowhere."""
    piece, full = np.asarray(piece, float), np.asarray(full, float)
    n, m = len(piece), len(full)
    pnan, fnan = np.isnan(piece).any(1), np.isnan(full).any(1)
    if n == 0 or n > m or pnan.all():
        return []
    k = int(np.flatnonzero(~pnan)[0])
    first = np.flatnonzero(~fnan & (np.abs(full - piece[k]) <= tol).all(1)) - k
    out = []
    for o in first[(first >= 0) & (first <= m - n)]:
        seg = full[o:o + n]
        if (fnan[o:o + n] == pnan).all() and (np.abs(seg[~pnan] - piece[~pnan]) <= tol).all():
            out.append(int(o))
    return out


def best_alignment(piece: np.ndarray, full: np.ndarray,
                   tol: float = ALIGN_TOL) -> tuple[int, int] | None:
    """(0-based offset, rows equal) of the offset where the most rows of `piece` equal
    `full[o:o + len(piece)]` (NaN rows equal NaN rows, values within `tol`), over the offsets where
    the piece's first annotated row matches; None when there is none."""
    piece, full = np.asarray(piece, float), np.asarray(full, float)
    pnan = np.isnan(piece).any(1)
    if len(piece) == 0 or len(piece) > len(full) or pnan.all():
        return None
    k = int(np.flatnonzero(~pnan)[0])
    fnan = np.isnan(full).any(1)
    cand = np.flatnonzero(~fnan & (np.abs(full - piece[k]) <= tol).all(1)) - k
    best = None
    for o in cand[(cand >= 0) & (cand <= len(full) - len(piece))]:
        seg = full[o:o + len(piece)]
        same = int(np.where(pnan, np.isnan(seg).any(1), (np.abs(seg - piece) <= tol).all(1)).sum())
        if best is None or same > best[1]:
            best = (int(o), same)
    return best


def _video_dir(root: Path, video: str) -> Path:
    return Path(root) / "data_seq" / "UAV123" / video


_CELL = re.compile(r"^\s*(seq\w+)\s*=\s*\{", re.M)
_STRUCT = re.compile(r"struct\(\s*'name'\s*,\s*'([^']+)'\s*,\s*'path'\s*,\s*'([^']*)'\s*,\s*"
                     r"'startFrame'\s*,\s*(\d+)\s*,\s*'endFrame'\s*,\s*(\d+)")


def parse_config_seqs(path: Path) -> dict[str, dict[str, tuple[str, int, int]]]:
    """The UAV123 toolkit's configSeqs.m: {cell array name (e.g. "seqUAV123", "seqUAV20L"):
    {sequence name: (video folder, startFrame, endFrame)}}. The folder is the last component of
    each struct's Windows `path`."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    heads = list(_CELL.finditer(text))
    if not heads:
        raise ValueError(f"{path}: no `seq... = {{` cell arrays")
    out: dict[str, dict[str, tuple[str, int, int]]] = {}
    for h, nxt in zip(heads, [*heads[1:], None], strict=True):
        block = text[h.end():nxt.start() if nxt else len(text)]
        seqs: dict[str, tuple[str, int, int]] = {}
        for name, vpath, start, end in _STRUCT.findall(block):
            folder = [p for p in re.split(r"[\\/]", vpath) if p][-1]
            if name in seqs and seqs[name] != (folder, int(start), int(end)):
                raise ValueError(f"{path}: {h.group(1)} lists {name} twice with different ranges")
            seqs[name] = (folder, int(start), int(end))
        out[h.group(1)] = seqs
    return out


def official_ranges(root: Path, names: Sequence[str], cell: str = "seqUAV123",
                    ) -> dict[str, tuple[str, int, int]]:
    """(video, start, end) per name from `<root>/configSeqs.m` (cell array `cell`); a name it
    does not list raises."""
    ranges = parse_config_seqs(Path(root) / CONFIG_SEQS).get(cell, {})
    missing = [n for n in names if n not in ranges]
    if missing:
        raise ValueError(f"{Path(root) / CONFIG_SEQS} {cell} does not list {missing}")
    return {n: ranges[n] for n in names}


def crosscheck_uav20l(root: Path, name: str, video: str, start: int) -> str | None:
    """Cross-check (never a source) of an official piece range: where the most of the piece's
    annotation rows equal the UAV20L full-length annotation of `video`. None when `video` is not
    in UAV20L; else a one-line verdict starting "agrees" or "DISAGREES"."""
    full_path = Path(root) / "anno" / "UAV20L" / f"{video}.txt"
    if not full_path.exists():
        return None
    piece = read_xywh(Path(root) / "anno" / "UAV123" / f"{name}.txt")
    best = best_alignment(piece, read_xywh(full_path))
    if best is None:
        return f"DISAGREES: {name} aligns nowhere in UAV20L {video}"
    o, same = best
    verdict = "agrees" if o == start - 1 else "DISAGREES"
    return (f"{verdict}: {name} best matches UAV20L {video} frames {o + 1}..{o + len(piece)} "
            f"({same}/{len(piece)} rows equal), official start {start}")


def load_uav123(
    root: Path, names: Sequence[str], ranges: Mapping[str, Sequence] | None = None,
    subset: str = "UAV123", fps: float = 30.0, log: Log = _quiet,
) -> list[SotSeq]:
    """UAV123 (`subset` "UAV123") or UAV20L ("UAV20L") sequences by name. A name in `ranges` is
    (video, start, end) (configSeqs.m); any other name is the whole video `name` from frame 1, one
    annotation row per frame. Checks: annotation rows == end - start + 1 and every frame of the
    range exists, else ValueError. Pieces are cross-checked against UAV20L (logged only)."""
    root = Path(root)
    ranges = dict(ranges or {})
    out = []
    for name in names:
        path = root / "anno" / subset / f"{name}.txt"
        if not path.exists():
            raise ValueError(f"{name}: no {subset} annotation at {path}")
        anno = read_xywh(path)
        video, start, end = ranges.get(name, (name, 1, len(anno)))
        start, end = int(start), int(end)
        if end - start + 1 != len(anno):
            raise ValueError(f"{name}: range {start}..{end} of {video} has {end - start + 1} "
                             f"frames, its annotation {len(anno)} rows")
        vdir = _video_dir(root, video)
        paths = [str(vdir / f"{i:06d}.jpg") for i in range(start, end + 1)]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise ValueError(f"{name}: {len(missing)} frames of {video} {start}..{end} are "
                             f"missing, e.g. {missing[0]}")
        if subset == "UAV123" and name != video:
            check = crosscheck_uav20l(root, name, video, start)
            if check is not None:
                log(f"[UAV123] {name} {video} {start}..{end}: UAV20L cross-check {check}")
        out.append(SotSeq(name, paths, xywh_to_xyxy(anno), float(fps), f"UAV123/{video}", start))
    return out


# --- configs/follow_sot.yaml ------------------------------------------------------------------

def check_sot_cfg(cfg: dict) -> dict:
    """Validate the declared benchmark lists: vehicle names only, ranges for UAV123 pieces, no
    name both included and excluded."""
    for key in ("classes", *SOT_SETS.values()):
        if key not in (cfg or {}):
            raise ValueError(f"follow_sot config: missing key {key!r}")
    bad = [c for c in cfg["classes"] if c not in VEHICLES]
    if bad:
        raise ValueError(f"follow_sot config: classes {bad} are not vehicle classes {VEHICLES}")
    for key in SOT_SETS.values():
        sec = cfg[key]
        for k in ("root", "sequences", "excluded"):
            if k not in sec:
                raise ValueError(f"follow_sot config: {key} lacks {k!r}")
        names = list(sec["sequences"])
        both = sorted(set(names) & set(sec["excluded"] or {}))
        if both:
            raise ValueError(f"follow_sot config: {key} lists {both} as included and excluded")
        pat = UAVDT_SEQ if key == "uavdt_s" else VEHICLE_NAME
        odd = [n for n in names if not pat.match(str(n))]
        if odd:
            raise ValueError(f"follow_sot config: {key} names {odd} are not "
                             f"{'UAVDT-S sequences' if key == 'uavdt_s' else 'car*/truck* videos'}")
    for name, r in cfg["uav123_veh"]["sequences"].items():
        if not (isinstance(r, list | tuple) and len(r) == 3):
            raise ValueError(f"follow_sot config: uav123_veh {name} needs [video, start, end]")
    if not isinstance(cfg["uav20l_veh"]["sequences"], list):
        raise ValueError("follow_sot config: uav20l_veh sequences must be a list of videos")
    return cfg


def load_sot_cfg(path: Path) -> dict:
    return check_sot_cfg(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def check_sets(sets: Sequence[str]) -> None:
    bad = [s for s in sets if s not in SOT_SETS]
    if bad:
        raise ValueError(f"unknown benchmark sets {bad}: expected some of {list(SOT_SETS)}")


def set_fps(follow_cfg: dict, set_name: str) -> float:
    fps = (follow_cfg.get("fps") or {}).get(set_name)
    if fps is None:
        raise ValueError(f"no fps for {set_name} in the follow config (fps: {follow_cfg.get('fps')})")
    return float(fps)


def sot_sequences(sot_cfg: dict, set_name: str, follow_cfg: dict,
                  log: Log = _quiet) -> list[SotSeq]:
    """The declared sequences of one benchmark set, checked against the files on disk (and, for
    UAV123-veh, the declared ranges against configSeqs.m)."""
    check_sets([set_name])
    fps = set_fps(follow_cfg, set_name)
    sec = sot_cfg[SOT_SETS[set_name]]
    for name, why in (sec["excluded"] or {}).items():
        log(f"[{set_name}] {name} excluded: {why}")
    root = Path(sec["root"])
    if set_name == "UAVDT-S":
        seqs = load_uavdt_sot(root, fps, log)
        found, want = {s.name for s in seqs}, set(sec["sequences"])
        if found != want:
            raise ValueError(f"UAVDT-S on disk differs from configs/follow_sot.yaml: missing "
                             f"{sorted(want - found)}, undeclared {sorted(found - want)}")
        return seqs
    if set_name == "UAV123-veh":
        declared = {n: tuple(r) for n, r in sec["sequences"].items()}
        official = official_ranges(root, list(declared))
        off = sorted(n for n in declared if declared[n] != official[n])
        if off:
            raise ValueError(f"UAV123-veh ranges {off} differ from {root / CONFIG_SEQS}: "
                             f"{ {n: official[n] for n in off} }")
        return load_uav123(root, list(declared), declared, "UAV123", fps, log)
    return load_uav123(root, list(sec["sequences"]), None, "UAV20L", fps, log)


# --- leakage ---------------------------------------------------------------------------------

def frame_ids(seq: SotSeq) -> list[str]:
    """Image ids of a sequence's frames: `<video>/<image number>`."""
    return [f"{seq.video}/{seq.start + i}" for i in range(len(seq.frame_paths))]


def training_side(images: pd.DataFrame) -> pd.Series:
    """True for rows whose `set_name` is a training/tuning-side set (`TRAINING_SIDE_ROLES` of
    `cvt.data.leakage`: VD-train, VD-tune, UD-pool, P-train, P-val, MOT-tune, AL-revealed). Test
    sets (T-VD, T-VN, T-MOT, T-UAVDT) and unknown sets are False."""
    return images["set_name"].map(SET_ROLES).isin(TRAINING_SIDE_ROLES)


def sot_leakage(
    seqs: Sequence[SotSeq], train: pd.DataFrame, cache_file: Path,
    max_dist: int = LEAK_MAX_DIST, test_set: str = "UAVDT-S", log: Log = _quiet,
) -> tuple[pd.DataFrame, list[str]]:
    """pHash near-duplicates (Hamming <= max_dist, `cvt.data.leakage`) between the frames of
    `seqs` and the training/tuning-side images of `train` (image_id, set_name, phash). Returns the
    pairs (LEAK_COLS) and the sorted names of the sequences with any match: those are dropped.

    Only training/tuning-side sets count (`training_side`): rows of test sets in `train` are
    ignored, so overlap with another test set (UAVDT-S shares source videos with the UAVDT MOT
    test set T-UAVDT) is never leakage and never drops a sequence."""
    train = train[training_side(train)].reset_index(drop=True)
    ims = pd.DataFrame({"image_id": [i for s in seqs for i in frame_ids(s)],
                        "path": [p for s in seqs for p in s.frame_paths]})
    owner = np.repeat([s.name for s in seqs], [len(s.frame_paths) for s in seqs])
    if ims.empty or train.empty:
        return pd.DataFrame(columns=LEAK_COLS), []
    test_h = load_or_compute_hashes(ims, cache_file, log=log).to_numpy(np.uint64)
    pairs = near_duplicates(train["phash"].to_numpy(np.uint64), test_h, max_dist, chunk=256)
    if not len(pairs):
        return pd.DataFrame(columns=LEAK_COLS), []
    s, t = pairs[:, 0], pairs[:, 1]
    rep = pd.DataFrame({"train_id": train["image_id"].to_numpy()[s],
                        "train_set": train["set_name"].to_numpy()[s],
                        "test_id": ims["image_id"].to_numpy()[t], "test_set": test_set,
                        "dist": pairs[:, 2]})[LEAK_COLS]
    return rep.sort_values(["test_id", "dist"]).reset_index(drop=True), sorted(set(owner[t]))
