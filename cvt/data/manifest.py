"""One parquet manifest for every labelled image, with named evaluation/training sets."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.adapters.uavdt import (
    SEQ_ATTR_COLS,
    SEQ_ATTR_DTYPES,
    empty_seq_attrs,
    load_uavdt,
)
from cvt.data.adapters.uitdrone7 import load_uitdrone7
from cvt.data.adapters.visdrone_det import load_visdrone_det
from cvt.data.adapters.visdrone_mot import (
    MOT_GT_COLS,
    MOT_GT_DTYPES,
    empty_mot_gt,
    load_visdrone_mot,
)
from cvt.data.schema import CLASSES, FrameAnn

# set name -> role. "test" sets are frozen; "pool" is the hidden-label AL pool.
SET_ROLES: dict[str, str] = {
    "VD-train": "train",
    "VD-tune": "tune",
    "T-VD": "test",
    "UD-pool": "pool",
    "T-VN": "test",
    "P-train": "train",
    "P-val": "tune",
    "MOT-tune": "tune",
    "T-MOT": "test",
    "T-UAVDT": "test",
    "AL-revealed": "train",
}
# Pool frames whose labels the AL oracle revealed (export only, never a manifest set).
REVEALED_SET = "AL-revealed"
IMAGE_COLS = ["image_id", "path", "width", "height", "source", "scene_id", "split", "set_name"]
BOX_COLS = ["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"]
BOX_DTYPES = {"image_id": "str", "x1": "float64", "y1": "float64", "x2": "float64",
              "y2": "float64", "label": "int64", "is_ignore": "bool"}
# whole sequences removed by a recorded decision (`drop_sequences` of a MOT source)
SEQ_DROP_DTYPES = {"source": "str", "seq_id": "str", "set_name": "str", "n_images": "int64",
                   "reason": "str"}
SEQ_DROP_COLS = list(SEQ_DROP_DTYPES)
DROP_SOURCES = ("visdrone_mot", "uavdt")


def empty_seq_drops() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in SEQ_DROP_DTYPES.items()})


class ManifestError(ValueError):
    pass


def _require(frames: list[FrameAnn], what: str) -> list[FrameAnn]:
    if not frames:
        raise ManifestError(f"{what}: no labelled images found (check the configured paths)")
    return frames


def tune_scenes(scene_sizes: dict[str, int], frac: float) -> set[str]:
    """Deterministic scene-level carve-out holding ~`frac` of the images.

    Scenes are taken in sha1 order until the image share reaches `frac`, so a few huge
    scenes cannot skew the fraction and the result does not depend on input order.
    """
    order = sorted(scene_sizes, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    target = frac * sum(scene_sizes.values())
    picked: set[str] = set()
    total = 0
    for s in order:
        if total >= target:
            break
        picked.add(s)
        total += scene_sizes[s]
    return picked


def frames_to_tables(
    frames: list[FrameAnn], set_name_of: Callable[[FrameAnn], str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    img_rows, box_rows = [], []
    for f in frames:
        img_rows.append(
            (f.image_id, f.path, f.width, f.height, f.source, f.scene_id, f.split, set_name_of(f))
        )
        for b, lab in zip(f.boxes, f.labels, strict=True):
            box_rows.append((f.image_id, *map(float, b), int(lab), False))
        for b in f.ignore:
            box_rows.append((f.image_id, *map(float, b), -1, True))
    return pd.DataFrame(img_rows, columns=IMAGE_COLS), pd.DataFrame(box_rows, columns=BOX_COLS)


def check_expected_counts(images: pd.DataFrame, expected: dict) -> None:
    """Pre-flight: `expected` maps set name -> [images, scenes]; any mismatch raises."""
    bad = []
    for set_name, want in expected.items():
        g = images[images.set_name == set_name]
        got = (len(g), g.scene_id.nunique())
        want = (int(want[0]), int(want[1]))
        if got != want:
            bad.append(f"{set_name}: got {got}, expected {want}")
    if bad:
        raise ManifestError("set sizes differ from the configured `expected` counts: " + "; ".join(bad))


def _drop_list(sources: dict) -> list[tuple[str, str, str]]:
    """(source, seq, reason) from the optional `drop_sequences: [{seq, reason}]` of MOT sources."""
    out = []
    for source in DROP_SOURCES:
        items = (sources.get(source) or {}).get("drop_sequences") or []
        if not isinstance(items, list):
            raise ManifestError(f"{source}.drop_sequences must be a list of {{seq, reason}}")
        for i, item in enumerate(items):
            ok = (isinstance(item, dict) and set(item) == {"seq", "reason"}
                  and all(isinstance(item[k], str) and item[k].strip() for k in ("seq", "reason")))
            if not ok:
                raise ManifestError(f"{source}.drop_sequences[{i}]: expected {{seq: <name>, "
                                    f"reason: <text>}} with non-empty strings, got {item!r}")
            if any(s == source and q == item["seq"] for s, q, _ in out):
                raise ManifestError(f"{source}.drop_sequences lists {item['seq']} twice")
            out.append((source, item["seq"], item["reason"].strip()))
    return out


def drop_sequences(
    images: pd.DataFrame, drops: list[tuple[str, str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(images without the dropped sequences, the drop record SEQ_DROP_COLS). A listed sequence
    that is not in the manifest raises."""
    rows, scenes = [], set()
    for source, seq, reason in drops:
        scene = f"{source}/{seq}"
        hit = images[(images.source == source) & (images.scene_id == scene)]
        if hit.empty:
            raise ManifestError(f"{source}.drop_sequences: sequence {seq} not found in the "
                                f"{source} frames")
        names = sorted(set(hit.set_name))
        rows.append((source, seq, ",".join(names), len(hit), reason))
        scenes.add(scene)
    record = (pd.DataFrame(rows, columns=SEQ_DROP_COLS).astype(SEQ_DROP_DTYPES) if rows
              else empty_seq_drops())
    return images[~images.scene_id.isin(scenes)].reset_index(drop=True), record


def build_manifest(
    cfg: dict, exclusions: set[str], report: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the manifest from the configured sources.

    Set sizes are checked against `cfg["expected"]` (if present) before sequence drops and
    exclusions are applied, since both are recorded decisions on top of the downloaded data.
    `drop_sequences` of a MOT source removes whole sequences from any set (test sets included:
    the leakage decision path); `exclusions` never touch test images. If `report` is given it
    receives `unmatched_exclusions`: exclusion ids that match no image, `seq_drops`: the drop
    record, and `mot_gt` and `seq_attrs`: the MOT ground-truth tracks and UAVDT sequence
    attributes of the surviving frames (empty tables when no MOT source is configured); pass
    the three tables to `save_manifest`.
    """
    sources = cfg["sources"]
    parts: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    pool_val_ids: dict[str, set[str]] = {}  # pool val video -> its frame image ids
    mot_parts: list[pd.DataFrame] = []
    attr_parts: list[pd.DataFrame] = []
    if "visdrone_det" in sources:
        vd = sources["visdrone_det"]
        frac, merge = float(vd["tune_frac"]), vd["rider_merge"]
        if not isinstance(merge, bool):
            raise ManifestError("visdrone_det.rider_merge must be true or false")
        if not 0 < frac < 1:
            raise ManifestError(f"visdrone_det.tune_frac must be in (0, 1), got {frac}")
        val = _require(load_visdrone_det(Path(vd["val_dir"]), "val", merge), "VisDrone val")
        val_scenes = {f.scene_id for f in val}
        # VisDrone train and val share some source sequences; keep the official val intact
        # and drop the train-side siblings so no sequence spans train and test.
        train = _require(
            [
                f for f in load_visdrone_det(Path(vd["train_dir"]), "train", merge)
                if f.scene_id not in val_scenes
            ],
            "VisDrone train (after dropping val siblings)",
        )
        tune = tune_scenes(Counter(f.scene_id for f in train), frac)
        parts.append(
            frames_to_tables(train, lambda f: "VD-tune" if f.scene_id in tune else "VD-train")
        )
        parts.append(frames_to_tables(val, lambda f: "T-VD"))
    if "uitdrone7" in sources:
        u = sources["uitdrone7"]
        imgs, labs = Path(u["images_dir"]), Path(u["labels_dir"])
        uit_train = _require(load_uitdrone7(imgs, labs, "train"), "UIT-Drone7 train")
        uit_test = _require(load_uitdrone7(imgs, labs, "test"), "UIT-Drone7 test")
        parts.append(frames_to_tables(uit_train, lambda f: "UD-pool"))
        parts.append(frames_to_tables(uit_test, lambda f: "T-VN"))
    if "visdrone_mot" in sources:
        vm = sources["visdrone_mot"]
        merge = vm["rider_merge"]
        if not isinstance(merge, bool):
            raise ManifestError("visdrone_mot.rider_merge must be true or false")
        for key, split, set_name, what in (
            ("val_dir", "val", "MOT-tune", "VisDrone-MOT val"),
            ("test_dir", "test", "T-MOT", "VisDrone-MOT test"),
        ):
            frames, gt = load_visdrone_mot(Path(vm[key]), split, merge)
            parts.append(frames_to_tables(_require(frames, what), lambda f, n=set_name: n))
            mot_parts.append(gt)
    if "uavdt" in sources:
        uc = sources["uavdt"]
        if "attr_dir" in uc:
            raise ManifestError("uavdt.attr_dir is no longer used: list the split in "
                                "uavdt.test_sequences and the flags in uavdt.attributes")
        seqs, attributes = uc.get("test_sequences"), uc.get("attributes")
        if not isinstance(seqs, list) or not seqs:
            raise ManifestError("uavdt.test_sequences must be a non-empty list of sequences")
        if not isinstance(attributes, dict) or not all(
            isinstance(v, list) for v in attributes.values()
        ):
            raise ManifestError("uavdt.attributes must map each attribute to a list of sequences")
        frames, gt, attrs = load_uavdt(
            Path(uc["images_dir"]), Path(uc["gt_dir"]), seqs, attributes, "test"
        )
        parts.append(frames_to_tables(_require(frames, "UAVDT test"), lambda f: "T-UAVDT"))
        mot_parts.append(gt)
        attr_parts.append(attrs)
    if "pool" in sources:
        pc = sources["pool"]
        index_file = Path(pc["index"])
        if not index_file.exists():
            raise ManifestError(f"pool index {index_file} missing: run `cvt pool extract` first")
        index = pd.read_parquet(index_file)
        if index.empty:
            raise ManifestError("pool index is empty")
        val = set(pc.get("val_videos") or [])
        unknown = sorted(val - set(index.video_id))
        if unknown:
            raise ManifestError(f"pool val_videos not in the index: {unknown}")
        pool = index[[c for c in IMAGE_COLS if c != "set_name"]].copy()
        pool["set_name"] = ["P-val" if v in val else "P-train" for v in index.video_id]
        pool_val_ids = {v: set(index.image_id[index.video_id == v]) for v in sorted(val)}
        parts.append((pool, pd.DataFrame(columns=BOX_COLS)))
    if not parts:
        raise ManifestError("no sources configured")

    images = pd.concat([p[0] for p in parts], ignore_index=True)
    box_parts = [p[1] for p in parts if len(p[1])]
    boxes = (pd.concat(box_parts, ignore_index=True) if box_parts
             else pd.DataFrame({c: pd.Series(dtype=d) for c, d in BOX_DTYPES.items()}))
    check_expected_counts(images, cfg.get("expected") or {})
    if report is not None:
        report["unmatched_exclusions"] = sorted(set(exclusions) - set(images.image_id))
    kept, seq_drops = drop_sequences(images, _drop_list(sources))
    drop = set(images.image_id) - set(kept.image_id)
    images = kept
    is_test = images.set_name.map(SET_ROLES) == "test"
    drop |= set(images.loc[images.image_id.isin(exclusions) & ~is_test, "image_id"])
    images = images[~images.image_id.isin(drop)].reset_index(drop=True)
    boxes = boxes[~boxes.image_id.isin(drop)].reset_index(drop=True)
    mot_gt = pd.concat([empty_mot_gt(), *mot_parts], ignore_index=True)
    mot_gt = mot_gt[mot_gt.image_id.isin(images.image_id)].reset_index(drop=True)
    seq_attrs = pd.concat([empty_seq_attrs(), *attr_parts], ignore_index=True)
    gone_uav = set(seq_drops.seq_id[seq_drops.source == "uavdt"])
    seq_attrs = seq_attrs[~seq_attrs.seq_id.isin(gone_uav)].reset_index(drop=True)
    if report is not None:
        report["mot_gt"], report["seq_attrs"] = mot_gt, seq_attrs
        report["seq_drops"] = seq_drops
    missing = [v for v, ids in pool_val_ids.items() if not images.image_id.isin(ids).any()]
    if missing:
        raise ManifestError(
            f"pool val video(s) {missing} have no frames left after exclusions; "
            "choose other val_videos"
        )
    validate_manifest(images, boxes)
    return images, boxes


def validate_manifest(images: pd.DataFrame, boxes: pd.DataFrame) -> None:
    if images.image_id.duplicated().any():
        raise ManifestError("duplicate image ids")
    unknown_sets = set(images.set_name) - set(SET_ROLES)
    if unknown_sets:
        raise ManifestError(f"unknown set names {sorted(unknown_sets)}")
    if (images.set_name == REVEALED_SET).any():
        raise ManifestError(f"{REVEALED_SET} is an export-only set, not a manifest set")
    orphans = ~boxes.image_id.isin(images.image_id)
    if orphans.any():
        raise ManifestError(f"{orphans.sum()} boxes reference unknown images")

    m = boxes.merge(images[["image_id", "width", "height"]], on="image_id")
    coords = m[["x1", "y1", "x2", "y2"]].to_numpy(float)
    malformed = (
        (m.x2 <= m.x1) | (m.y2 <= m.y1) | (m.x1 < 0) | (m.y1 < 0) | (m.x2 > m.width) | (m.y2 > m.height)
        | ~np.isfinite(coords).all(axis=1)
    )
    bad_label = ~m.is_ignore & ((m.label < 0) | (m.label >= len(CLASSES)))
    if malformed.any() or bad_label.any():
        first = m[malformed | bad_label].head(3).to_dict("records")
        raise ManifestError(
            f"{malformed.sum()} malformed boxes, {bad_label.sum()} bad labels; first: {first}"
        )

    for source, g in images.groupby("source"):
        scenes = g.groupby("set_name").scene_id.apply(set)
        names = list(scenes.index)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = scenes[a] & scenes[b]
                if shared:
                    raise ManifestError(
                        f"{source}: scenes shared between {a} and {b}: {sorted(shared)[:5]}"
                    )


def manifest_hash(
    images: pd.DataFrame,
    boxes: pd.DataFrame,
    mot_gt: pd.DataFrame | None = None,
    seq_attrs: pd.DataFrame | None = None,
    seq_drops: pd.DataFrame | None = None,
) -> str:
    """Content hash; excludes `path` so relocating the data folder keeps the hash. MOT tables
    (track ids, sequence attributes) and the sequence-drop record are hashed only when
    non-empty, so a manifest without them keeps the hash it had before they existed."""
    h = hashlib.sha256()
    img = images[[c for c in IMAGE_COLS if c != "path"]].astype({"width": "int64", "height": "int64"})
    box = boxes[BOX_COLS].astype(
        {"x1": "float64", "y1": "float64", "x2": "float64", "y2": "float64",
         "label": "int64", "is_ignore": bool}
    )
    extra = [t[cols].astype(dtypes) for t, cols, dtypes in (
        (mot_gt, MOT_GT_COLS, MOT_GT_DTYPES), (seq_attrs, SEQ_ATTR_COLS, SEQ_ATTR_DTYPES),
        (seq_drops, SEQ_DROP_COLS, SEQ_DROP_DTYPES))
        if t is not None and len(t)]
    for df in (img, box, *extra):
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def save_manifest(
    images: pd.DataFrame,
    boxes: pd.DataFrame,
    out_dir: Path,
    mot_gt: pd.DataFrame | None = None,
    seq_attrs: pd.DataFrame | None = None,
    seq_drops: pd.DataFrame | None = None,
) -> str:
    """Write the manifest; the MOT tables and the sequence-drop record are always written (empty
    when not given) so a rebuild never leaves stale tracks or drops from an earlier build behind."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mot_gt = empty_mot_gt() if mot_gt is None else mot_gt
    seq_attrs = empty_seq_attrs() if seq_attrs is None else seq_attrs
    seq_drops = empty_seq_drops() if seq_drops is None else seq_drops
    images.to_parquet(out_dir / "images.parquet", index=False)
    boxes.to_parquet(out_dir / "boxes.parquet", index=False)
    mot_gt.to_parquet(out_dir / "mot_gt.parquet", index=False)
    seq_attrs.to_parquet(out_dir / "seq_attrs.parquet", index=False)
    seq_drops.to_parquet(out_dir / "seq_drops.parquet", index=False)
    h = manifest_hash(images, boxes, mot_gt, seq_attrs, seq_drops)
    (out_dir / "HASH").write_text(h, encoding="utf-8")
    return h


def load_manifest(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    out_dir = Path(out_dir)
    images = pd.read_parquet(out_dir / "images.parquet")
    boxes = pd.read_parquet(out_dir / "boxes.parquet")
    return images, boxes, manifest_hash(images, boxes, *load_mot(out_dir),
                                        load_seq_drops(out_dir))


def load_seq_drops(out_dir: Path) -> pd.DataFrame:
    """The sequence-drop record (SEQ_DROP_COLS); empty when the manifest has none."""
    path = Path(out_dir) / "seq_drops.parquet"
    return pd.read_parquet(path) if path.exists() else empty_seq_drops()


def load_mot(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(MOT ground-truth tracks `MOT_GT_COLS`, UAVDT sequence attributes `SEQ_ATTR_COLS`); both
    empty with the right columns when the manifest has no MOT sets."""
    out_dir = Path(out_dir)
    gt, attrs = out_dir / "mot_gt.parquet", out_dir / "seq_attrs.parquet"
    return (
        pd.read_parquet(gt) if gt.exists() else empty_mot_gt(),
        pd.read_parquet(attrs) if attrs.exists() else empty_seq_attrs(),
    )


def frames_from_manifest(
    images: pd.DataFrame, boxes: pd.DataFrame, set_name: str
) -> list[FrameAnn]:
    if set_name not in SET_ROLES:
        raise ManifestError(f"unknown set {set_name!r}")
    sel = images[images.set_name == set_name]
    by_id = {k: g for k, g in boxes[boxes.image_id.isin(sel.image_id)].groupby("image_id")}
    no_boxes = pd.DataFrame(columns=BOX_COLS)
    frames = []
    for r in sel.itertuples(index=False):
        g = by_id.get(r.image_id, no_boxes)
        obj, ign = g[~g.is_ignore.astype(bool)], g[g.is_ignore.astype(bool)]
        frames.append(
            FrameAnn(
                image_id=r.image_id,
                path=r.path,
                width=int(r.width),
                height=int(r.height),
                source=r.source,
                scene_id=r.scene_id,
                split=r.split,
                boxes=obj[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                labels=obj.label.to_numpy(np.int64),
                ignore=ign[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
            )
        )
    return frames
