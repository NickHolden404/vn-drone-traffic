"""Manifest sets / pseudo labels -> Ultralytics YOLO dataset folders, list files and yaml.

Only training-side sets (roles train/tune) can be exported: test labels never leave the
manifest, and pool labels leave it only through `export_revealed` (frames the AL oracle revealed).
Ignore regions are painted grey (114, Ultralytics' letterbox fill) so the model is neither
rewarded nor punished there, and object labels >= 50% inside an ignore region are dropped (the
rule evaluation uses).

Exported images are written atomically (temp file + os.replace) and fingerprinted in
`<out_root>/<name>/.export.json` (source path, size, mtime and ignore regions), so a re-export
rewrites exactly the images whose source or ignore regions changed. Images without ignore
regions are hard-linked to the raw dataset file when possible: never edit exported images in
place, it would change the raw dataset file."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml
from PIL import Image, ImageDraw

from cvt.data.manifest import REVEALED_SET, SET_ROLES
from cvt.data.schema import CLASSES
from cvt.eval.coco_bridge import drop_in_ignore
from cvt.utils import stable_hash

if TYPE_CHECKING:
    from cvt.al.oracle import Revealed

EXPORTABLE_ROLES = {"train", "tune"}
FILL = (114, 114, 114)
SIDECAR = ".export.json"
TMP_SUFFIX = ".tmp.jpg"


class ExportError(ValueError):
    pass


def _stem(image_id: str) -> str:
    return image_id.replace("/", "__")


def _tmp(dst: Path) -> Path:
    return dst.with_name(dst.stem + TMP_SUFFIX)


def _link_or_copy(src: Path, dst: Path) -> None:
    tmp = _tmp(dst)
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _paint(src: Path, dst: Path, regions: list[list[float]]) -> None:
    with Image.open(src) as f:
        im = f.convert("RGB")
    draw = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in regions:
        draw.rectangle([x1, y1, x2, y2], fill=FILL)
    tmp = _tmp(dst)
    im.save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dst)


def _fingerprint(src: Path, regions: list[list[float]]) -> str:
    st = src.stat()
    return stable_hash([str(src), st.st_size, st.st_mtime_ns, regions])


def _read_sidecar(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sidecar(path: Path, data: dict[str, str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def manifest_labels(images: pd.DataFrame, boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(object labels, ignore regions) of `images` from the manifest boxes. Objects >= 50%
    inside an ignore region are dropped, the same rule evaluation applies."""
    b = boxes[boxes.image_id.isin(images.image_id)]
    ign = b.is_ignore.astype(bool)
    return drop_in_ignore(b[~ign], b), b[ign]


def export_labels(
    images: pd.DataFrame, labels: pd.DataFrame, ignore: pd.DataFrame, name: str, out_root: Path,
) -> Path:
    """Write out_root/name/{images,labels}/ and the list file out_root/name.txt."""
    roles = set(images.set_name.map(SET_ROLES))
    if not roles <= EXPORTABLE_ROLES:
        raise ExportError(
            f"refusing to export labels of {sorted(set(images.set_name))} (roles {sorted(roles)}): "
            "test and pool labels never leave the manifest"
        )
    stems = images.image_id.map(_stem)
    folded = stems.str.lower()
    if folded.duplicated().any():
        clash = sorted(set(images.image_id[folded.duplicated(keep=False)]))
        raise ExportError(f"image ids collide on a case-insensitive file system: {clash[:10]}")
    root = Path(out_root) / name
    img_dir, lab_dir = root / "images", root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    by_lab = {k: g for k, g in labels.groupby("image_id")}
    by_ign = {k: g for k, g in ignore.groupby("image_id")}
    regions_of = {
        iid: sorted([round(float(v), 2) for v in row]
                    for row in g[["x1", "y1", "x2", "y2"]].to_numpy(float))
        for iid, g in by_ign.items()
    }
    sidecar = root / SIDECAR
    old = _read_sidecar(sidecar)
    new = {r.image_id: _fingerprint(Path(r.path), regions_of.get(r.image_id, []))
           for r in images.itertuples(index=False)}
    stale = {iid for iid, fp in new.items()
             if old.get(iid) != fp or not (img_dir / f"{_stem(iid)}.jpg").exists()}
    if stale & old.keys():  # forget them first, so a crash mid-export cannot keep a stale match
        _write_sidecar(sidecar, {k: v for k, v in old.items() if k not in stale})
    wanted = {f"{st}.jpg" for st in stems}
    for t in img_dir.glob(f"*{TMP_SUFFIX}"):  # leftovers of an interrupted export
        if t.name not in wanted:
            t.unlink(missing_ok=True)
    paths = []
    for r in images.itertuples(index=False):
        stem = _stem(r.image_id)
        dst = img_dir / f"{stem}.jpg"
        if r.image_id in stale:
            regions = regions_of.get(r.image_id, [])
            if regions:
                _paint(Path(r.path), dst, regions)
            else:
                _link_or_copy(Path(r.path), dst)
        g = by_lab.get(r.image_id)
        lines = []
        if g is not None:
            for x1, y1, x2, y2, c in g[["x1", "y1", "x2", "y2", "label"]].to_numpy(float):
                cx, cy = (x1 + x2) / 2 / r.width, (y1 + y2) / 2 / r.height
                w, h = (x2 - x1) / r.width, (y2 - y1) / r.height
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lab_dir / f"{stem}.txt").write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
        paths.append(dst.resolve().as_posix())
    _write_sidecar(sidecar, new)
    lst = Path(out_root) / f"{name}.txt"
    lst.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    return lst


def export_revealed(revealed: Revealed, name: str, out_root: Path) -> Path:
    """Export pool frames the AL oracle revealed (the only export path for pool frames): they are
    relabelled `AL-revealed` (role train) and written like any training set."""
    images = revealed.images
    roles = set(images.set_name.map(SET_ROLES))
    if images.empty or roles != {"pool"}:
        raise ExportError(f"export_revealed takes revealed pool frames only, got sets "
                          f"{sorted(set(images.set_name))}")
    images = images.assign(set_name=REVEALED_SET)
    labels, ignore = manifest_labels(images, revealed.boxes)
    return export_labels(images, labels, ignore, name, out_root)


def write_dataset_yaml(path: Path, train: list[Path], val: list[Path]) -> Path:
    data = {
        "train": [Path(p).resolve().as_posix() for p in train],
        "val": [Path(p).resolve().as_posix() for p in val],
        "names": dict(enumerate(CLASSES)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")
    return path
