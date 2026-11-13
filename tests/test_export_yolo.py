from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from cvt.data.export_yolo import ExportError, export_labels, manifest_labels, write_dataset_yaml
from cvt.data.manifest import BOX_COLS, build_manifest
from tests.helpers import make_uitdrone7, make_visdrone


def _manifest(tmp_path: Path):
    make_visdrone(tmp_path / "vd" / "train", n_images=2, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=1, seed0=100)
    make_uitdrone7(tmp_path / "uit")
    cfg = {"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd/train"), "val_dir": str(tmp_path / "vd/val"),
                         "rider_merge": True, "tune_frac": 0.4},
        "uitdrone7": {"images_dir": str(tmp_path / "uit/images"),
                      "labels_dir": str(tmp_path / "uit/annotations")},
    }}
    return build_manifest(cfg, exclusions=set())


def test_export_writes_normalised_labels_and_blacks_out_ignores(tmp_path: Path):
    images, boxes = _manifest(tmp_path)
    tr = images[images.set_name.isin(["VD-train", "VD-tune"])].head(1)
    labels, ignore = manifest_labels(tr, boxes)
    lst = export_labels(tr, labels, ignore, "vd", tmp_path / "yolo")
    img_path = Path(lst.read_text().split()[0])
    assert "/images/" in img_path.as_posix() and img_path.exists()
    lab = Path(img_path.as_posix().replace("/images/", "/labels/")).with_suffix(".txt")
    rows = [list(map(float, ln.split())) for ln in lab.read_text().splitlines()]
    # motor grown to include its rider: xyxy (10, 8, 30, 24) on 64x48
    assert rows[0] == pytest.approx([0, 20 / 64, 16 / 48, 20 / 64, 16 / 48], abs=1e-6)
    px = np.asarray(Image.open(img_path).convert("RGB"))
    assert np.abs(px[35, 55].astype(int) - 114).max() <= 6  # inside the tricycle ignore region


def test_export_refuses_test_and_pool_labels(tmp_path: Path):
    images, boxes = _manifest(tmp_path)
    for s in ("T-VD", "T-VN", "UD-pool"):
        sel = images[images.set_name == s]
        labels, ignore = manifest_labels(sel, boxes)
        with pytest.raises(ExportError, match="refusing"):
            export_labels(sel, labels, ignore, s, tmp_path / "yolo")


def test_dataset_yaml(tmp_path: Path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("x\n")
    b.write_text("y\n")
    p = write_dataset_yaml(tmp_path / "d.yaml", [a], [b])
    d = yaml.safe_load(p.read_text())
    assert d["train"] == [a.resolve().as_posix()] and d["val"] == [b.resolve().as_posix()]
    assert isinstance(d["names"], dict) and d["names"][0] == "motorbike"
    assert len(d["names"]) == 6


def _one(tmp_path: Path):
    images, boxes = _manifest(tmp_path)
    tr = images[images.set_name.isin(["VD-train", "VD-tune"])].head(1).reset_index(drop=True)
    return tr, boxes


def _boxes(iid: str, rows) -> pd.DataFrame:
    return pd.DataFrame([(iid, *r) for r in rows], columns=BOX_COLS).astype(
        {"label": "int64", "is_ignore": bool})


def _img(lst: Path) -> Path:
    return Path(lst.read_text().split()[0])


def test_reexport_rewrites_painted_image_only_when_ignore_regions_change(tmp_path: Path):
    tr, boxes = _one(tmp_path)
    out = tmp_path / "yolo"
    labels, ignore = manifest_labels(tr, boxes)
    dst = _img(export_labels(tr, labels, ignore, "vd", out))
    assert (out / "vd" / ".export.json").exists()
    m0 = dst.stat().st_mtime_ns
    export_labels(tr, labels, ignore, "vd", out)  # unchanged -> kept
    assert dst.stat().st_mtime_ns == m0
    # a new ignore region top-left -> the exported image is repainted there
    extra = _boxes(tr.image_id[0], [(0, 0, 20, 20, 0, True)])
    labels, ignore = manifest_labels(tr, pd.concat([boxes, extra], ignore_index=True))
    export_labels(tr, labels, ignore, "vd", out)
    px = np.asarray(Image.open(dst).convert("RGB")).astype(int)
    assert np.abs(px[2:18, 2:18] - 114).max() <= 6
    # back to the original regions -> repainted from the raw image again
    labels, ignore = manifest_labels(tr, boxes)
    export_labels(tr, labels, ignore, "vd", out)
    raw = np.asarray(Image.open(tr.path[0]).convert("RGB")).astype(int)
    px = np.asarray(Image.open(dst).convert("RGB")).astype(int)
    assert np.abs(px[2:18, 2:18] - raw[2:18, 2:18]).mean() < 6


def test_reexport_ignores_leftover_temp_file_and_never_touches_raw(tmp_path: Path):
    images, boxes = _manifest(tmp_path)
    tr = images[images.set_name.isin(["VD-train", "VD-tune"])].reset_index(drop=True)
    out = tmp_path / "yolo"
    labels, ignore = manifest_labels(tr, boxes)
    lst = export_labels(tr, labels, ignore, "vd", out)
    img_dir = out / "vd" / "images"
    raw_bytes = {p: Path(p).read_bytes() for p in tr.path}
    # a crash mid-write left temp files behind and one exported image missing
    for p in img_dir.glob("*.jpg"):
        p.with_name(p.stem + ".tmp.jpg").write_bytes(b"half")
    first = _img(lst)
    first.unlink()
    lst = export_labels(tr, labels, ignore, "vd", out)
    for p in lst.read_text().split():
        with Image.open(p) as im:
            im.load()
    assert first.exists() and not list(img_dir.glob("*.tmp.jpg"))
    assert {p: Path(p).read_bytes() for p in tr.path} == raw_bytes
    # no ignore regions: the image is linked/copied, still re-exported when its fingerprint changes
    labels, ignore = manifest_labels(tr, boxes[~boxes.is_ignore])
    lst = export_labels(tr, labels, ignore, "vd", out)
    for p, raw in zip(lst.read_text().split(), tr.path, strict=True):
        assert Path(p).read_bytes() == Path(raw).read_bytes()


def test_export_refuses_case_insensitive_collisions(tmp_path: Path):
    tr, boxes = _one(tmp_path)
    twin = tr.assign(image_id=tr.image_id.str.upper())
    both = pd.concat([tr, twin], ignore_index=True)
    labels, ignore = manifest_labels(both, boxes)
    with pytest.raises(ExportError, match="case-insensitive"):
        export_labels(both, labels, ignore, "vd", tmp_path / "yolo")


def test_manifest_labels_drop_objects_inside_ignore_regions(tmp_path: Path):
    tr, _ = _one(tmp_path)
    iid = tr.image_id[0]
    b = _boxes(iid, [
        (0, 0, 20, 20, 0, True),  # ignore region
        (5, 5, 15, 15, 1, False),  # fully inside -> dropped
        (14, 0, 34, 10, 2, False),  # 30% inside -> kept
        (40, 30, 50, 40, 3, False),  # outside -> kept
    ])
    labels, ignore = manifest_labels(tr, b)
    assert sorted(labels.label) == [2, 3] and len(ignore) == 1
    lst = export_labels(tr, labels, ignore, "vd", tmp_path / "yolo")
    lab = Path(_img(lst).as_posix().replace("/images/", "/labels/")).with_suffix(".txt")
    assert sorted(int(ln.split()[0]) for ln in lab.read_text().splitlines()) == [2, 3]
