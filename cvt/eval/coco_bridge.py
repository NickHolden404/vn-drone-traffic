"""Manifest/prediction frames -> pycocotools objects at a canonical 1920 px width."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from cvt.data.schema import CANONICAL_WIDTH, CLASSES


def _quiet_coco(dataset: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def drop_in_ignore(rows: pd.DataFrame, gt: pd.DataFrame, min_inside: float = 0.5) -> pd.DataFrame:
    """VisDrone convention (toolkit `dropObjectsInIgr`): discard boxes (detections or GT objects)
    with >= `min_inside` of their area inside an ignore region of the same image."""
    ign = gt[gt.is_ignore.astype(bool)]
    d = rows.reset_index(drop=True)
    if ign.empty or d.empty:
        return d
    regions = {k: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for k, g in ign.groupby("image_id")}
    keep = np.ones(len(d), bool)
    for iid, idx in d.groupby("image_id").indices.items():
        r = regions.get(iid)
        if r is None:
            continue
        b = d.loc[idx, ["x1", "y1", "x2", "y2"]].to_numpy(float)
        iw = np.minimum(b[:, None, 2], r[None, :, 2]) - np.maximum(b[:, None, 0], r[None, :, 0])
        ih = np.minimum(b[:, None, 3], r[None, :, 3]) - np.maximum(b[:, None, 1], r[None, :, 1])
        inside = (iw.clip(0) * ih.clip(0)) / np.maximum(
            ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[:, None], 1e-9
        )
        keep[idx[inside.max(axis=1) >= min_inside]] = False
    return d[keep].reset_index(drop=True)


def to_coco(
    images: pd.DataFrame, gt: pd.DataFrame, dt: pd.DataFrame, class_ids: Sequence[int]
) -> tuple[COCO, COCO, list[int]]:
    """COCO category id = our class id + 1. Ignore regions become per-class crowd boxes, and
    detections and GT objects >= 50% inside an ignore region are discarded first.
    Unknown image ids, duplicate images and out-of-range detection labels raise ValueError."""
    if not images.image_id.is_unique:
        raise ValueError("duplicate image ids in images")
    known = set(images.image_id)
    for name, df in (("gt", gt), ("dt", dt)):
        unknown = set(df.image_id) - known
        if unknown:
            raise ValueError(f"{name} rows reference {len(unknown)} unknown images, e.g. {sorted(unknown)[:3]}")
    bad = ~dt.label.isin(range(len(CLASSES)))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} detections with labels outside 0..{len(CLASSES) - 1}")
    dt = drop_in_ignore(dt, gt)
    is_ign = gt.is_ignore.astype(bool)
    gt = pd.concat([drop_in_ignore(gt[~is_ign], gt), gt[is_ign]], ignore_index=True)
    class_ids = sorted(class_ids)
    wanted = set(class_ids)
    ids = {iid: n + 1 for n, iid in enumerate(images.image_id)}
    scale = dict(zip(images.image_id, CANONICAL_WIDTH / images.width.astype(float), strict=True))
    img_entries = [
        {"id": ids[i], "width": round(w * scale[i]), "height": round(h * scale[i])}
        for i, w, h in zip(images.image_id, images.width, images.height, strict=True)
    ]
    cats = [{"id": c + 1, "name": CLASSES[c]} for c in class_ids]

    def bbox(r) -> list[float]:
        s = scale[r.image_id]
        return [r.x1 * s, r.y1 * s, (r.x2 - r.x1) * s, (r.y2 - r.y1) * s]

    gt_anns: list[dict] = []
    for r in gt[gt.image_id.isin(ids)].itertuples(index=False):
        bb = bbox(r)
        if r.is_ignore:
            targets, crowd = class_ids, 1
        elif r.label in wanted:
            targets, crowd = [int(r.label)], 0
        else:
            continue
        for c in targets:
            gt_anns.append(
                {"id": len(gt_anns) + 1, "image_id": ids[r.image_id], "category_id": c + 1,
                 "bbox": bb, "area": bb[2] * bb[3], "iscrowd": crowd}
            )

    dt_anns: list[dict] = []
    for r in dt[dt.image_id.isin(ids)].itertuples(index=False):
        if r.label not in wanted:
            continue
        bb = bbox(r)
        dt_anns.append(
            {"id": len(dt_anns) + 1, "image_id": ids[r.image_id], "category_id": int(r.label) + 1,
             "bbox": bb, "area": bb[2] * bb[3], "score": float(r.score), "iscrowd": 0}
        )

    coco_gt = _quiet_coco({"images": img_entries, "annotations": gt_anns, "categories": cats})
    coco_dt = _quiet_coco({"images": img_entries, "annotations": dt_anns, "categories": cats})
    return coco_gt, coco_dt, [ids[i] for i in images.image_id]
