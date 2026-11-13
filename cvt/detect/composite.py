"""Composite detectors: native-resolution tiling and multi-model fusion."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from cvt.detect.base import Detections, Detector
from cvt.detect.fusion import nms, wbf
from cvt.eval.predictions import PRED_COLS
from cvt.utils import stable_hash


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles (x0, y0, x1, y1) covering the image; the last tile is edge-aligned."""
    if tile <= 0 or not 0 <= overlap < 1:
        raise ValueError(f"need tile > 0 and 0 <= overlap < 1, got tile={tile}, overlap={overlap}")

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        step = max(1, int(round(tile * (1 - overlap))))
        return sorted(set(range(0, n - tile, step)) | {n - tile})

    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height) for x in starts(width)]


def shift_and_trim(
    d: Detections, x0: int, y0: int, x1: int, y1: int, width: int, height: int, edge_px: int = 2
) -> Detections:
    """Tile-local boxes -> image coordinates, dropping boxes that touch an inner tile edge."""
    b = d.boxes + np.array([x0, y0, x0, y0], np.float32)
    cut = np.zeros(len(b), bool)
    if x0 > 0:
        cut |= b[:, 0] <= x0 + edge_px
    if x1 < width:
        cut |= b[:, 2] >= x1 - edge_px
    if y0 > 0:
        cut |= b[:, 1] <= y0 + edge_px
    if y1 < height:
        cut |= b[:, 3] >= y1 - edge_px
    k = ~cut
    return Detections(b[k], d.scores[k], d.labels[k])


class TiledDetector:
    """Full-frame pass plus overlapping native-resolution tiles (small objects), merged per class
    by NMS. `base` must provide `predict_arrays(images, imgsz=None) -> list[Detections]`; the
    tiles go to `tile_base` when given (e.g. a static engine built at the tile size)."""

    def __init__(
        self, name: str, base, tile: int = 640, overlap: float = 0.2,
        tile_imgsz: int | None = None, nms_iou: float = 0.6, edge_px: int = 8,
        reader: Callable[[str], np.ndarray | None] | None = None, max_det: int = 1000,
        tile_base=None,
    ):
        self.name, self.base = name, base
        self.tile_base = base if tile_base is None else tile_base
        self.tile, self.overlap = tile, overlap
        self.tile_imgsz = tile_imgsz or tile
        self.nms_iou, self.edge_px, self.max_det = nms_iou, edge_px, max_det
        self._reader = reader
        self.cache_tag = stable_hash({
            "base": base.cache_tag, "tile": tile, "overlap": overlap,
            "tile_imgsz": self.tile_imgsz, "nms_iou": nms_iou, "edge_px": edge_px,
            "max_det": max_det, "algo": 1,
            # only when given, so existing single-detector tags (and caches) stay valid
            **({} if tile_base is None else {"tile_base": tile_base.cache_tag}),
        })[:12]

    def _read(self, path: str) -> np.ndarray:
        if self._reader is None:
            from ultralytics.utils.patches import imread

            self._reader = imread
        im = self._reader(path)
        if im is None:
            raise OSError(f"unreadable image: {path}")
        return im

    def predict_paths(self, paths: Sequence[str]) -> Iterator[Detections]:
        for p in paths:
            yield self._predict(self._read(p))

    def predict_arrays(self, ims: Sequence[np.ndarray]) -> list[Detections]:
        """BGR numpy images -> Detections (what predict_paths does after decoding)."""
        return [self._predict(im) for im in ims]

    def _predict(self, im: np.ndarray) -> Detections:
        h, w = im.shape[:2]
        parts = [self.base.predict_arrays([im])[0]]
        grid = tile_grid(w, h, self.tile, self.overlap)
        if len(grid) > 1:
            crops = [im[y0:y1, x0:x1] for x0, y0, x1, y1 in grid]
            tiles = self.tile_base.predict_arrays(crops, imgsz=self.tile_imgsz)
            for (x0, y0, x1, y1), d in zip(grid, tiles, strict=True):
                parts.append(shift_and_trim(d, x0, y0, x1, y1, w, h, self.edge_px))
        boxes = np.concatenate([d.boxes for d in parts]).reshape(-1, 4)
        scores = np.concatenate([d.scores for d in parts])
        labels = np.concatenate([d.labels for d in parts])
        keep = nms(boxes, scores, labels, self.nms_iou)[: self.max_det]
        return Detections(boxes[keep].astype(np.float32), scores[keep].astype(np.float32),
                          labels[keep].astype(np.int64))


class FusedDetector:
    """WBF over member detectors. During evaluation the fusion is computed from the members'
    cached predictions (see `cvt.eval.run.set_predictions`); `predict_paths` is the
    equivalent streaming form."""

    def __init__(self, name: str, members: Sequence[Detector], iou_thr: float = 0.55,
                 skip_thr: float = 0.0, max_det: int = 1000):
        self.name, self.members = name, list(members)
        self.iou_thr, self.skip_thr, self.max_det = iou_thr, skip_thr, max_det
        self.cache_tag = stable_hash({"members": [m.cache_tag for m in self.members],
                                      "iou": iou_thr, "skip": skip_thr, "max_det": max_det,
                                      "algo": 1})[:12]

    def predict_with_members(self, paths: Sequence[str]) -> Iterator[tuple[Detections, list[Detections]]]:
        streams = [m.predict_paths(paths) for m in self.members]
        for dets in zip(*streams, strict=True):
            yield _top(wbf(dets, self.iou_thr, self.skip_thr), self.max_det), list(dets)

    def predict_paths(self, paths: Sequence[str]) -> Iterator[Detections]:
        for fused, _ in self.predict_with_members(paths):
            yield fused


def _top(d: Detections, max_det: int) -> Detections:
    """Keep the `max_det` highest-scoring boxes."""
    if len(d.scores) <= max_det:
        return d
    k = np.argsort(-d.scores, kind="stable")[:max_det]
    return Detections(d.boxes[k], d.scores[k], d.labels[k])


def _dets(g: pd.DataFrame | None) -> Detections:
    if g is None or g.empty:
        return Detections.empty()
    return Detections(g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                      g.score.to_numpy(np.float32), g.label.to_numpy(np.int64))


def fuse_predictions(
    member_dts: Sequence[pd.DataFrame], images: pd.DataFrame, iou_thr: float, skip_thr: float,
    max_det: int = 1000,
) -> pd.DataFrame:
    """WBF per image over members' prediction frames (PRED_COLS), in `images` order; keeps the
    top-`max_det` fused boxes per image."""
    groups = [{k: g for k, g in dt.groupby("image_id")} for dt in member_dts]
    ids, parts = [], []
    for iid in images.image_id:
        f = _top(wbf([_dets(g.get(iid)) for g in groups], iou_thr, skip_thr), max_det)
        if len(f.boxes):
            ids.append(np.full(len(f.boxes), iid, dtype=object))
            parts.append(np.column_stack([f.boxes, f.scores, f.labels]).astype(np.float64))
    if not parts:
        return pd.DataFrame(columns=PRED_COLS)
    a = np.concatenate(parts)
    return pd.DataFrame({"image_id": np.concatenate(ids), "x1": a[:, 0], "y1": a[:, 1],
                         "x2": a[:, 2], "y2": a[:, 3], "score": a[:, 4],
                         "label": a[:, 5].astype(np.int64)})
