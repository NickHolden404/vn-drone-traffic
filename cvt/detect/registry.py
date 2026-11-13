"""Build a detector from configs/models.yaml (oracle, yolo, rtdetr, tiled, fused).

A yolo/rtdetr entry with `imgsz` (generated TensorRT engines, FP32 grid points) only builds at
that size, `rect: false` forces the square letterbox (FP32 grid points, to match the static
engines); a tiled entry may name a separate `tile_base` detector for its tiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvt.detect import ultralytics_det
from cvt.detect.base import Detector
from cvt.detect.composite import FusedDetector, TiledDetector
from cvt.detect.oracle import OracleDetector


def build_detector(
    name: str, models_cfg: dict, imgsz: int, conf: float, images: pd.DataFrame | None,
    boxes: pd.DataFrame | None,
) -> Detector:
    return _build(name, models_cfg, imgsz, conf, images, boxes, frozenset())


def _build(
    name: str, models_cfg: dict, imgsz: int, conf: float, images: pd.DataFrame | None,
    boxes: pd.DataFrame | None, seen: frozenset[str],
) -> Detector:
    if name not in models_cfg:
        raise KeyError(f"model {name!r} not in configs/models.yaml ({sorted(models_cfg)})")
    if name in seen:
        raise ValueError(f"cyclic model reference: {name}")
    seen = seen | {name}
    cfg = models_cfg[name]
    arch = cfg["arch"]
    if arch == "oracle":
        return OracleDetector(images, boxes)
    if arch == "tiled":
        for key in ("base", "tile_base"):
            ref = cfg.get(key)
            if ref is None:
                continue
            if ref not in models_cfg:
                raise KeyError(f"model {ref!r} not in configs/models.yaml ({sorted(models_cfg)})")
            if models_cfg[ref]["arch"] not in ("yolo", "rtdetr"):
                raise ValueError(f"{name}: tiled {key} {ref!r} must be a yolo/rtdetr model")
        tile = int(cfg.get("tile", 640))
        tile_imgsz = cfg.get("tile_imgsz")
        tile_base = None
        if "tile_base" in cfg:  # a second detector for the tiles, e.g. an engine built at 640
            tb = cfg["tile_base"]
            tile_imgsz = int(tile_imgsz or models_cfg[tb].get("imgsz") or tile)
            tile_base = _build(tb, models_cfg, tile_imgsz, conf, images, boxes, seen)
        return TiledDetector(
            name, _build(cfg["base"], models_cfg, imgsz, conf, images, boxes, seen),
            tile=tile, overlap=float(cfg.get("overlap", 0.2)),
            tile_imgsz=None if tile_imgsz is None else int(tile_imgsz),
            nms_iou=float(cfg.get("nms_iou", 0.6)), edge_px=int(cfg.get("edge_px", 8)),
            max_det=int(cfg.get("max_det", 1000)), tile_base=tile_base,
        )
    if arch == "fused":
        members = [_build(m, models_cfg, imgsz, conf, images, boxes, seen) for m in cfg["members"]]
        return FusedDetector(name, members, iou_thr=float(cfg.get("iou", 0.55)),
                             skip_thr=float(cfg.get("skip", 0.0)),
                             max_det=int(cfg.get("max_det", 1000)))
    if "imgsz" in cfg and int(cfg["imgsz"]) != imgsz:  # e.g. a static TensorRT engine
        raise ValueError(f"{name}: entry fixes imgsz {int(cfg['imgsz'])}, requested {imgsz} "
                         f"(pass --imgsz {int(cfg['imgsz'])})")
    weights = Path(cfg["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"{name}: weights not found: {weights}")
    return ultralytics_det.UltralyticsDetector(
        name=name, weights=str(weights), arch=arch, names_map=cfg["names_map"], imgsz=imgsz,
        rider_merge=bool(cfg.get("rider_merge", False)), conf=conf,
        **({"rect": bool(cfg["rect"])} if "rect" in cfg else {}),
    )
