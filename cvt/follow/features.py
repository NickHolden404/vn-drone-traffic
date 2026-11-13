"""Appearance features for cached detections.

TensorRT engines expose no feature maps, so the same weights' `.pt` is run on the frames and
the stride-8 P3 map (the Detect head's first input) is ROI-pooled at the cached engine boxes of
vehicle classes. Features are L2-normalised and stored float16 per set, keyed by `det_row`: the
row index into the detections returned by `cached_detections` (cache row order).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cvt.data.manifest import load_mot
from cvt.data.schema import CLASS_ID
from cvt.eval.run import load_checked_manifest, set_hash_of
from cvt.track.pipeline import cached_detections, load_set
from cvt.utils import file_sha256

Log = Callable[[str], None]

VEHICLE_IDS = tuple(sorted(CLASS_ID[c] for c in ("car", "truck", "bus", "motorbike")))
LAYER = "P3"  # the Detect head's first input (stride 8)
ROI_SIZE = 7
PAD_VALUE = 114  # Ultralytics LetterBox fill


def letterbox_boxes(boxes: np.ndarray, w: int, h: int, imgsz: int) -> np.ndarray:
    """Image-coordinate xyxy -> the square `imgsz` letterbox (centred, as `LetterBox(auto=False)`)."""
    r = imgsz / max(w, h)
    px, py = (imgsz - round(w * r)) / 2, (imgsz - round(h * r)) / 2
    out = np.asarray(boxes, np.float32).reshape(-1, 4) * np.float32(r)
    return out + np.array([px, py, px, py], np.float32)


def pool_features(fmap, boxes_lb: np.ndarray, stride: int) -> np.ndarray:
    """(N, C) float32: ROI-align 7x7 at the boxes, global mean, L2 normalisation."""
    import torch
    from torchvision.ops import roi_align

    boxes = torch.as_tensor(np.asarray(boxes_lb, np.float32).reshape(-1, 4), device=fmap.device)
    pooled = roi_align(fmap.float(), [boxes], output_size=ROI_SIZE, spatial_scale=1.0 / stride,
                       aligned=True)
    v = pooled.mean(dim=(2, 3))
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return v.cpu().numpy().astype(np.float32)


def features_path(cache_dir: Path, det_model: str, det_tag: str, set_name: str) -> Path:
    return Path(cache_dir) / f"{det_model}_{det_tag}_{set_name}.npz"


def _read_meta(path: Path) -> dict | None:
    try:
        with np.load(path) as z:
            return json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError):
        return None


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(det_row int64, feat float16 (N, C))."""
    with np.load(path) as z:
        return z["det_row"].astype(np.int64), z["feat"]


def _save(path: Path, det_row: np.ndarray, feat: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, det_row=det_row.astype(np.int64), feat=feat.astype(np.float16),
                 meta=np.array(json.dumps(meta, sort_keys=True)))
    tmp.replace(path)


class _P3Extractor:
    """The `.pt` network with a forward pre-hook on its Detect head capturing the P3 map."""

    def __init__(self, weights: str, imgsz: int):
        import torch

        from cvt.train import job

        job.no_network()
        from ultralytics import YOLO

        self.torch, self.imgsz = torch, imgsz
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        net = YOLO(weights).model.to(self.device).eval()
        if self.half:
            net = net.half()
        self.net = net
        head = net.model[-1]
        self.stride = int(head.stride[0])
        self._fmap = None
        self._channels: int | None = None

        def grab(_module, args):
            self._fmap = args[0][0]
            self._channels = int(self._fmap.shape[1])

        head.register_forward_pre_hook(grab)

    def fmap(self, bgr: np.ndarray):
        import cv2

        h, w = bgr.shape[:2]
        r = self.imgsz / max(h, w)
        nw, nh = round(w * r), round(h * r)
        im = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
        canvas = np.full((self.imgsz, self.imgsz, 3), PAD_VALUE, np.uint8)
        y0, x0 = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = im
        t = self.torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
        t = t.to(self.device)[None]
        t = (t.half() if self.half else t.float()) / 255.0
        with self.torch.inference_mode():
            self.net(t)
        return self._fmap

    @property
    def channels(self) -> int:
        """P3 channel count (one forward pass on a blank canvas if no frame was run yet)."""
        if self._channels is None:
            self.fmap(np.full((self.imgsz, self.imgsz, 3), PAD_VALUE, np.uint8))
        return self._channels


def extract_set(
    extractor, s, dets: pd.DataFrame, min_score: float, log: Log,
) -> tuple[np.ndarray, np.ndarray]:
    """(det_row, feat float16 (N, C)) for vehicle detections with score >= min_score; each
    frame's pooled features are cast to float16 as they are collected (halves peak RAM). An empty
    selection gives (0, C) with C = `extractor.channels`."""
    import cv2

    keep = dets.label.isin(VEHICLE_IDS) & (dets.score >= min_score)
    sel = dets[keep]
    paths = s.frames.set_index(["seq_id", "frame"]).path
    rows, feats = [], []
    groups = sel.groupby(["seq_id", "frame"], sort=False)
    for i, (key, g) in enumerate(groups):
        img = cv2.imread(str(paths[key]))
        if img is None:
            raise FileNotFoundError(f"cannot read image {paths[key]}")
        h, w = img.shape[:2]
        lb = letterbox_boxes(g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32), w, h,
                             extractor.imgsz)
        feats.append(pool_features(extractor.fmap(img), lb, extractor.stride).astype(np.float16))
        rows.append(g.index.to_numpy(np.int64))
        if (i + 1) % 200 == 0:
            log(f"[{s.name}] {i + 1}/{len(groups)} frames")
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, extractor.channels), np.float16)
    return np.concatenate(rows), np.concatenate(feats)


def compute_features(
    weights_model: str, boxes_of: str, imgsz: int, sets: list[str], min_score: float, *,
    models_cfg: dict, manifest_dir: Path, hash_cache: Path, pred_cache: Path, cache_dir: Path,
    log: Log = lambda m: None,
) -> list[Path]:
    """Features for each set -> `features_path`; a set whose file has matching meta is skipped."""
    if weights_model not in models_cfg:
        raise KeyError(f"model {weights_model!r} not in models config ({sorted(models_cfg)})")
    weights = models_cfg[weights_model].get("weights")
    if not weights or not Path(weights).exists():
        raise FileNotFoundError(f"{weights_model}: weights not found: {weights}")
    sha = file_sha256(weights)[:12]
    images, boxes, mhash = load_checked_manifest(sets, manifest_dir, hash_cache, log)
    mot_gt = load_mot(manifest_dir)[0]
    extractor, out = None, []
    for name in sets:
        s = load_set(name, images, boxes, mot_gt)
        dets, tag = cached_detections(boxes_of, models_cfg, s, images, boxes, pred_cache, log)
        path = features_path(cache_dir, boxes_of, tag, name)
        meta = {"weights_model": weights_model, "weights_sha": sha, "det_model": boxes_of,
                "det_tag": tag, "imgsz": imgsz, "layer": LAYER, "min_score": min_score,
                "set_hash": set_hash_of(s.im), "manifest_hash": mhash}
        if path.exists() and _read_meta(path) == meta:
            log(f"[{name}] features up to date: {path}")
            out.append(path)
            continue
        if extractor is None:
            extractor = _P3Extractor(weights, imgsz)
        row, feat = extract_set(extractor, s, dets, min_score, log)
        _save(path, row, feat, meta)
        log(f"[{name}] {len(row)} features -> {path}")
        out.append(path)
    return out
