"""Ultralytics YOLO / RT-DETR checkpoints mapped onto our 6-class schema."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator, Sequence
from itertools import batched
from pathlib import Path

import numpy as np

from cvt.data.classmaps import NAME_MAPS, RIDER_NAMES
from cvt.data.riders import merge_riders
from cvt.data.schema import CLASS_ID
from cvt.detect.base import Detections
from cvt.utils import file_sha256, stable_hash


def build_class_map(model_names: dict[int, str], names_key: str) -> np.ndarray:
    """Model class index -> our class id (-1 = drop)."""
    mapping = NAME_MAPS[names_key]
    out = np.full(max(model_names) + 1, -1, np.int64)
    for i, n in model_names.items():
        if n not in mapping:
            raise KeyError(f"model class {n!r} has no entry in names map {names_key!r}")
        target = mapping[n]
        out[i] = -1 if target is None else CLASS_ID[target]
    return out


def rider_ids(model_names: dict[int, str], names_key: str) -> tuple[int, tuple[int, ...]] | None:
    spec = RIDER_NAMES.get(names_key)
    if spec is None:
        return None
    inv = {n: i for i, n in model_names.items()}
    return inv[spec[0]], tuple(inv[v] for v in spec[1])


RIDER_MIN_VEHICLE_SCORE = 0.25  # low-confidence vehicle boxes may not absorb a predicted rider


def map_and_merge(
    xyxy: np.ndarray, scores: np.ndarray, cls: np.ndarray, class_map: np.ndarray,
    rider: tuple[int, tuple[int, ...]] | None,
) -> Detections:
    scores = np.asarray(scores, np.float32)
    if rider is not None:
        xyxy, cls, _, scores = merge_riders(
            xyxy, cls, rider[0], rider[1], scores=scores,
            min_vehicle_score=RIDER_MIN_VEHICLE_SCORE,
        )
    labels = class_map[np.asarray(cls, np.int64)]
    m = labels >= 0
    return Detections(
        np.asarray(xyxy, np.float32)[m], scores[m].astype(np.float32), labels[m].astype(np.int64)
    )


def detections_from_result(result, class_map: np.ndarray, rider) -> Detections:
    """One Ultralytics Results object -> Detections in our schema (float32 preserved)."""
    b = result.boxes
    if b is None or len(b) == 0:
        return Detections.empty()
    d = b.data.float().cpu().numpy()
    return map_and_merge(d[:, :4], d[:, 4], d[:, 5].astype(np.int64), class_map, rider)


def _rider_rule_defaults() -> dict:
    """merge_riders' default thresholds, so changing the rule invalidates cached predictions."""
    params = inspect.signature(merge_riders).parameters
    return {k: params[k].default for k in ("min_inside", "feet_min_inside")}


def engine_metadata(path: str | Path) -> dict:
    """The JSON header Ultralytics writes ahead of a TensorRT engine (4-byte little-endian length,
    then JSON), read without TensorRT; {} when the file has none."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        size = f.seek(0, 2)
        if not 0 < n <= size - 4:
            return {}
        f.seek(4)
        try:
            meta = json.loads(f.read(n))
        except ValueError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _check_engine(name: str, weights: str, imgsz: int) -> int:
    """An engine runs at its export size only (Ultralytics silently replaces a different predict
    imgsz with it) and at its export batch: a batch-N engine (batched tile engines) takes up to N
    images per call, Ultralytics zero-pads a shorter batch. Returns the batch to predict with;
    dynamic engines are refused (their memory and timing depend on the profile, not the input)."""
    meta = engine_metadata(weights)
    if "imgsz" not in meta:
        raise ValueError(f"{name}: {weights} has no Ultralytics metadata header")
    built = [int(v) for v in meta["imgsz"]]
    if built != [imgsz, imgsz]:
        raise ValueError(f"{name}: engine {weights} was built for imgsz {built[0]}"
                         f"{'' if built[0] == built[-1] else 'x' + str(built[-1])}, not {imgsz}")
    if meta.get("dynamic"):
        raise ValueError(f"{name}: engine {weights} has a dynamic shape; only static engines "
                         "are supported")
    return int(meta.get("batch", 1))


class UltralyticsDetector:
    def __init__(
        self, name: str, weights: str, arch: str, names_map: str, imgsz: int,
        rider_merge: bool, conf: float = 0.01, iou: float = 0.6, max_det: int = 1000,
        batch: int = 4, device: int | str = 0, half: bool = False, rect: bool | None = None,
    ):
        """`rect=False` forces Ultralytics' square letterbox to imgsz x imgsz for a .pt model
        (its predict default is the minimal stride-aligned rectangle; engines are always square);
        None keeps the default and leaves the cache tag unchanged."""
        import ultralytics  # heavy import, only when actually used
        from ultralytics import RTDETR, YOLO

        self.name = name
        # TensorRT engine: precision is baked in at export, input is static (imgsz, batch: 1, or
        # N for a batched tile engine)
        self.engine = Path(weights).suffix == ".engine"
        if self.engine:
            if arch != "yolo":
                raise ValueError(f"{name}: engines are supported for arch yolo only, not {arch}")
            batch = _check_engine(name, weights, imgsz)
            self.model = YOLO(weights, task="detect")
        else:
            self.model = (RTDETR if arch == "rtdetr" else YOLO)(weights)
        names = {int(k): v for k, v in self.model.names.items()}
        self.class_map = build_class_map(names, names_map)
        self.rider = rider_ids(names, names_map) if rider_merge else None
        self.batch = batch
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device,
                       verbose=False)
        if not self.engine:  # Ultralytics >= 8.4 renamed `half` to `quantize` (16/32)
            self.kw["quantize"] = 16 if half else 32
        if rect is not None:  # only when set, so existing cache tags stay valid
            self.kw["rect"] = bool(rect)
        self.cache_tag = stable_hash({
            "weights": file_sha256(weights), "arch": arch, "names": names,
            "names_map": NAME_MAPS[names_map],
            "rider": RIDER_NAMES.get(names_map) if rider_merge else None,
            "rider_min_vehicle_score": RIDER_MIN_VEHICLE_SCORE, "batch": batch,
            "rider_rule": _rider_rule_defaults(),
            "ultralytics": ultralytics.__version__,
            **{k: v for k, v in self.kw.items() if k != "verbose"},
            # weights sha256 above pins the engine bytes, imgsz in kw its input size
            **({"format": "engine"} if self.engine else {}),
        })[:12]

    def predict_arrays(self, ims: Sequence[np.ndarray], imgsz: int | None = None) -> list[Detections]:
        """BGR numpy images -> Detections, `batch` at a time; `imgsz` overrides (tiles)."""
        if self.engine and imgsz is not None and imgsz != self.kw["imgsz"]:
            raise ValueError(f"{self.name}: engine was built for imgsz {self.kw['imgsz']}, "
                             f"not {imgsz} (use a tile_base engine for tiles)")
        kw = self.kw if imgsz is None else {**self.kw, "imgsz": imgsz}
        out: list[Detections] = []
        for chunk in batched(ims, self.batch):
            results = self.model.predict(source=list(chunk), **kw)
            if len(results) != len(chunk):
                raise RuntimeError(f"{len(results)} results for {len(chunk)} images")
            out.extend(detections_from_result(r, self.class_map, self.rider) for r in results)
        return out

    def predict_paths(self, paths: Sequence[str]) -> Iterator[Detections]:
        """Decode `batch` images at a time (a list source would make Ultralytics decode every
        image up front) and yield one Detections per path, in input order."""
        from ultralytics.utils.patches import imread

        for chunk in batched(paths, self.batch):
            ims = []
            for p in chunk:
                im = imread(p)
                if im is None:
                    raise OSError(f"unreadable image: {p}")
                ims.append(im)
            yield from self.predict_arrays(ims)
