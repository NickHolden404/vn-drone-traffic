"""Deployment grid (configs/deploy.yaml), TensorRT export wrapper and generated registry entries.

Engines are exported with Ultralytics (`YOLO(weights).export(format="engine", ...)`, static
shape, batch 1) to `data/engines/<model>@<imgsz>-<precision>.engine` with a JSON sidecar. INT8
calibration uses a seeded sample of the training list (VD-train + P-train) only.

A grid whose `tiled` block sets `tile_batch: N` (N > 1) gets batched tile engines instead: static
batch N at the tile size, `<model>@<tile>-<precision>-b<N>.engine`, used only as the tiles of its
tiled points (`<model>@<full>+tiles<tile>b<N>-<precision>`), so one call runs up to N of a frame's
tiles (Ultralytics zero-pads a shorter batch). Static, not dynamic: Ultralytics' dynamic profile
allows up to 2x the export size, so the engine would reserve memory for N images at 2x imgsz.
Without `tile_batch` every name and engine is as before."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from cvt.config import load_yaml
from cvt.data.export_yolo import _stem, write_dataset_yaml
from cvt.data.manifest import SET_ROLES, load_manifest
from cvt.utils import file_sha256

ENGINES = Path("data/engines")
MANIFEST = Path("data/manifest")
PRECISIONS = ("fp32", "fp16", "int8")
QUANTIZE = {"fp16": 16, "int8": 8}
# Export settings: static shape, batch 1, no onnxslim, 4 GiB workspace, GPU 0
EXPORT_ARGS = {"format": "engine", "batch": 1, "dynamic": False, "simplify": False,
               "workspace": 4, "device": 0}

Log = Callable[[str], None]


def _quiet(msg: str) -> None:
    pass


@dataclass(frozen=True)
class Point:
    model: str
    imgsz: int
    precision: str  # fp32 | fp16 | int8
    tiled: bool
    batch: int = 1  # engine: static batch; tiled point: its tile engine's batch

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.tiled and self.precision == "fp32":
            raise ValueError("tiled points are engine points (fp16/int8)")
        if self.batch < 1 or (self.batch > 1 and self.precision == "fp32"):
            raise ValueError(f"batch must be >= 1 and only engines batch, got {self.batch}")

    @property
    def is_engine(self) -> bool:
        return not self.tiled and self.precision in QUANTIZE


def grid_points(cfg: dict) -> list[Point]:
    """Full-frame engines (students x sizes x precisions), FP32 references, tiled points.
    `fp32_sizes: []` and `tiled: null` leave out the references / tiled points."""
    st, t = cfg["students"], cfg.get("tiled")
    pts = [Point(m, int(s), p, False) for m in st for s in cfg["sizes"] for p in cfg["precisions"]]
    pts += [Point(m, int(s), "fp32", False) for m in st for s in cfg.get("fp32_sizes") or []]
    if t:
        pts += [Point(m, int(t["full_imgsz"]), p, True, tile_batch(cfg))
                for m in st for p in cfg["precisions"]]
    return pts


def tile_batch(cfg: dict) -> int:
    """The tiles' engine batch: `tiled.tile_batch`, 1 (static batch-1 engines in `sizes`) without."""
    t = cfg.get("tiled")
    return int(t.get("tile_batch", 1)) if t else 1


def grid_tile(cfg: dict) -> int:
    """The grid's tile size (part of tiled points' registry names); 640 without tiled points."""
    t = cfg.get("tiled")
    return int(t["tile"]) if t else 640


def tile_engine_points(cfg: dict) -> list[Point]:
    """The batched tile engines of a `tile_batch` grid (none without it)."""
    b = tile_batch(cfg)
    if b == 1:
        return []
    return [Point(m, grid_tile(cfg), p, False, b) for m in cfg["students"] for p in cfg["precisions"]]


def engine_points(cfg: dict) -> list[Point]:
    return [p for p in grid_points(cfg) if p.is_engine] + tile_engine_points(cfg)


def engine_name(model: str, imgsz: int, precision: str, batch: int = 1) -> str:
    return f"{model}@{imgsz}-{precision}" + (f"-b{batch}" if batch > 1 else "")


def engine_path(model: str, imgsz: int, precision: str, root: Path = ENGINES,
                batch: int = 1) -> Path:
    return Path(root) / f"{engine_name(model, imgsz, precision, batch)}.engine"


def registry_name(point: Point, tile: int = 640) -> str:
    if point.tiled:
        b = f"b{point.batch}" if point.batch > 1 else ""
        return f"{point.model}@{point.imgsz}+tiles{tile}{b}-{point.precision}"
    return engine_name(point.model, point.imgsz, point.precision, point.batch)


# --- INT8 calibration ---------------------------------------------------------------------------

def _list_paths(data_yaml: Path, key: str) -> list[str]:
    files = load_yaml(data_yaml).get(key)
    if isinstance(files, str):
        files = [files]
    if not files:
        raise ValueError(f"{data_yaml}: no {key!r} lists")
    out: list[str] = []
    for f in files:
        if Path(f).suffix != ".txt":
            raise ValueError(f"{data_yaml}: {key} entry {f!r} is not a .txt image list")
        lines = Path(f).read_text(encoding="utf-8").splitlines()
        out += [ln.strip() for ln in lines if ln.strip()]
    return out


def calibration_list(cfg: dict, manifest_dir: Path = MANIFEST) -> list[str]:
    """Seeded sample of `calibration.n` images of the training list of `calibration.list`
    (list order kept). Exported images are named `<image_id with / -> __>.jpg` by
    `cvt yolo export`, so each maps back to its manifest image; raises unless every sampled
    image belongs to a set whose role is `train`."""
    c = cfg["calibration"]
    paths = _list_paths(Path(c["list"]), "train")
    n = int(c["n"])
    if n > len(paths):
        raise ValueError(f"calibration needs {n} images but {c['list']} lists only {len(paths)}")
    idx = np.sort(np.random.default_rng(int(c["seed"])).choice(len(paths), n, replace=False))
    sample = [paths[i] for i in idx]
    images, _, _ = load_manifest(manifest_dir)
    stems = images.image_id.map(_stem)
    set_of = dict(zip(stems, images.set_name, strict=True))
    dup = set(stems[stems.duplicated()])
    for p in sample:
        stem = Path(p).stem
        if stem not in set_of:
            raise ValueError(f"calibration image {p} is not in the manifest (stem {stem!r})")
        if stem in dup:
            raise ValueError(f"calibration image {p}: stem {stem!r} matches several manifest "
                             "images")
        role = SET_ROLES.get(set_of[stem])
        if role != "train":
            raise ValueError(f"calibration image {p} belongs to {set_of[stem]} (role {role!r}); "
                             "INT8 calibration uses training images only")
    return sample


def _calibration_text(cfg: dict, manifest_dir: Path) -> str:
    return "".join(f"{p}\n" for p in calibration_list(cfg, manifest_dir))


# --- export -------------------------------------------------------------------------------------

TENSORRT_MAJOR = 10
TENSORRT_PIN = "tensorrt-cu12==10.16.1.11"


def tensorrt_version() -> str:
    """Installed TensorRT version, from package metadata (does not import tensorrt); raises
    unless it is TensorRT 10."""
    for dist in ("tensorrt", "tensorrt-cu12", "tensorrt-cu13", "tensorrt-cu11"):
        try:
            v = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
        if v.split(".")[0] != str(TENSORRT_MAJOR):
            raise RuntimeError(
                f"TensorRT {TENSORRT_MAJOR} is required, {dist} {v} is installed: TensorRT 11 "
                "drops the INT8 calibrator path Ultralytics uses for INT8 engines; pin "
                f"{TENSORRT_PIN}")
        return v
    raise RuntimeError("TensorRT is not installed (no tensorrt package): install it before "
                       f"`cvt deploy export` ({TENSORRT_PIN})")


def export_args(point: Point, calib_yaml: Path | None = None) -> dict:
    args = {**EXPORT_ARGS, "quantize": QUANTIZE[point.precision], "imgsz": point.imgsz}
    if point.batch > 1:  # static batch; shorter batches are zero-padded by Ultralytics
        args |= {"batch": point.batch}
    if point.precision == "int8":
        args |= {"data": Path(calib_yaml).resolve().as_posix(), "fraction": 1.0}
    return args


def _without_data(args: dict | None) -> dict | None:
    """Export args minus `data` (an absolute path: moving the repo must not make an engine
    stale; the calibration content is compared by its sha256 instead)."""
    return None if args is None else {k: v for k, v in args.items() if k != "data"}


def _offline_guard() -> None:
    """The training job's in-process guard: Ultralytics downloads raise, fonts are found locally.
    INT8 calibration builds a dataset, whose check fetches Arial.ttf when Ultralytics' config dir
    lacks it (it did once on 2026-10-25)."""
    from cvt.train.job import no_network

    no_network()


def export_engine(
    point: Point, models_cfg: dict, cfg: dict, out_dir: Path = ENGINES, *,
    manifest_dir: Path = MANIFEST, log: Log = _quiet,
) -> Path:
    """Export one engine point; skip when the engine is up to date (same source sha256, export
    args other than the absolute calibration yaml path `data`, and for INT8 the same
    calibration list sha256), refuse a stale one. Returns the engine path."""
    if not point.is_engine:
        raise ValueError(f"{registry_name(point)} is not an engine point (fp16/int8, full frame)")
    if point.model not in models_cfg:
        raise KeyError(f"model {point.model!r} not in models config")
    src = models_cfg[point.model]["weights"]
    if Path(src).suffix != ".pt" or not Path(src).exists():
        raise FileNotFoundError(f"{point.model}: source weights not found (or not .pt): {src}")
    out_dir = Path(out_dir)
    name = engine_name(point.model, point.imgsz, point.precision, point.batch)
    dst = engine_path(point.model, point.imgsz, point.precision, out_dir, point.batch)
    sidecar = out_dir / f"{name}.json"
    calib_yaml = out_dir / f"{name}.calib.yaml" if point.precision == "int8" else None
    args, sha = export_args(point, calib_yaml), file_sha256(src)
    if dst.exists():
        if not sidecar.exists():
            raise RuntimeError(f"{dst} has no sidecar {sidecar}: delete the engine and re-export")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        same_calib = calib_yaml is None or meta.get("calibration_sha256") == hashlib.sha256(
            _calibration_text(cfg, manifest_dir).encode("utf-8")).hexdigest()
        if (meta.get("source_sha256") == sha and same_calib
                and _without_data(meta.get("args")) == _without_data(args)):
            log(f"{name}: up to date, skipped")
            return dst
        raise RuntimeError(f"{dst} is stale (source {src}, export args or INT8 calibration list "
                           "changed since it was built): delete it and re-export")
    trt = tensorrt_version()  # fail before a long export when TensorRT is missing
    calib_sha = None
    if calib_yaml is not None:
        calib_txt = out_dir / f"{name}.calib.txt"
        calib_txt.parent.mkdir(parents=True, exist_ok=True)
        calib_txt.write_text(_calibration_text(cfg, manifest_dir), encoding="utf-8", newline="\n")
        write_dataset_yaml(calib_yaml, [calib_txt], [calib_txt])  # Ultralytics calibrates on val
        calib_sha = hashlib.sha256(calib_txt.read_bytes()).hexdigest()
    _offline_guard()  # no silent downloads (fonts, AMP-check weights) during export/calibration
    import ultralytics
    from ultralytics import YOLO

    log(f"{name}: exporting {src} ({args})")
    built = Path(YOLO(src).export(**args))
    if built.suffix != ".engine" or not built.exists():
        raise RuntimeError(f"{name}: Ultralytics export returned {built}, not an engine file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.move(built, tmp)
    sidecar.write_text(json.dumps({
        "source_weights": src, "source_sha256": sha, "tensorrt": trt,
        "ultralytics": ultralytics.__version__, "args": args, "calibration_sha256": calib_sha,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, dst)  # the engine appears only once its sidecar is written
    log(f"{name}: -> {dst}")
    return dst


# --- generated registry -------------------------------------------------------------------------

def registry_entries(cfg: dict, models_cfg: dict, engines_dir: Path = ENGINES) -> dict[str, dict]:
    """One models.yaml entry per grid point (engines, FP32 references, tiled)."""
    t = cfg.get("tiled")
    tile, tb = grid_tile(cfg), tile_batch(cfg)
    full = int(t["full_imgsz"]) if t else 0
    sizes = [int(s) for s in cfg["sizes"]]
    needs = ((full, "full_imgsz"), (tile, "tile")) if tb == 1 else ((full, "full_imgsz"),)
    for need, what in needs if t else ():
        if need not in sizes:
            raise ValueError(f"tiled {what} {need} has no engine: not in sizes {sizes}")
    out: dict[str, dict] = {}
    for p in grid_points(cfg) + tile_engine_points(cfg):
        if p.model not in models_cfg:
            raise KeyError(f"student {p.model!r} not in models config")
        student = models_cfg[p.model]
        if p.tiled:
            entry = {"arch": "tiled",
                     "base": registry_name(Point(p.model, full, p.precision, False)),
                     "tile_base": registry_name(Point(p.model, tile, p.precision, False, tb)),
                     "tile": tile, "overlap": float(t["overlap"])}
        elif p.precision == "fp32":  # square letterbox, as the static engines use
            entry = {"arch": "yolo", "weights": student["weights"],
                     "names_map": student["names_map"], "imgsz": p.imgsz, "rect": False}
        else:
            entry = {"arch": "yolo",
                     "weights": engine_path(p.model, p.imgsz, p.precision, engines_dir,
                                            p.batch).as_posix(),
                     "names_map": student["names_map"], "imgsz": p.imgsz}
        out[registry_name(p, tile)] = entry
    clash = sorted(set(out) & set(models_cfg))
    if clash:
        raise ValueError(f"generated names already in the models config: {clash}")
    return out


def write_registry(
    cfg: dict, models_cfg: dict, out: Path = Path("configs/models_deploy.yaml"),
    engines_dir: Path = ENGINES, *, config: str | Path = "configs/deploy.yaml",
    models: str = "configs/models.yaml",
) -> Path:
    """Write the grid's entries to `out`; the header names the grid `config` and the registry
    yamls to use them with (`models`, the registry the students were read from, plus `out`)."""
    entries = registry_entries(cfg, models_cfg, engines_dir)
    out = Path(out)
    head = (f"# Generated by `cvt deploy register` from {Path(config).as_posix()}; do not edit.\n"
            f"# Use with: --models {models},{out.as_posix()}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + yaml.safe_dump(entries, sort_keys=False), encoding="utf-8", newline="\n")
    return out
