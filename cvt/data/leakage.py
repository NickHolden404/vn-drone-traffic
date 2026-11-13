"""Perceptual-hash guard: no training-side image may near-duplicate a frozen test image.

Training-side means train, tune and pool: pool labels are revealed during active learning,
so a pool image that near-duplicates a test image leaks just like a train image does.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from cvt.data.manifest import SET_ROLES
from cvt.utils import write_parquet_atomic

# training/tuning-side roles: images whose labels shape a model or its settings; a near-duplicate
# of a test image among them is leakage. Test-vs-test overlap is not.
TRAINING_SIDE_ROLES = ("train", "tune", "pool")
REPORT_COLS = ["train_id", "train_set", "test_id", "test_set", "dist"]
SCENE_COLS = [
    "train_scene", "train_set", "test_scene", "test_set", "pairs", "train_images_flagged",
    "train_scene_images",
]


class LeakageError(RuntimeError):
    pass


def phash_u64(path: str | Path) -> int:
    with Image.open(path) as im:
        im.draft("L", (256, 256))  # fast JPEG DCT-domain downscale for 4K frames
        bits = imagehash.phash(im.convert("L")).hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _phash_named(image_id: str, path: str) -> int:
    try:
        return phash_u64(path)
    except Exception as e:
        raise OSError(f"cannot hash {image_id} ({path}): {e}") from e


def load_or_compute_hashes(
    images: pd.DataFrame, cache_file: Path, chunk: int = 500, workers: int = 8,
    log: Callable[[str], None] | None = None,
) -> pd.Series:
    """pHash per image_id, cached in parquet. Returns a Series aligned to `images.image_id`.

    Missing hashes are computed in chunks on a thread pool (PIL decoding releases the GIL);
    the cache is rewritten after every chunk, so an interrupted run resumes where it stopped.
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
    else:
        cached = pd.DataFrame({"image_id": pd.Series(dtype=str), "phash": pd.Series(dtype="uint64")})
    todo = images.loc[~images.image_id.isin(cached.image_id), ["image_id", "path"]].drop_duplicates("image_id")
    if len(todo):
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ids, paths = todo.image_id.to_numpy(), todo.path.astype(str).to_numpy()
        with ThreadPoolExecutor(max(1, workers)) as ex:
            for s in range(0, len(ids), chunk):
                cid, cpath = ids[s : s + chunk], paths[s : s + chunk]
                new = pd.DataFrame(
                    {"image_id": cid,
                     "phash": np.array(list(ex.map(_phash_named, cid, cpath)), dtype=np.uint64)}
                )
                cached = pd.concat([cached, new], ignore_index=True).drop_duplicates(
                    "image_id", keep="last"
                )
                write_parquet_atomic(cached, cache_file)
                if log is not None:
                    log(f"pHash {s + len(cid)}/{len(ids)}")
    return cached.set_index("image_id").phash.astype("uint64").loc[images.image_id.to_numpy()]


def near_duplicates(a: np.ndarray, b: np.ndarray, max_dist: int, chunk: int = 1024) -> np.ndarray:
    """All (i, j, hamming) with hamming(a[i], b[j]) <= max_dist."""
    a, b = np.asarray(a, np.uint64), np.asarray(b, np.uint64)
    out = []
    for s in range(0, len(a), chunk):
        d = np.bitwise_count(a[s : s + chunk, None] ^ b[None, :])
        i, j = np.nonzero(d <= max_dist)
        out.append(np.stack([i + s, j, d[i, j]], axis=1))
    return np.concatenate(out).astype(np.int64) if out else np.zeros((0, 3), np.int64)


def leakage_report(images: pd.DataFrame, hashes: pd.Series, max_dist: int = 6) -> pd.DataFrame:
    roles = images.set_name.map(SET_ROLES)
    test = images[roles == "test"]
    test_h = hashes.loc[test.image_id.to_numpy()].to_numpy(np.uint64)
    parts = []
    for role in TRAINING_SIDE_ROLES:
        src = images[roles == role]
        if src.empty or test.empty:
            continue
        pairs = near_duplicates(hashes.loc[src.image_id.to_numpy()].to_numpy(np.uint64), test_h, max_dist)
        if not len(pairs):
            continue
        s, t = pairs[:, 0], pairs[:, 1]
        parts.append(
            pd.DataFrame(
                {
                    "train_id": src.image_id.to_numpy()[s],
                    "train_set": src.set_name.to_numpy()[s],
                    "test_id": test.image_id.to_numpy()[t],
                    "test_set": test.set_name.to_numpy()[t],
                    "dist": pairs[:, 2],
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REPORT_COLS)


def scene_summary(report: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Near-duplicate pairs aggregated per (training scene, test scene), most pairs first."""
    if report.empty:
        return pd.DataFrame(columns=SCENE_COLS)
    scene = images.drop_duplicates("image_id").set_index("image_id").scene_id
    r = report.assign(train_scene=report.train_id.map(scene), test_scene=report.test_id.map(scene))
    out = (
        r.groupby(["train_scene", "train_set", "test_scene", "test_set"], sort=True)
        .agg(pairs=("dist", "size"), train_images_flagged=("train_id", "nunique"))
        .reset_index()
    )
    out["train_scene_images"] = out.train_scene.map(images.groupby("scene_id").image_id.nunique())
    return out.sort_values("pairs", ascending=False, kind="stable").reset_index(drop=True)[SCENE_COLS]


def assert_no_leakage(report: pd.DataFrame) -> None:
    if len(report):
        raise LeakageError(
            f"{report.train_id.nunique()} training-side images near-duplicate a test image "
            f"({len(report)} pairs). Run `cvt leakage check --write-exclusions configs/exclusions.txt`, "
            "review the pairs, and rebuild the manifest."
        )
