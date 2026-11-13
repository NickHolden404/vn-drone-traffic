"""The oracle: the only holder of the hidden UD-pool labels in AL code.

It reveals labels only for requested pool frames and appends every reveal (one row per frame:
timestamp, strategy, budget, image_id) to an append-only CSV log before handing them out. Other
`cvt.al` modules get pool metadata (`pool_ids`, `pool_frames`) from here, never from the
manifest."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cvt.data.manifest import load_manifest

POOL_SET = "UD-pool"
LOG_HEADER = ["timestamp", "strategy", "budget", "image_id"]


class OracleError(ValueError):
    pass


@dataclass(frozen=True)
class Revealed:
    """Revealed pool frames: manifest image rows and their boxes (ignore rows included)."""

    images: pd.DataFrame
    boxes: pd.DataFrame


def _clip_frame(image_id: str, scene_id: str) -> tuple[str, int]:
    """UIT-Drone7 pool frame `uitdrone7/train/<clip>_<n>` of scene `uitdrone7/<clip>`."""
    clip = scene_id.split("/", 1)[-1]
    m = re.fullmatch(re.escape(clip) + r"_(\d+)", image_id.rsplit("/", 1)[-1])
    if not m:
        raise OracleError(f"pool frame {image_id!r} (scene {scene_id!r}): no `<clip>_<frame "
                          f"number>` file name")
    return clip, int(m[1])


class Oracle:
    """Built by `oracle_from_manifest`; keeps the pool rows private."""

    def __init__(self, images: pd.DataFrame, boxes: pd.DataFrame, log_path: Path,
                 manifest_hash: str = "") -> None:
        pool = images[images.set_name == POOL_SET].sort_values("image_id").reset_index(drop=True)
        if pool.empty:
            raise OracleError(f"the manifest has no {POOL_SET} frames")
        cf = [_clip_frame(i, s) for i, s in zip(pool.image_id, pool.scene_id, strict=True)]
        frames = pd.DataFrame({"image_id": pool.image_id, "clip": [c for c, _ in cf],
                               "frame": pd.Series([f for _, f in cf], dtype="int64"),
                               "path": pool.path})
        dup = frames.duplicated(["clip", "frame"], keep=False)
        if dup.any():
            raise OracleError(f"pool frames share a clip and frame number: "
                              f"{sorted(frames.image_id[dup])[:10]}")
        self._images = pool
        self._boxes = boxes[boxes.image_id.isin(pool.image_id)].reset_index(drop=True)
        self._frames = frames
        self._log = Path(log_path)
        self._manifest_hash = manifest_hash

    @property
    def manifest_hash(self) -> str:
        """Hash of the manifest this oracle was built from ("" if built from bare tables)."""
        return self._manifest_hash

    def pool_ids(self) -> list[str]:
        return list(self._frames.image_id)

    def pool_frames(self) -> pd.DataFrame:
        """image_id, clip, frame, path of every pool frame (no labels)."""
        return self._frames.copy()

    def reveal(self, ids: Sequence[str], *, strategy: str, budget: int) -> Revealed:
        """Labels of `ids` (in that order); every id is logged before anything is returned."""
        ids = list(ids)
        if not ids:
            raise OracleError("nothing to reveal")
        if len(set(ids)) != len(ids):
            raise OracleError("duplicate ids in a reveal")
        pool = set(self._frames.image_id)
        unknown = [i for i in ids if i not in pool]
        if unknown:
            raise OracleError(f"{len(unknown)} ids are not {POOL_SET} frames: {unknown[:5]}")
        if not isinstance(strategy, str) or not strategy.strip():
            raise OracleError("strategy must be a non-empty name")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise OracleError(f"budget must be a positive int, got {budget!r}")
        if len(ids) > budget:
            raise OracleError(f"{len(ids)} ids exceed the budget {budget}")
        self._append_log(ids, strategy, budget)
        images = self._images.set_index("image_id").loc[ids].reset_index()
        boxes = self._boxes[self._boxes.image_id.isin(ids)].reset_index(drop=True)
        return Revealed(images=images, boxes=boxes)

    def _append_log(self, ids: list[str], strategy: str, budget: int) -> None:
        self._log.parent.mkdir(parents=True, exist_ok=True)
        new = not self._log.exists() or self._log.stat().st_size == 0
        if not new:
            with self._log.open(newline="", encoding="utf-8") as fh:
                head = next(csv.reader(fh), [])
            if head != LOG_HEADER:
                raise OracleError(f"{self._log} is not a reveal log (header {head}, expected "
                                  f"{LOG_HEADER}); refusing to append to it")
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        with self._log.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            if new:
                w.writerow(LOG_HEADER)
            w.writerows([stamp, strategy, budget, i] for i in ids)


def oracle_from_manifest(manifest_dir: Path, log_path: Path = Path("reports/al_reveals.csv")) -> Oracle:
    """The only function in `cvt.al` that reads manifest boxes."""
    images, boxes, mhash = load_manifest(Path(manifest_dir))
    return Oracle(images, boxes, log_path, mhash)
