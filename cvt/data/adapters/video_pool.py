"""Unlabelled drone video -> sampled JPEG frames + an index parquet (the pseudo-label pool P).

No within-video deduplication: for a hovering drone, frames seconds apart share the static
background (near-identical pHash) while the traffic differs, which is exactly what we want."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from cvt.utils import image_size, write_parquet_atomic

INDEX_COLS = ["image_id", "path", "width", "height", "source", "scene_id", "split", "video_id", "t_s"]


def sample_video(video: Path, frames_dir: Path, every_s: float, source: str) -> pd.DataFrame:
    """One JPEG (quality 95) every `every_s` seconds; frames already on disk are reused.

    `t_s` is frame index / fps, which assumes a constant frame rate. Warns when far fewer
    frames decode than the container reports (truncated or corrupt file)."""
    import cv2

    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        raise OSError(f"{video}: unknown frame rate")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps * every_s))
    vid = video.stem
    out_dir = Path(frames_dir) / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, i = [], 0
    try:
        while cap.grab():
            if i % step == 0:
                path = out_dir / f"{vid}_{i:07d}.jpg"
                if path.exists():
                    w, h = image_size(path)
                else:
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise OSError(f"{video}: cannot decode frame {i}")
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise OSError(f"{video}: cannot encode frame {i}")
                    tmp = path.with_name(path.name + ".tmp")
                    tmp.write_bytes(buf.tobytes())
                    os.replace(tmp, path)
                    h, w = frame.shape[:2]
                rows.append((f"{source}/frames/{path.stem}", str(path), int(w), int(h), source,
                             f"{source}/{vid}", "frames", vid, i / fps))
            i += 1
    finally:
        cap.release()
    if n > 0 and i < 0.95 * n:
        warnings.warn(f"{video}: decoded {i} of {n} frames (truncated or corrupt?)", stacklevel=2)
    return pd.DataFrame(rows, columns=INDEX_COLS)


def extract_pool(
    videos: Sequence[Path], frames_dir: Path, every_s: float, source: str, index_file: Path,
    log: Callable[[str], None] = lambda m: None,
) -> pd.DataFrame:
    stems = [Path(v).stem.lower() for v in videos]
    dupes = {s for s in stems if stems.count(s) > 1}
    if dupes:
        raise ValueError(f"pool videos with the same file stem: {sorted(dupes)}")
    parts = []
    for v in videos:
        if not Path(v).exists():
            raise FileNotFoundError(f"pool video not found: {v}")
        df = sample_video(Path(v), frames_dir, every_s, source)
        log(f"{Path(v).name}: {len(df)} frames")
        parts.append(df)
    index = pd.concat(parts, ignore_index=True)
    if not index.image_id.is_unique:
        raise ValueError("duplicate image ids in the pool index")
    write_parquet_atomic(index, index_file)
    return index
