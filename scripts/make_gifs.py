"""Make the README's animated clips from rendered demo videos (follow demos, analytics overlays).

Each job takes a segment of an mp4, scales it to a width, samples it at a lower frame rate and
writes an animated WebP (default; plays inline on GitHub like a GIF at a fraction of the size) or a
GIF. The source videos are local renders (never committed); the clips go to docs/img.

    uv run --extra ml python scripts/make_gifs.py --all            # every job in JOBS below
    uv run --extra ml python scripts/make_gifs.py --job follow_DJI_0143_car --format gif
    uv run --extra ml python scripts/make_gifs.py --job my_clip --video path/to/render.mp4 --start 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT = Path("docs/img")
FOLLOW = Path("data/demo_out/follow/picked")
ANALYTICS = Path("data/demo_out/analytics")
WIDTH, FPS, SECONDS, QUALITY, GIF_COLORS = 960, 8.0, 8.0, 65, 128
# name: (source video, start in seconds)
JOBS: dict[str, tuple[Path, float]] = {
    "analytics_DJI_0143": (ANALYTICS / "DJI_0143_analytics.mp4", 180.0),
    "analytics_DJI_0075": (ANALYTICS / "DJI_0075_analytics.mp4", 240.0),
    "follow_DJI_0143_car": (FOLLOW / "DJI_0143_car.mp4", 4.0),
    "follow_DJI_0144_motorbike": (FOLLOW / "DJI_0144_motorbike.mp4", 4.0),
    "follow_DJI_0065_motorbike": (FOLLOW / "DJI_0065_motorbike.mp4", 4.0),
    "follow_50m_90d_morning_16_3_motorbike": (FOLLOW / "50m_90d_morning_16_3_motorbike.mp4", 1.0),
}


def read_segment(video: Path, start: float, seconds: float, width: int, fps: float) -> list:
    """RGB frames of `video` from `start` for `seconds`, one every 1/fps s, scaled to `width`.
    A clip that ends before `start + seconds` gives its last `seconds` instead."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = int(round(seconds * src_fps))
    first = min(int(round(start * src_fps)), max(0, n - span))
    last = min(n, first + span)
    step = src_fps / fps
    wanted = {int(round(first + k * step)) for k in range(int((last - first) / step))}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for i in range(first, last):
        ok, im = cap.read()
        if not ok:
            break
        if i in wanted:
            h = round(im.shape[0] * width / im.shape[1]) // 2 * 2
            im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"{video}: no frames from {start} s")
    return frames


def write_webp(frames: list, out: Path, fps: float, quality: int = QUALITY) -> int:
    images = [Image.fromarray(f) for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, quality=quality, method=4)
    return out.stat().st_size


def write_gif(frames: list, out: Path, fps: float, colors: int = GIF_COLORS) -> int:
    """One palette from a mosaic of sampled frames, no dithering (small files, no flicker)."""
    sample = frames[:: max(1, len(frames) // 8)][:8]
    palette = Image.fromarray(np.concatenate(sample, axis=0)).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)
              for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=True, disposal=1)
    return out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="every job in JOBS")
    ap.add_argument("--job", action="append", default=[],
                    help="a job name from JOBS (repeatable), or a new name together with --video")
    ap.add_argument("--video", type=Path, help="source video for a job not in JOBS")
    ap.add_argument("--start", type=float, default=0.0, help="start in seconds (with --video)")
    ap.add_argument("--format", choices=["webp", "gif"], default="webp")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--fps", type=float, default=FPS)
    a = ap.parse_args()
    jobs = dict(JOBS)
    if a.video is not None:
        if len(a.job) != 1:
            ap.error("--video needs exactly one --job name")
        jobs[a.job[0]] = (a.video, a.start)
    names = list(JOBS) if a.all else a.job
    unknown = [n for n in names if n not in jobs]
    if not names or unknown:
        ap.error(f"give --all or --job from: {', '.join(JOBS)} (or a new name with --video)")
    OUT.mkdir(parents=True, exist_ok=True)
    write = write_webp if a.format == "webp" else write_gif
    for name in names:
        video, start = jobs[name]
        out = OUT / f"{name}.{a.format}"
        size = write(read_segment(video, start, a.seconds, a.width, a.fps), out, a.fps)
        print(f"{out.name}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
