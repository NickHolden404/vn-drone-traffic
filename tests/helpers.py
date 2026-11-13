"""Tiny synthetic datasets mirroring the on-disk layouts of VisDrone-DET and UIT-Drone7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

W, H = 64, 48

VISDRONE_LINES = [
    "10,10,20,14,1,10,0,0",  # motor
    "14,8,8,10,1,2,0,0,",  # people riding the motor (trailing comma on purpose)
    "40,20,6,12,1,1,0,0",  # pedestrian
    "50,30,10,10,1,7,0,0",  # tricycle -> ignore region
    "0,40,10,8,0,0,0,0",  # ignored region
]

UIT_LINES = "1 0.5 0.5 0.25 0.5\n4 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.05 0.1\n"


def write_noise_jpg(path: Path, seed: int, size: tuple[int, int] = (W, H)) -> None:
    """Smooth random image (coarse noise upsampled) so pHash is stable under re-compression."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (max(size[1] // 8, 2), max(size[0] // 8, 2), 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize(size, Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def make_visdrone(root: Path, n_images: int, seed0: int) -> Path:
    """root/images/*.jpg + root/annotations/*.txt; every image is its own sequence."""
    for i in range(n_images):
        stem = f"{seed0 + i:07d}_00001_d_0000001"
        write_noise_jpg(root / "images" / f"{stem}.jpg", seed0 + i)
        ann = root / "annotations" / f"{stem}.txt"
        ann.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text("\n".join(VISDRONE_LINES) + "\n", encoding="utf-8")
    return root


def make_uitdrone7(root: Path, n_train: int = 3, n_test: int = 3, seed0: int = 1000) -> Path:
    """root/images/{split}/<CLIP>_<n>.jpg + root/annotations/{split}/<CLIP>_<n>.txt."""
    seed = seed0
    for split, n in (("train", n_train), ("test", n_test)):
        for i in range(1, n + 1):
            stem = f"CLIP{split}_{i}"
            write_noise_jpg(root / "images" / split / f"{stem}.jpg", seed)
            seed += 1
            lab = root / "annotations" / split / f"{stem}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text(UIT_LINES, encoding="utf-8")
    # an image with no label file must be skipped, like the 40 unlabeled DJI_0971 frames
    write_noise_jpg(root / "images" / "test" / "CLIPtest_99.jpg", seed)
    return root


VISDRONE_MOT_FRAMES = {
    1: [
        "1,1,10,10,20,14,1,10,0,0",  # motor, track 1
        "1,2,14,6,8,10,1,2,0,0",  # people, 60% inside the motor -> rider, merged away
        "1,3,40,20,6,12,1,1,0,0",  # pedestrian, track 3
        "1,4,0,40,10,8,0,0,0,0",  # ignored region
        "1,5,30,30,10,10,0,4,0,0,",  # score 0 car -> ignore (trailing comma on purpose)
    ],
    2: ["2,1,11,10,20,14,1,10,0,0", "2,3,41,20,6,12,1,1,0,0"],
}


def make_visdrone_mot(
    root: Path, seqs: tuple[str, ...] = ("uav0000013_00000_v",), seed0: int = 0,
    frames: dict[int, list[str]] | None = None,
) -> Path:
    """root/sequences/<seq>/<frame:07d>.jpg + root/annotations/<seq>.txt."""
    frames = VISDRONE_MOT_FRAMES if frames is None else frames
    for k, seq in enumerate(seqs):
        for f in frames:
            write_noise_jpg(root / "sequences" / seq / f"{f:07d}.jpg", seed0 + 10 * k + f)
        ann = root / "annotations" / f"{seq}.txt"
        ann.parent.mkdir(parents=True, exist_ok=True)
        ann.write_text("\n".join(line for ls in frames.values() for line in ls) + "\n")
    return root


# Real UAV-benchmark-MOTD_v1.0 formats: `_gt.txt` = frame,id,x,y,w,h,score,1,-1 (MOT GT, no
# class); `_gt_whole.txt` = frame,id,x,y,w,h,out_of_view,occlusion,category (1 car 2 truck 3 bus);
# `_gt_ignore.txt` = frame,id,x,y,w,h,1,-1,-1.
UAVDT_GT = ["1,1,10,10,20,14,1,1,-1", "1,2,40,20,12,12,1,1,-1", "2,1,11,10,20,14,1,1,-1",
            "2,3,5,30,16,10,1,1,-1",
            "2,4,30,2,8,6,0,1,-1"]  # score 0 -> not evaluated -> ignore region
UAVDT_WHOLE = ["1,1,10,10,20,14,1,1,1", "1,2,40,20,12,12,1,4,2",
               "1,5,50,0,14,10,2,1,1",  # only in _gt_whole (out of view) -> ignore region
               "2,1,11,10,20,14,1,1,1", "2,3,5,30,16,10,1,1,3", "2,4,30,2,8,6,1,1,1"]
UAVDT_IGNORE = ["1,7,0,40,10,8,1,-1,-1", "2,7,0,40,10,8,1,-1,-1"]


def uavdt_attributes(seqs: tuple[str, ...] = ("M0101",)) -> dict[str, list[str]]:
    """All 9 attribute keys: `seqs` are daylight, medium altitude, front view."""
    on = {"daylight", "medium_alt", "front_view"}
    keys = ["daylight", "night", "fog", "low_alt", "medium_alt", "high_alt", "front_view",
            "side_view", "bird_view"]
    return {k: list(seqs) if k in on else [] for k in keys}


def make_uavdt(root: Path, seqs: tuple[str, ...] = ("M0101",), n_frames: int = 2,
               seed0: int = 500) -> tuple[Path, Path]:
    """(images_dir, gt_dir) in the UAV-benchmark-M / UAV-benchmark-MOTD_v1.0 GT layouts."""
    images_dir, gt_dir = root / "images", root / "GT"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for k, seq in enumerate(seqs):
        for f in range(1, n_frames + 1):
            write_noise_jpg(images_dir / seq / f"img{f:06d}.jpg", seed0 + 10 * k + f)
        for suffix, lines in (("gt", UAVDT_GT), ("gt_whole", UAVDT_WHOLE),
                              ("gt_ignore", UAVDT_IGNORE)):
            (gt_dir / f"{seq}_{suffix}.txt").write_text("\n".join(lines) + "\n")
    return images_dir, gt_dir
