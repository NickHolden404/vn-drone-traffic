"""Day-1 check: are VisDrone riders separate `people` boxes sitting on motor/bicycle boxes?

For every `people` box, measures the largest fraction of its area inside a motor/bicycle box,
prints the distribution, and writes two mosaics for visual confirmation:
  reports/rider_check.jpg       people boxes >= 50% inside a rideable vehicle (merged today)
  reports/rider_check_band.jpg  people boxes 25-50% inside (merged when the feet are on the vehicle)
Each crop shows the person box in red and the best-overlapping vehicle box in green.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cvt.data.adapters.visdrone_det import _parse

ROOT = Path("data/external/visdrone/VisDrone2019-DET-train")
PEOPLE, RIDEABLE = 2, (3, 10)
CROP, COLS, ROWS = 96, 8, 6


def _crop(img: Image.Image, person: np.ndarray, vehicle: np.ndarray) -> Image.Image:
    x1, y1 = min(person[0], vehicle[0]), min(person[1], vehicle[1])
    x2, y2 = max(person[2], vehicle[2]), max(person[3], vehicle[3])
    pad = max(x2 - x1, y2 - y1)
    box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    c = img.crop(box)
    d = ImageDraw.Draw(c)
    for b, color in ((vehicle, "lime"), (person, "red")):
        d.rectangle([b[0] - box[0], b[1] - box[1], b[2] - box[0], b[3] - box[1]], outline=color)
    return c.resize((CROP, CROP))


def _mosaic(crops: list[Image.Image], path: str) -> None:
    m = Image.new("RGB", (CROP * COLS, CROP * ROWS))
    for i, c in enumerate(crops[: COLS * ROWS]):
        m.paste(c, ((i % COLS) * CROP, (i // COLS) * CROP))
    m.save(path)


def main() -> None:
    fracs: list[float] = []
    merged_crops: list[Image.Image] = []
    band_crops: list[Image.Image] = []
    for ann in sorted((ROOT / "annotations").glob("*.txt")):
        a = _parse(ann)
        xyxy = np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], 1).astype(float)
        people = xyxy[a[:, 5] == PEOPLE]
        veh = xyxy[np.isin(a[:, 5], RIDEABLE)]
        if not len(people):
            continue
        if not len(veh):
            fracs.extend([0.0] * len(people))
            continue
        iw = np.minimum(people[:, None, 2], veh[None, :, 2]) - np.maximum(people[:, None, 0], veh[None, :, 0])
        ih = np.minimum(people[:, None, 3], veh[None, :, 3]) - np.maximum(people[:, None, 1], veh[None, :, 1])
        frac = (iw.clip(0) * ih.clip(0)) / np.maximum(
            ((people[:, 2] - people[:, 0]) * (people[:, 3] - people[:, 1]))[:, None], 1e-9
        )
        best, best_frac = frac.argmax(1), frac.max(1)
        fracs.extend(best_frac.tolist())
        img = None
        for target, lo, hi in ((merged_crops, 0.5, 1.01), (band_crops, 0.25, 0.5)):
            idx = np.flatnonzero((best_frac >= lo) & (best_frac < hi))
            if len(idx) and len(target) < COLS * ROWS:
                img = img or Image.open(ROOT / "images" / f"{ann.stem}.jpg")
                i = idx[0]
                target.append(_crop(img, people[i], veh[best[i]]))

    f = np.array(fracs)
    print(f"people boxes: {len(f)}")
    for lo, hi in ((0, 1e-9), (1e-9, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        n = int(((f >= lo) & (f < hi)).sum())
        print(f"  inside fraction [{lo:.2f}, {min(hi, 1):.2f}): {n:6d} ({100 * n / len(f):.1f}%)")
    Path("reports").mkdir(exist_ok=True)
    _mosaic(merged_crops, "reports/rider_check.jpg")
    _mosaic(band_crops, "reports/rider_check_band.jpg")
    print("wrote reports/rider_check.jpg and reports/rider_check_band.jpg")


if __name__ == "__main__":
    main()
