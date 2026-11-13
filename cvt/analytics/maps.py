"""Ground-grid maps (density, speed, stationary time) and their heatmap rendering."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from cvt.analytics.events import VEHICLE_IDS

_ALPHA = 0.55
_BAR_MAX_FRAC = 0.12


def grid_maps(kin: pd.DataFrame, mpp: float, size: tuple[int, int], fps: float,
              cell_m: float, v_stop_kmh: float, min_samples: int = 5) -> dict[str, np.ndarray]:
    """Per-cell vehicle-seconds, mean speed (km/h) and stationary seconds over the reference frame."""
    w, h = size
    n_rows, n_cols = math.ceil(h * mpp / cell_m), math.ceil(w * mpp / cell_m)
    x = kin["xs_m"].to_numpy(float)
    y = kin["ys_m"].to_numpy(float)
    speed = kin["speed_kmh"].to_numpy(float)
    ok = (kin["label"].isin(VEHICLE_IDS).to_numpy() & np.isfinite(x) & np.isfinite(y)
          & (x >= 0) & (x < w * mpp) & (y >= 0) & (y < h * mpp))
    r = np.minimum((y[ok] // cell_m).astype(int), n_rows - 1)
    c = np.minimum((x[ok] // cell_m).astype(int), n_cols - 1)
    speed = speed[ok]
    shape = (n_rows, n_cols)

    density = np.zeros(shape)
    np.add.at(density, (r, c), 1.0 / fps)
    stationary = np.zeros(shape)
    still = speed < v_stop_kmh  # NaN compares False
    np.add.at(stationary, (r[still], c[still]), 1.0 / fps)

    fin = np.isfinite(speed)
    count = np.zeros(shape)
    total = np.zeros(shape)
    np.add.at(count, (r[fin], c[fin]), 1.0)
    np.add.at(total, (r[fin], c[fin]), speed[fin])
    mean = np.full(shape, np.nan)
    enough = count >= min_samples
    mean[enough] = total[enough] / count[enough]
    return {"density": density, "speed": mean, "stationary": stationary}


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    thick = max(1, round(scale * 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _cells_to_pixels(grid: np.ndarray, h: int, w: int, cell_px: float) -> np.ndarray:
    """(h, w) image of cell values: cell (r, c) covers rows [r, r + 1) * cell_px and columns
    [c, c + 1) * cell_px; partial cells are cropped at the frame edge, pixels past the grid NaN."""
    r = (np.arange(h) // cell_px).astype(int)
    c = (np.arange(w) // cell_px).astype(int)
    rin, cin = r < grid.shape[0], c < grid.shape[1]
    big = np.full((h, w), np.nan, np.float32)
    big[np.ix_(rin, cin)] = grid[np.ix_(r[rin], c[cin])]
    return big


def render_map(ref_bgr: np.ndarray, grid: np.ndarray, title: str, unit: str,
               vmax: float | None = None, *, cell_px: float | None = None) -> np.ndarray:
    """Blend `grid` over the frame: cell (r, c) covers the `cell_px` square block at
    (r * cell_px, c * cell_px); with `cell_px` None the grid is stretched to the frame."""
    out = ref_bgr.copy()
    h, w = out.shape[:2]
    if vmax is None:
        finite = grid[np.isfinite(grid)]
        vmax = float(finite.max()) if finite.size else 0.0
    if not vmax > 0:
        return out

    if cell_px is None:
        big = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        big = _cells_to_pixels(grid, h, w, cell_px)
    show = np.isfinite(big) & (big > 0)
    level = np.clip(np.nan_to_num(big) / vmax * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    blend = cv2.addWeighted(out, 1 - _ALPHA, heat, _ALPHA, 0)
    out[show] = blend[show]

    scale = max(0.4, h / 900.0)
    bar_w = max(2, min(int(w * 0.04), int(w * _BAR_MAX_FRAC)))
    bar_h = max(2, int(h * 0.5))
    y0 = (h - bar_h) // 2
    ramp = np.linspace(255, 0, bar_h).astype(np.uint8)[:, None]
    bar = cv2.applyColorMap(np.repeat(ramp, bar_w, axis=1), cv2.COLORMAP_TURBO)
    out[y0:y0 + bar_h, w - bar_w:] = bar

    def right_text(text: str, y: int) -> None:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        _label(out, text, (max(0, w - bar_w - 3 - tw), y), scale)

    right_text(f"{vmax:.3g} {unit}", y0 + int(12 * scale) + 2)
    right_text("0", y0 + bar_h)
    _label(out, title, (5, int(14 * scale / 0.4)), scale)
    return out
