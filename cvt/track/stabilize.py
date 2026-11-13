"""Stabilization: map every frame (and the track boxes in it) into one reference frame's pixels.

ORB keypoints (detected-object boxes masked out) are matched with BF Hamming + ratio test and a
RANSAC homography is fitted at a downscaled size, then converted back to full-resolution pixels.
Each frame is matched to the reference; if that fails it is chained through the last valid frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import pandas as pd

RATIO = 0.75
RANSAC_PX = 3.0
DET_RANGE = (0.5, 2.0)  # allowed determinant of the homography's top-left 2x2


def apply_h(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map (M, 2) points through a 3x3 homography (or (M, 3, 3), one per point) -> (M, 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64)
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = ph @ H.T if H.ndim == 2 else np.einsum("mij,mj->mi", H, ph)
    return out[:, :2] / out[:, 2:3]


def _is_degenerate(H: np.ndarray, size: tuple[int, int]) -> bool:
    """True if H is unusable for an image of `size` = (w, h).

    Degenerate means: non-finite or H[2,2] ~ 0; determinant of the top-left 2x2 (after
    normalising H[2,2] = 1) outside [0.5, 2.0]; a mapped frame corner on or behind the line at
    infinity; or the mapped frame corners self-intersect / reverse orientation.
    """
    H = np.asarray(H, dtype=np.float64)
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return True
    H = H / H[2, 2]
    det = np.linalg.det(H[:2, :2])
    if not DET_RANGE[0] <= det <= DET_RANGE[1]:
        return True
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    ph = np.concatenate([corners, np.ones((4, 1))], axis=1) @ H.T
    if (ph[:, 2] <= 1e-9).any():
        return True
    q = ph[:, :2] / ph[:, 2:3]
    edges = np.roll(q, -1, axis=0) - q
    cross = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    return not (cross > 0).all()  # a simple convex quad keeps the identity's orientation


class _Feats:
    """ORB keypoints/descriptors of one frame plus the image size they were computed at."""

    def __init__(self, kp: np.ndarray, desc: np.ndarray | None, size: tuple[int, int]):
        self.kp, self.desc, self.size = kp, desc, size


def _load_small(path: str, max_side: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read frame image: {path}")
    h, w = img.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, img.shape[1] / w, (w, h)


def _keypoint_mask(shape: tuple[int, int], boxes: np.ndarray | None, s: float) -> np.ndarray | None:
    if boxes is None or len(boxes) == 0:
        return None
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * s:
        px, py = max(4.0, 0.1 * (x2 - x1)), max(4.0, 0.1 * (y2 - y1))
        a, b = max(0, int(np.floor(x1 - px))), max(0, int(np.floor(y1 - py)))
        c, d = min(w, int(np.ceil(x2 + px))), min(h, int(np.ceil(y2 + py)))
        mask[b:d, a:c] = 0
    return mask


def _features(orb, img: np.ndarray, mask: np.ndarray | None) -> _Feats:
    kps, desc = orb.detectAndCompute(img, mask)
    kp = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    return _Feats(kp, desc, (img.shape[1], img.shape[0]))


def _match(bf, src: _Feats, dst: _Feats, min_inliers: int) -> np.ndarray | None:
    """Homography mapping `src` pixels -> `dst` pixels (small scale), or None if it fails."""
    if src.desc is None or dst.desc is None or len(src.desc) < 2 or len(dst.desc) < 2:
        return None
    pairs = bf.knnMatch(src.desc, dst.desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < max(4, min_inliers):
        return None
    a = src.kp[[m.queryIdx for m in good]]
    b = dst.kp[[m.trainIdx for m in good]]
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, RANSAC_PX)
    if H is None or int(inl.sum()) < min_inliers or _is_degenerate(H, src.size):
        return None
    return H / H[2, 2]


def reference_homographies(
    frame_paths: Sequence[str],
    boxes: Sequence[np.ndarray] | None = None,
    ref: int = 0,
    max_side: int = 1600,
    n_features: int = 4000,
    min_inliers: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame homographies H (N, 3, 3), frame pixels -> reference-frame pixels, and valid (N,).

    Frames are downscaled so the longer side is `max_side` (never upscaled); homographies are
    estimated there and returned in full-resolution pixels. `boxes[i]` is a (K, 4) xyxy array
    (full-res) whose padded regions are excluded from frame i's keypoints. Frame `ref` gets the
    identity and is always valid. Every other frame is matched to the reference; if that gives
    < `min_inliers` RANSAC inliers or a degenerate homography (see `_is_degenerate`: 2x2 determinant
    outside [0.5, 2.0] or self-intersecting mapped frame corners) it is matched to the last valid
    frame and composed. If both fail, valid[t] is False and H[t] is the last valid frame's H.
    """
    n = len(frame_paths)
    if n == 0:
        raise ValueError("frame_paths is empty")
    if not 0 <= ref < n:
        raise ValueError(f"ref={ref} out of range for {n} frames")
    if boxes is not None and len(boxes) != n:
        raise ValueError(f"boxes has {len(boxes)} entries but there are {n} frames")
    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def feats(i: int) -> tuple[_Feats, float]:
        img, s, _ = _load_small(frame_paths[i], max_side)
        mask = _keypoint_mask(img.shape, None if boxes is None else boxes[i], s)
        return _features(orb, img, mask), s

    ref_feats, ref_s = feats(ref)
    H_small = np.tile(np.eye(3), (n, 1, 1))
    scales = np.ones(n)
    scales[ref] = ref_s
    valid = np.zeros(n, bool)
    valid[ref] = True
    for order in (range(ref + 1, n), range(ref - 1, -1, -1)):
        last_feats, last_H = ref_feats, np.eye(3)  # last valid frame, its small-scale H
        for t in order:
            f, scales[t] = feats(t)
            H = _match(bf, f, ref_feats, min_inliers)
            if H is None and last_feats is not ref_feats:
                Hs = _match(bf, f, last_feats, min_inliers)
                if Hs is not None:
                    H = last_H @ Hs
                    H = None if _is_degenerate(H, f.size) else H / H[2, 2]
            if H is None:
                H_small[t] = last_H
                continue
            H_small[t], valid[t] = H, True
            last_feats, last_H = f, H
    H_full = np.empty_like(H_small)
    S_ref_inv = np.diag([1 / ref_s, 1 / ref_s, 1.0])
    for t in range(n):
        S = np.diag([scales[t], scales[t], 1.0])
        # an invalid frame carries the previous frame's H (small-scale of that frame's size);
        # frames share one resolution in practice, so this is exact there
        H_full[t] = S_ref_inv @ H_small[t] @ S
        H_full[t] /= H_full[t][2, 2]
    return H_full, valid


def stabilize_tracks(
    tracks: pd.DataFrame, H: np.ndarray, valid: np.ndarray, frames: Sequence[int]
) -> pd.DataFrame:
    """Add `cx_ref, cy_ref, valid`: box centres mapped through H[i], i = index of the row's frame.

    `frames` are actual frame numbers (H, valid are aligned to them); a row whose frame is not
    in `frames` raises.
    """
    H, valid = np.asarray(H), np.asarray(valid, dtype=bool)
    if not (len(H) == len(valid) == len(frames)):
        raise ValueError(
            f"H ({len(H)}), valid ({len(valid)}) and frames ({len(frames)}) must have equal length"
        )
    index = {int(f): i for i, f in enumerate(frames)}
    try:
        idx = np.array([index[int(f)] for f in tracks["frame"]], dtype=int)
    except KeyError as e:
        raise ValueError(f"track frame {e.args[0]} not in frames") from None
    out = tracks.copy()
    centres = np.stack(
        [
            (tracks["x1"] + tracks["x2"]).to_numpy() / 2,
            (tracks["y1"] + tracks["y2"]).to_numpy() / 2,
        ],
        axis=1,
    )
    ref = apply_h(centres, H[idx]) if len(idx) else np.zeros((0, 2))
    out["cx_ref"], out["cy_ref"] = ref[:, 0], ref[:, 1]
    out["valid"] = valid[idx]
    return out
