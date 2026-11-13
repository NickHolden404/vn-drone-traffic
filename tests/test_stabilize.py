import cv2
import numpy as np
import pandas as pd
import pytest

from cvt.track.stabilize import (
    _is_degenerate,
    apply_h,
    reference_homographies,
    stabilize_tracks,
)


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    """Seeded multi-scale blurred noise: dense, ORB-friendly, aperiodic grey texture."""
    rng = np.random.default_rng(seed)
    fine = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    coarse = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 5.0)
    img = 0.5 * (fine - fine.mean()) / fine.std() + (coarse - coarse.mean()) / coarse.std()
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _write(tmp_path, imgs) -> list[str]:
    paths = []
    for i, img in enumerate(imgs):
        p = tmp_path / f"f{i:03d}.png"
        assert cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def _similarity(tx: float, ty: float, deg: float, scale: float, centre) -> np.ndarray:
    """3x3 map base -> frame: rotate/scale about `centre`, then translate."""
    M = np.eye(3)
    M[:2] = cv2.getRotationMatrix2D(centre, deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def _grid(w: int, h: int, margin: int = 150) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(margin, w - margin, 10), np.linspace(margin, h - margin, 10))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def _max_err(H_est: np.ndarray, H_true: np.ndarray, pts: np.ndarray) -> float:
    return float(np.linalg.norm(apply_h(pts, H_est) - apply_h(pts, H_true), axis=1).max())


W, H_ = 1280, 720


def test_recovers_synthetic_homography(tmp_path):
    base = _texture(H_, W, seed=1)
    M = _similarity(15.0, -10.0, 1.0, 1.02, (W / 2, H_ / 2))  # ref -> frame 1
    warped = cv2.warpPerspective(base, M, (W, H_))
    H, valid = reference_homographies(_write(tmp_path, [base, warped]))
    assert H.shape == (2, 3, 3)
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(H[0], np.eye(3))
    assert _max_err(H[1], np.linalg.inv(M), _grid(W, H_)) < 1.0


def test_recovers_homography_after_downscale(tmp_path):
    base = _texture(H_, W, seed=2)
    M = _similarity(15.0, 8.0, -1.0, 0.98, (W / 2, H_ / 2))
    warped = cv2.warpPerspective(base, M, (W, H_))
    H, valid = reference_homographies(_write(tmp_path, [base, warped]), max_side=640)
    assert valid.tolist() == [True, True]
    # H is in full-resolution pixels even though matching ran at half size
    assert _max_err(H[1], np.linalg.inv(M), _grid(W, H_)) < 2.0


def _blobs(img: np.ndarray, rects, seed: int) -> np.ndarray:
    out = img.copy()
    for k, (x1, y1, x2, y2) in enumerate(rects):
        out[y1:y2, x1:x2] = _texture(y2 - y1, x2 - x1, seed + k)
    return out


def test_masks_exclude_moving_blobs(tmp_path):
    base = _texture(H_, W, seed=3)
    M = _similarity(12.0, 6.0, 0.5, 1.01, (W / 2, H_ / 2))
    rects0 = [(200, 150, 500, 400), (700, 300, 1000, 600)]
    # in frame 1 the blobs have moved with their own motion, not the camera's
    rects1 = [(260, 200, 560, 450), (640, 250, 940, 550)]
    f0 = _blobs(base, rects0, seed=10)
    f1 = _blobs(cv2.warpPerspective(base, M, (W, H_)), rects1, seed=20)
    paths = _write(tmp_path, [f0, f1])
    boxes = [np.array(rects0, dtype=float), np.array(rects1, dtype=float)]
    H, valid = reference_homographies(paths, boxes=boxes)
    assert valid.tolist() == [True, True]
    pts = _grid(W, H_)
    err_masked = _max_err(H[1], np.linalg.inv(M), pts)
    assert err_masked < 1.0
    # sanity: the blobs are strong enough that ignoring the boxes is worse
    H_no, _ = reference_homographies(paths)
    assert _max_err(H_no[1], np.linalg.inv(M), pts) > err_masked


def test_chaining_when_reference_match_fails(tmp_path):
    canvas = _texture(480, 1500, seed=4)
    shift = 250  # frame 3 (x 750..1390) has no overlap with the reference (x 0..640)
    frames = [canvas[:, k * shift : k * shift + 640].copy() for k in range(4)]
    paths = _write(tmp_path, frames)
    H, valid = reference_homographies(paths)
    assert valid.tolist() == [True] * 4
    for k in range(4):
        # frame pixel x sits at canvas x + k*shift == reference x
        T = np.array([[1, 0, k * shift], [0, 1, 0], [0, 0, 1.0]])
        # errors accumulate along the chain (partial overlaps extrapolate), so looser than 1 px
        assert _max_err(H[k], T, _grid(640, 480, margin=80)) < 1.5


def test_textureless_frame_flagged_invalid(tmp_path):
    base = _texture(H_, W, seed=5)
    flat = np.full_like(base, 128)
    M = _similarity(10.0, 5.0, 0.0, 1.0, (W / 2, H_ / 2))
    paths = _write(tmp_path, [base, flat, cv2.warpPerspective(base, M, (W, H_))])
    H, valid = reference_homographies(paths)
    assert valid.tolist() == [True, False, True]
    np.testing.assert_allclose(H[1], np.eye(3))  # last valid H (the reference)
    assert _max_err(H[2], np.linalg.inv(M), _grid(W, H_)) < 1.0


def test_invalid_frame_carries_last_valid_h(tmp_path):
    base = _texture(H_, W, seed=6)
    M = _similarity(10.0, 5.0, 0.0, 1.0, (W / 2, H_ / 2))
    warped = cv2.warpPerspective(base, M, (W, H_))
    flat = np.full_like(base, 90)
    H, valid = reference_homographies(_write(tmp_path, [base, warped, flat, flat]))
    assert valid.tolist() == [True, True, False, False]
    np.testing.assert_allclose(H[2], H[1])
    np.testing.assert_allclose(H[3], H[1])


def test_reference_not_first(tmp_path):
    base = _texture(H_, W, seed=7)
    M = _similarity(10.0, 5.0, 0.5, 1.0, (W / 2, H_ / 2))
    warped = cv2.warpPerspective(base, M, (W, H_))
    H, valid = reference_homographies(_write(tmp_path, [warped, base]), ref=1)
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(H[1], np.eye(3))
    assert _max_err(H[0], np.linalg.inv(M), _grid(W, H_)) < 1.0  # frame 0 = M(base)


def test_bad_arguments(tmp_path):
    paths = _write(tmp_path, [_texture(100, 100, 0)])
    with pytest.raises(ValueError, match="ref"):
        reference_homographies(paths, ref=3)
    with pytest.raises(ValueError, match="boxes"):
        reference_homographies(paths, boxes=[np.zeros((0, 4)), np.zeros((0, 4))])
    with pytest.raises(ValueError, match="frame_paths"):
        reference_homographies([])
    with pytest.raises(FileNotFoundError):
        reference_homographies([str(tmp_path / "missing.png")])


def test_is_degenerate():
    size = (640, 480)
    assert not _is_degenerate(np.eye(3), size)
    assert not _is_degenerate(_similarity(20, 10, 2.0, 1.1, (320, 240)), size)
    assert _is_degenerate(np.diag([3.0, 3.0, 1.0]), size)  # det 9
    assert _is_degenerate(np.diag([0.5, 0.5, 1.0]), size)  # det 0.25
    flip = np.diag([1.0, -1.0, 1.0])  # det -1: mirrored, corners reverse orientation
    assert _is_degenerate(flip, size)
    # strong perspective term pushes a corner through the line at infinity: corners self-intersect
    persp = np.array([[1, 0, 0], [0, 1, 0], [-0.004, 0, 1.0]])
    assert _is_degenerate(persp, size)
    assert _is_degenerate(np.full((3, 3), np.nan), size)


def test_apply_h_identity():
    pts = np.array([[0.0, 0.0], [10.5, 20.25], [-3.0, 7.0]])
    np.testing.assert_allclose(apply_h(pts, np.eye(3)), pts)
    assert apply_h(np.zeros((0, 2)), np.eye(3)).shape == (0, 2)


def test_apply_h_translation_and_projective():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    T = np.array([[1, 0, 5], [0, 1, -2], [0, 0, 1.0]])
    np.testing.assert_allclose(apply_h(pts, T), pts + [5, -2])
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2.0]])
    np.testing.assert_allclose(apply_h(pts, P), pts / 2)


def test_stabilize_tracks_uses_row_frame_index():
    frames = [10, 11, 12]  # actual frame numbers, not positions
    H = np.stack([np.eye(3)] * 3)
    H[1, 0, 2], H[2, 0, 2], H[2, 1, 2] = 5.0, 7.0, -4.0
    valid = np.array([True, True, False])
    tracks = pd.DataFrame(
        {
            "seq_id": "s", "frame": [12, 10, 11], "track_id": [1, 1, 1],
            "x1": [0.0, 0.0, 10.0], "y1": [0.0, 0.0, 10.0],
            "x2": [10.0, 10.0, 20.0], "y2": [20.0, 20.0, 30.0],
        }
    )  # fmt: skip
    out = stabilize_tracks(tracks, H, valid, frames)
    assert list(out["cx_ref"]) == [5.0 + 7.0, 5.0, 15.0 + 5.0]
    assert list(out["cy_ref"]) == [10.0 - 4.0, 10.0, 20.0]
    assert list(out["valid"]) == [False, True, True]
    assert "cx_ref" not in tracks.columns  # input untouched


def test_stabilize_tracks_errors_and_empty():
    H = np.stack([np.eye(3)] * 2)
    valid = np.array([True, True])
    tracks = pd.DataFrame({"frame": [1, 9], "x1": [0.0] * 2, "y1": [0.0] * 2,
                           "x2": [2.0] * 2, "y2": [2.0] * 2})  # fmt: skip
    with pytest.raises(ValueError, match="frame"):
        stabilize_tracks(tracks, H, valid, [1, 2])
    with pytest.raises(ValueError, match="frames"):
        stabilize_tracks(tracks, H, valid, [1])
    out = stabilize_tracks(tracks.iloc[:0], H, valid, [1, 2])
    assert len(out) == 0 and {"cx_ref", "cy_ref", "valid"} <= set(out.columns)
