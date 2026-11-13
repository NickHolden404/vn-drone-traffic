"""Demo video: detect, track, stabilize and count on a segment of a drone video, then render it.

Every frame of the segment is read (frame numbers 0..N-1 within the segment), detected in chunks,
tracked (BoT-SORT with precomputed GMC warps, or ByteTrack), stabilized to the scene's reference
frame and counted on the scene's lines exactly as `cvt.track.pipeline` does. The output mp4 shows
boxes coloured by class with track ids, the counting line(s) mapped back into each frame with the
inverse homography, and a panel of live per-class, per-direction counts.

The path-based GMC / stabilization functions need frame files, so the segment is always written as
JPEGs into a temporary directory (removed afterwards unless `keep_frames`); decoded frames are
also kept in memory when the segment fits `MEM_BUDGET`, else they are re-read from the JPEGs
(the JPEG round trip is the only difference between the two).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cvt.data.schema import CLASSES
from cvt.track.count import Line, crossings
from cvt.track.pipeline import MASK_SCORE, _anchors, _uses_gmc
from cvt.track.run import DET_COLS, gmc_warps, track_sequence
from cvt.track.stabilize import apply_h, reference_homographies

Log = Callable[[str], None]

MEM_BUDGET = 1_500_000_000  # bytes of decoded frames kept in memory
DET_CHUNK = 8  # frames per predict_arrays call
JPEG_QUALITY = 95
STILL_JPEG_QUALITY = 88  # the README still `<out stem>_still.jpg`
FOURCCS = ("avc1", "mp4v")  # H.264 first, then MPEG-4 part 2
# one fixed BGR colour per class id (motorbike, car, bus, truck, bicycle, pedestrian)
CLASS_BGR = [(0, 165, 255), (60, 200, 60), (230, 120, 30), (60, 60, 230), (200, 60, 200),
             (30, 220, 230)]
assert len(CLASS_BGR) == len(CLASSES)
LINE_BGR = (255, 255, 0)


def _quiet(msg: str) -> None:
    pass


class _Frames:
    """The segment's frames: JPEG paths plus an in-memory copy when small enough."""

    def __init__(self, paths: list[str], mem: list[np.ndarray] | None):
        self.paths, self.mem = paths, mem

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, i: int) -> np.ndarray:
        if self.mem is not None:
            return self.mem[i]
        im = cv2.imread(self.paths[i])
        if im is None:
            raise FileNotFoundError(f"cannot read frame image: {self.paths[i]}")
        return im


def _read_segment(
    video: Path, start_s: float, duration_s: float, work: Path, log: Log
) -> tuple[_Frames, float, tuple[int, int]]:
    """Write the segment's frames as JPEGs under `work`; (frames, fps, (width, height))."""
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{video}: unreadable frame rate ({fps})")
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        n_max = max(1, round(duration_s * fps))
        paths: list[str] = []
        mem: list[np.ndarray] | None = []
        size = (0, 0)
        while len(paths) < n_max:
            ok, im = cap.read()
            if not ok:
                break
            if not paths:
                size = (im.shape[1], im.shape[0])
                if n_max * im.nbytes > MEM_BUDGET:
                    mem = None
            elif (im.shape[1], im.shape[0]) != size:
                raise ValueError(f"{video}: frame size changed mid-segment")
            p = work / f"f{len(paths):06d}.jpg"
            if not cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise OSError(f"cannot write frame image: {p}")
            paths.append(str(p))
            if mem is not None:
                mem.append(im)
    finally:
        cap.release()
    if not paths:
        raise ValueError(f"{video}: no frames at start={start_s}s for {duration_s}s")
    log(f"read {len(paths)} frames at {fps:.2f} fps ({size[0]}x{size[1]}) from {video.name}")
    return _Frames(paths, mem), fps, size


def _detect(frames: _Frames, detector, log: Log) -> pd.DataFrame:
    rows = []
    for a in range(0, len(frames), DET_CHUNK):
        ims = [frames.get(i) for i in range(a, min(a + DET_CHUNK, len(frames)))]
        dets = detector.predict_arrays(ims)
        if len(dets) != len(ims):
            raise RuntimeError(f"detector returned {len(dets)} results for {len(ims)} images")
        for k, d in enumerate(dets):
            for (x1, y1, x2, y2), s, lab in zip(d.boxes, d.scores, d.labels, strict=True):
                rows.append((a + k, x1, y1, x2, y2, s, int(lab)))
        log(f"detected {min(a + DET_CHUNK, len(frames))}/{len(frames)}")
    df = pd.DataFrame(rows, columns=DET_COLS)
    return df.astype({c: "float64" for c in ("x1", "y1", "x2", "y2", "score")})


def _inverse_line(line: Line, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The line's end points (reference pixels) in a frame whose frame->reference map is H."""
    p = apply_h(np.array([line.p1, line.p2]), np.linalg.inv(H))
    return p[0], p[1]


def _live_counts(events: pd.DataFrame, line: str, frame: int) -> dict[int, dict[str, int]]:
    """{label: {"pos": n, "neg": n}} of the line's events with event frame <= `frame`."""
    ev = events[(events["line"] == line) & (events["frame"] <= frame)]
    out: dict[int, dict[str, int]] = {}
    for label, direction in zip(ev["label"], ev["direction"], strict=True):
        out.setdefault(int(label), {"pos": 0, "neg": 0})[direction] += 1
    return out


def _px(p: np.ndarray, s: float) -> tuple[int, int] | None:
    q = np.asarray(p, dtype=float) * s
    if not np.isfinite(q).all():
        return None
    q = np.clip(np.round(q), -1e5, 1e5).astype(int)
    return int(q[0]), int(q[1])


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, thick: int,
          color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_panel(img: np.ndarray, lines: list[Line], shown: dict[str, list[int]],
                counts: dict[str, dict[int, dict[str, int]]], scale: float) -> None:
    """Semi-transparent top-left panel: per line, one row per class that ever crosses it."""
    thick = max(1, round(scale * 2))
    row = int(34 * scale)
    rows: list[tuple[str, int | None, str]] = []  # (text, label or None for headers, counts text)
    for ln in lines:
        rows.append((ln.name, None, ""))
        if not shown[ln.name]:
            rows.append(("no crossings", None, ""))
        for label in shown[ln.name]:
            c = counts[ln.name].get(label, {"pos": 0, "neg": 0})
            rows.append((CLASSES[label], label, f"pos {c['pos']}  neg {c['neg']}"))
    pad = int(10 * scale)
    widths = [cv2.getTextSize(f"{a}    {b}", cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, thick)[0][0]
              for a, _, b in rows]
    pw, ph = max(widths) + 3 * pad + int(22 * scale), len(rows) * row + 2 * pad
    pw, ph = min(pw, img.shape[1]), min(ph, img.shape[0])
    roi = img[:ph, :pw]
    img[:ph, :pw] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    y = pad + row - int(10 * scale)
    for text, label, nums in rows:
        x = pad
        if label is not None:
            sw = int(14 * scale)
            cv2.rectangle(img, (x, y - sw), (x + sw, y), CLASS_BGR[label], -1)
            x += sw + int(8 * scale)
        _text(img, text, (x, y), scale * 0.8, thick, (255, 255, 255))
        if nums:
            _text(img, nums, (x + int(150 * scale), y), scale * 0.8, thick, (255, 255, 255))
        y += row


def _draw_frame(
    im: np.ndarray, size_out: tuple[int, int], tracks: pd.DataFrame,
    seg_pts: list[tuple[str, tuple[np.ndarray, np.ndarray]]], lines: list[Line],
    shown: dict[str, list[int]], counts: dict[str, dict[int, dict[str, int]]],
) -> np.ndarray:
    wo, ho = size_out
    s = wo / im.shape[1]
    img = cv2.resize(im, (wo, ho), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    scale = max(0.5, wo / 1920)
    thick = max(1, round(scale * 2))
    for _, (a, b) in seg_pts:
        pa, pb = _px(a, s), _px(b, s)
        if pa is not None and pb is not None:
            cv2.line(img, pa, pb, LINE_BGR, thick + 1, cv2.LINE_AA)
    for x1, y1, x2, y2, tid, label in zip(
        tracks["x1"], tracks["y1"], tracks["x2"], tracks["y2"], tracks["track_id"], tracks["label"],
        strict=True,
    ):
        color = CLASS_BGR[int(label)]
        p1, p2 = _px(np.array([x1, y1]), s), _px(np.array([x2, y2]), s)
        if p1 is None or p2 is None:
            continue
        cv2.rectangle(img, p1, p2, color, thick)
        _text(img, str(int(tid)), (p1[0], max(p1[1] - 4, int(14 * scale))), 0.5 * scale + 0.15,
              thick, color)
    _draw_panel(img, lines, shown, counts, scale)
    return img


def _open_writer(out: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 through Windows Media Foundation first (OpenCV's FFmpeg build has no H.264 encoder
    there and logs OpenH264 errors before falling back), then the FFmpeg FOURCCS."""
    if os.name == "nt":
        vw = cv2.VideoWriter(str(out), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, size)
        if vw.isOpened():
            return vw, "H264 (Media Foundation)"
        vw.release()
    for code in FOURCCS:
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*code), fps, size)
        if vw.isOpened():
            return vw, code
        vw.release()
    raise RuntimeError(f"cannot open an mp4 writer for {out} (tried {', '.join(FOURCCS)})")


def render_demo(
    video: Path, start_s: float, duration_s: float, detector, tracker_cfg: dict, scene: dict,
    out: Path, fps_out: float | None = None, width_out: int = 1920, *, keep_frames: bool = False,
    tmp_root: Path = Path("data/cache/demo"), log: Log = _quiet,
) -> dict:
    """Render the demo mp4 (+ 3 PNG stills `<out stem>_still{1,2,3}.png` and the middle one as
    the README JPEG `<out stem>_still.jpg`, quality STILL_JPEG_QUALITY) and return the counts.

    `detector` is anything with `predict_arrays(list of BGR arrays) -> list[Detections]`. `scene`
    is a `load_scene` dict; its lines are in the pixels of frame index `ref_frame` of the segment.
    `fps_out` None writes at the source video's frame rate. Returns {"counts": {line: {class
    name: {"pos": n, "neg": n}}}, "total", "frames", "fps" (source), "fps_out", "tracks",
    "unstabilized_frames", "codec", "out", "stills", "still"}.
    """
    video, out = Path(video), Path(out)
    if not hasattr(detector, "predict_arrays"):
        raise ValueError(
            f"{type(detector).__name__} has no predict_arrays: the demo supports only "
            "yolo/rtdetr models from configs/models.yaml (not tiled, fused or oracle ones)"
        )
    lines: list[Line] = list(scene["lines"])
    if not lines:
        raise ValueError("scene has no lines to count")
    if width_out < 16:
        raise ValueError(f"width_out must be >= 16, got {width_out}")
    if fps_out is not None and not (np.isfinite(fps_out) and fps_out > 0):
        raise ValueError(f"fps_out must be a positive frame rate or None, got {fps_out}")
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{video.stem}_", dir=tmp_root))
    try:
        frames, fps, (w, h) = _read_segment(video, start_s, duration_s, work, log)
        fps_out = fps if fps_out is None else float(fps_out)
        n = len(frames)
        ref = int(scene["ref_frame"])
        if not 0 <= ref < n:
            raise ValueError(f"scene ref_frame {ref} outside the segment's frames 0..{n - 1}")
        dets = _detect(frames, detector, log)
        warps = None
        if _uses_gmc(tracker_cfg):
            log(f"GMC {tracker_cfg['gmc_method']} over {n} frames")
            warps = gmc_warps(frames.paths, method=tracker_cfg["gmc_method"])
        fr = list(range(n))
        tracks = track_sequence(dets, fr, tracker_cfg, seq_id=str(scene["seq_id"]), fps=fps,
                                warps=warps)
        sure = dets[dets["score"] >= MASK_SCORE]
        by_frame = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(float)
                    for f, g in sure.groupby("frame")}
        log(f"stabilizing {n} frames to frame {ref}")
        Hs, valid = reference_homographies(
            frames.paths, boxes=[by_frame.get(f, np.zeros((0, 4))) for f in fr], ref=ref
        )
        pts = _anchors(tracks, Hs, valid, fr)
        parts = [crossings(pts, ln).assign(line=ln.name) for ln in lines]
        events = pd.concat(parts, ignore_index=True)
        shown = {ln.name: sorted(set(events.loc[events["line"] == ln.name, "label"].astype(int)))
                 for ln in lines}

        ho = max(2, round(h * width_out / w) // 2 * 2)
        wo = width_out // 2 * 2
        out.parent.mkdir(parents=True, exist_ok=True)
        writer, codec = _open_writer(out, fps_out, (wo, ho))
        still_idx = [(k * n) // 4 for k in (1, 2, 3)]
        stills = [out.parent / f"{out.stem}_still{k}.png" for k in (1, 2, 3)]
        still_jpg = out.parent / f"{out.stem}_still.jpg"
        by_track = {int(f): g for f, g in tracks.groupby("frame")}
        empty = tracks.iloc[:0]
        try:
            for i in range(n):
                segs = [(ln.name, _inverse_line(ln, Hs[i])) for ln in lines]
                counts = {ln.name: _live_counts(events, ln.name, i) for ln in lines}
                img = _draw_frame(frames.get(i), (wo, ho), by_track.get(i, empty), segs, lines,
                                  shown, counts)
                writer.write(img)
                for k, idx in enumerate(still_idx):
                    if idx == i and not cv2.imwrite(str(stills[k]), img):
                        raise OSError(f"cannot write still: {stills[k]}")
                if i == still_idx[1] and not cv2.imwrite(
                        str(still_jpg), img, [cv2.IMWRITE_JPEG_QUALITY, STILL_JPEG_QUALITY]):
                    raise OSError(f"cannot write still: {still_jpg}")
        finally:
            writer.release()
        last = {ln.name: _live_counts(events, ln.name, n - 1) for ln in lines}
        result_counts = {
            name: {CLASSES[label]: dict(c) for label, c in sorted(by_label.items())}
            for name, by_label in last.items()
        }
        log(f"wrote {out} ({n} frames, {codec})")
        return {
            "counts": result_counts,
            "total": int(len(events)),
            "frames": n,
            "fps": fps,
            "fps_out": fps_out,
            "tracks": int(tracks["track_id"].nunique()),
            "unstabilized_frames": int((~valid).sum()),
            "codec": codec,
            "out": str(out),
            "stills": [str(p) for p in stills],
            "still": str(still_jpg),
        }
    finally:
        if not keep_frames:
            shutil.rmtree(work, ignore_errors=True)
