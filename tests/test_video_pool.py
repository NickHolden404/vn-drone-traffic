from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvt.data.adapters.video_pool import extract_pool, sample_video
from cvt.data.manifest import SET_ROLES, ManifestError, build_manifest, save_manifest
from cvt.eval.run import compare_models, evaluate_model, model_thresholds

cv2 = pytest.importorskip("cv2")  # opencv ships with the `ml` extra


def make_video(path: Path, seconds: int = 3, fps: int = 10, size=(64, 48)) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        vw.write(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8))
    vw.release()
    return path


def test_sample_video_every_second_and_resumable(tmp_path: Path):
    v = make_video(tmp_path / "DJI_0001.MP4")
    df = sample_video(v, tmp_path / "frames", every_s=1.0, source="uitadrone")
    assert len(df) == 3
    assert df.image_id.iloc[0] == "uitadrone/frames/DJI_0001_0000000"
    assert set(df.scene_id) == {"uitadrone/DJI_0001"} and (df.width == 64).all()
    first = Path(df.path.iloc[0]).stat().st_mtime_ns
    again = sample_video(v, tmp_path / "frames", every_s=1.0, source="uitadrone")
    pd.testing.assert_frame_equal(df, again)
    assert Path(df.path.iloc[0]).stat().st_mtime_ns == first  # existing frames are not rewritten


def test_pool_sets_in_manifest(tmp_path: Path):
    vids = [make_video(tmp_path / f"V{i}.MP4") for i in range(3)]
    index = tmp_path / "index.parquet"
    extract_pool(vids, tmp_path / "frames", 1.0, "uitadrone", index)
    cfg = {"sources": {"pool": {"index": str(index), "val_videos": ["V2"]}}}
    images, boxes = build_manifest(cfg, exclusions=set())
    counts = images.set_name.value_counts().to_dict()
    assert counts == {"P-train": 6, "P-val": 3} and boxes.empty
    assert SET_ROLES["P-train"] == "train" and SET_ROLES["P-val"] == "tune"
    with pytest.raises(ManifestError, match="val_videos"):
        build_manifest({"sources": {"pool": {"index": str(index), "val_videos": ["nope"]}}}, set())
    with pytest.raises(ManifestError, match="cvt pool extract"):
        build_manifest({"sources": {"pool": {"index": str(tmp_path / "missing.parquet")}}}, set())


def test_extract_pool_rejects_duplicate_stems(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    vids = [make_video(tmp_path / "a" / "DJI_0001.MP4"), make_video(tmp_path / "b" / "dji_0001.MP4")]
    with pytest.raises(ValueError, match="dji_0001"):
        extract_pool(vids, tmp_path / "frames", 1.0, "uitadrone", tmp_path / "index.parquet")
    assert not (tmp_path / "index.parquet").exists()


def test_sample_video_restores_missing_frame(tmp_path: Path):
    v = make_video(tmp_path / "DJI_0002.MP4")
    df = sample_video(v, tmp_path / "frames", every_s=1.0, source="uitadrone")
    lost = Path(df.path.iloc[1])
    lost.unlink()  # a half-extracted video
    again = sample_video(v, tmp_path / "frames", every_s=1.0, source="uitadrone")
    pd.testing.assert_frame_equal(df, again)
    assert lost.exists()


class _FakeCap:
    """cv2.VideoCapture stand-in: `n_grab` decodable frames, reported fps and frame count."""

    def __init__(self, fps: float, count: float, n_grab: int):
        self.fps, self.count, self.left = fps, count, n_grab

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count}.get(prop, 0.0)

    def grab(self):
        self.left -= 1
        return self.left >= 0

    def retrieve(self):
        return True, np.zeros((8, 8, 3), np.uint8)

    def release(self):
        pass


def test_sample_video_unknown_fps_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda p: _FakeCap(0.0, 0.0, 0))
    with pytest.raises(OSError, match="frame rate"):
        sample_video(tmp_path / "X.MP4", tmp_path / "frames", 1.0, "uitadrone")


def test_sample_video_warns_on_truncated_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda p: _FakeCap(10.0, 100.0, 20))
    with pytest.warns(UserWarning, match="decoded 20 of 100 frames"):
        df = sample_video(tmp_path / "X.MP4", tmp_path / "frames", 1.0, "uitadrone")
    assert len(df) == 2


def test_pool_val_video_fully_excluded_fails(tmp_path: Path):
    vids = [make_video(tmp_path / f"V{i}.MP4") for i in range(2)]
    index = tmp_path / "index.parquet"
    idx = extract_pool(vids, tmp_path / "frames", 1.0, "uitadrone", index)
    gone = set(idx.image_id[idx.video_id == "V1"])
    cfg = {"sources": {"pool": {"index": str(index), "val_videos": ["V1"]}}}
    with pytest.raises(ManifestError, match="V1"):
        build_manifest(cfg, exclusions=gone)
    images, _ = build_manifest(cfg, exclusions=set(list(gone)[:1]))  # partial exclusion is fine
    assert (images.set_name == "P-val").sum() == 2


def _pool_manifest(tmp_path: Path) -> dict:
    vids = [make_video(tmp_path / f"V{i}.MP4") for i in range(2)]
    index = tmp_path / "index.parquet"
    extract_pool(vids, tmp_path / "frames", 1.0, "uitadrone", index)
    images, boxes = build_manifest(
        {"sources": {"pool": {"index": str(index), "val_videos": ["V1"]}}}, set())
    save_manifest(images, boxes, tmp_path / "m")
    return dict(models_cfg={"oracle": {"arch": "oracle"}}, manifest_dir=tmp_path / "m",
                hash_cache=tmp_path / "ph.parquet", pred_cache=tmp_path / "preds")


def test_eval_and_thresholds_refuse_sets_without_ground_truth(tmp_path: Path):
    kw = _pool_manifest(tmp_path)
    with pytest.raises(ValueError, match="'P-val' has no ground truth"):
        model_thresholds("oracle", "P-val", 1280, 0.01, **kw)
    with pytest.raises(ValueError, match="'P-train' has no ground truth"):
        next(evaluate_model("oracle", ["P-train"], 1280, 10, 0, 0.01, **kw))
    with pytest.raises(ValueError, match="no ground truth"):
        next(compare_models("oracle", "oracle", ["P-val"], 1280, 640, 10, 0, 0.01, **kw))
    assert not (tmp_path / "preds").exists()  # refused before any inference
