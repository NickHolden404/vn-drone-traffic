from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cvt.data.leakage import (
    REPORT_COLS,
    LeakageError,
    assert_no_leakage,
    leakage_report,
    load_or_compute_hashes,
    near_duplicates,
    phash_u64,
)
from tests.helpers import write_noise_jpg


def test_near_duplicates_on_raw_bits():
    a = np.array([0b0, 0xFF00], np.uint64)
    b = np.array([0b111, np.iinfo(np.uint64).max], np.uint64)
    pairs = near_duplicates(a, b, max_dist=6)
    assert pairs.tolist() == [[0, 0, 3]]


def test_phash_catches_recompressed_copy(tmp_path: Path):
    write_noise_jpg(tmp_path / "a.jpg", 1, size=(256, 192))
    write_noise_jpg(tmp_path / "b.jpg", 2, size=(256, 192))
    Image.open(tmp_path / "a.jpg").save(tmp_path / "a_copy.jpg", quality=60)
    ha, hb, hc = (phash_u64(tmp_path / n) for n in ("a.jpg", "b.jpg", "a_copy.jpg"))
    assert int(np.bitwise_count(np.uint64(ha) ^ np.uint64(hc))) <= 6
    assert int(np.bitwise_count(np.uint64(ha) ^ np.uint64(hb))) > 6


def _images(tmp_path: Path) -> pd.DataFrame:
    for i, name in enumerate(["tr1", "tr2", "te1", "pool1"]):
        write_noise_jpg(tmp_path / f"{name}.jpg", i, size=(256, 192))
    Image.open(tmp_path / "tr1.jpg").save(tmp_path / "te_dup.jpg", quality=70)
    rows = [
        ("tr1", "VD-train"), ("tr2", "VD-tune"), ("te1", "T-VN"),
        ("te_dup", "T-VN"), ("pool1", "UD-pool"),
    ]
    return pd.DataFrame(
        {"image_id": [r[0] for r in rows],
         "path": [str(tmp_path / f"{r[0]}.jpg") for r in rows],
         "set_name": [r[1] for r in rows]}
    )


def test_report_flags_train_test_pair_and_cache_reused(tmp_path: Path):
    images = _images(tmp_path)
    cache = tmp_path / "phash.parquet"
    hashes = load_or_compute_hashes(images, cache)
    assert cache.exists() and list(hashes.index) == list(images.image_id)
    rep = leakage_report(images, hashes, max_dist=6)
    assert rep[["train_id", "test_id"]].values.tolist() == [["tr1", "te_dup"]]
    with pytest.raises(LeakageError, match="1 training-side images"):
        assert_no_leakage(rep)
    again = load_or_compute_hashes(images, cache)
    assert (again == hashes).all()


def test_pool_pairs_are_errors(tmp_path: Path):
    images = _images(tmp_path)
    images.loc[images.image_id == "tr1", "set_name"] = "UD-pool"
    hashes = load_or_compute_hashes(images, tmp_path / "c.parquet")
    rep = leakage_report(images, hashes, max_dist=6)
    assert rep.train_set.tolist() == ["UD-pool"]
    with pytest.raises(LeakageError):
        assert_no_leakage(rep)


def test_duplicate_ids_do_not_poison_cache(tmp_path: Path):
    images = _images(tmp_path)
    cache = tmp_path / "phash.parquet"
    out = load_or_compute_hashes(pd.concat([images, images.iloc[:1]]), cache)
    assert len(out) == 6
    assert pd.read_parquet(cache).image_id.is_unique
    again = load_or_compute_hashes(images, cache)
    assert len(again) == 5 and list(again.index) == list(images.image_id)


def test_report_is_empty_frame_with_columns_when_clean(tmp_path: Path):
    images = _images(tmp_path)
    images = images[images.image_id != "te_dup"]
    hashes = load_or_compute_hashes(images, tmp_path / "c.parquet")
    rep = leakage_report(images, hashes, max_dist=6)
    assert rep.empty and list(rep.columns) == REPORT_COLS


def test_scene_summary_counts_pairs_per_scene_pair(tmp_path: Path):
    from cvt.data.leakage import scene_summary

    images = _images(tmp_path).assign(scene_id=["s_tr", "s_tr", "s_te", "s_te", "s_pool"])
    rep = leakage_report(images, load_or_compute_hashes(images, tmp_path / "c.parquet"))
    sc = scene_summary(rep, images)
    assert sc.to_dict("records") == [{
        "train_scene": "s_tr", "train_set": "VD-train", "test_scene": "s_te", "test_set": "T-VN",
        "pairs": 1, "train_images_flagged": 1, "train_scene_images": 2,
    }]
    assert scene_summary(rep.iloc[:0], images).empty


def test_hashing_is_chunked_logged_and_names_corrupt_images(tmp_path: Path):
    images = _images(tmp_path)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes((tmp_path / "tr1.jpg").read_bytes()[:200])  # truncated JPEG
    images = pd.concat([images, pd.DataFrame({"image_id": ["bad1"], "path": [str(bad)],
                                              "set_name": ["VD-train"]})], ignore_index=True)
    cache, msgs = tmp_path / "c.parquet", []
    with pytest.raises(OSError, match="bad1"):
        load_or_compute_hashes(images, cache, chunk=1, workers=1, log=msgs.append)
    assert set(pd.read_parquet(cache).image_id) == set(images.image_id[:-1])
    assert msgs[-1] == "pHash 5/6"
    # resumable: only the remaining images are hashed once the file is fixed
    write_noise_jpg(bad, 77, size=(256, 192))
    msgs.clear()
    out = load_or_compute_hashes(images, cache, chunk=500, workers=4, log=msgs.append)
    assert list(out.index) == list(images.image_id) and msgs == ["pHash 1/1"]
