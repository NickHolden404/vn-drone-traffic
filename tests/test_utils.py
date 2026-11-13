from pathlib import Path

import numpy as np
from PIL import Image

from cvt.config import load_yaml, read_exclusions
from cvt.utils import file_sha256, image_size, stable_hash


def test_image_size_reads_header(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(p)
    assert image_size(p) == (64, 48)


def test_file_sha256_is_stable(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert file_sha256(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_load_yaml_requires_mapping(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("x: 1\n", encoding="utf-8")
    assert load_yaml(p) == {"x": 1}
    p.write_text("- 1\n", encoding="utf-8")
    try:
        load_yaml(p)
    except ValueError as e:
        assert "mapping" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_read_exclusions_skips_comments_and_blanks(tmp_path: Path):
    p = tmp_path / "ex.txt"
    p.write_text("# comment\n\nvisdrone/train/a\n  uitdrone7/train/b  \n", encoding="utf-8")
    assert read_exclusions(p) == {"visdrone/train/a", "uitdrone7/train/b"}
    assert read_exclusions(tmp_path / "missing.txt") == set()


def test_git_commit_works_from_any_directory(tmp_path: Path, monkeypatch):
    from cvt.utils import git_commit

    monkeypatch.chdir(tmp_path)
    assert git_commit() != "nogit"


def test_write_parquet_atomic(tmp_path: Path):
    import pandas as pd

    from cvt.utils import write_parquet_atomic

    p = tmp_path / "sub" / "x.parquet"
    df = pd.DataFrame({"a": [1, 2]})
    write_parquet_atomic(df, p)
    write_parquet_atomic(df.assign(a=[3, 4]), p)  # replaces an existing file
    assert pd.read_parquet(p).a.tolist() == [3, 4]
    assert sorted(f.name for f in p.parent.iterdir()) == ["x.parquet"]
