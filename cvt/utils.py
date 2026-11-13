"""Small helpers shared across the package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

_REPO_DIR = Path(__file__).resolve().parent


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) read from the image header without decoding pixels."""
    with Image.open(path) as im:
        return im.size


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_parquet_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Write via a temp file + os.replace so a crash never leaves a truncated parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def stable_hash(obj: object) -> str:
    """Order-independent short hash of a JSON-serialisable config."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_commit() -> str:
    """Short HEAD sha, suffixed with -dirty when tracked files are modified."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_DIR,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_DIR,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return sha + ("-dirty" if dirty else "")
