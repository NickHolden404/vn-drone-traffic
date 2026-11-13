"""CLI behaviour on tiny synthetic data: error paths, leakage handling, eval options."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import cvt.eval.run as run_mod
from cvt.cli import app
from cvt.config import load_yaml, read_exclusions
from cvt.data.manifest import build_manifest, load_manifest, save_manifest
from cvt.detect.base import Detections
from cvt.eval.predictions import PRED_COLS
from cvt.eval.run import evaluate_model
from cvt.pseudo.label import save_pseudo
from tests.helpers import make_uitdrone7, make_visdrone

VAL_STEM = "0000100_00001_d_0000001"  # first T-VD fixture image
DUP_STEM = "0000050_00001_d_0000001"  # its copy on the training side


def _setup(tmp_path: Path, *, dup: bool = False) -> dict[str, Path]:
    make_visdrone(tmp_path / "vd" / "train", n_images=4, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=3, seed0=100)
    make_uitdrone7(tmp_path / "uit")
    if dup:  # a training image that is an exact copy of a test image
        tr, va = tmp_path / "vd" / "train", tmp_path / "vd" / "val"
        shutil.copy(va / "images" / f"{VAL_STEM}.jpg", tr / "images" / f"{DUP_STEM}.jpg")
        shutil.copy(va / "annotations" / f"{VAL_STEM}.txt", tr / "annotations" / f"{DUP_STEM}.txt")
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text(yaml.safe_dump({"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd/train"), "val_dir": str(tmp_path / "vd/val"),
                         "rider_merge": True, "tune_frac": 0.1},
        "uitdrone7": {"images_dir": str(tmp_path / "uit/images"),
                      "labels_dir": str(tmp_path / "uit/annotations")},
    }}))
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump({"oracle": {"arch": "oracle"}}))
    p = {"datasets": datasets, "models": models, "manifest": tmp_path / "manifest",
         "hash": tmp_path / "ph.parquet", "preds": tmp_path / "preds",
         "results": tmp_path / "results.csv", "pairs": tmp_path / "pairs.csv",
         "excl": tmp_path / "excl.txt"}
    r = CliRunner().invoke(app, ["manifest", "build", "--datasets", str(datasets), "--out",
                                 str(p["manifest"]), "--exclusions", str(p["excl"])])
    assert r.exit_code == 0, r.output
    return p


def _eval_args(p: dict[str, Path], *extra: str) -> list[str]:
    return ["eval", "--n-boot", "20", "--models", str(p["models"]), "--manifest",
            str(p["manifest"]), "--hash-cache", str(p["hash"]), "--pred-cache", str(p["preds"]),
            "--results", str(p["results"]), *extra]


def _leak_args(p: dict[str, Path], *extra: str) -> list[str]:
    return ["leakage", "check", "--manifest", str(p["manifest"]), "--cache", str(p["hash"]),
            "--out", str(p["pairs"]), *extra]


# --- leakage check -------------------------------------------------------------------------

def test_leakage_check_fails_and_appends_after_unterminated_line(tmp_path: Path):
    p = _setup(tmp_path, dup=True)
    p["excl"].write_bytes(b"x")  # no trailing newline
    r = CliRunner().invoke(app, _leak_args(p, "--write-exclusions", str(p["excl"])))
    assert r.exit_code == 1, r.output
    lines = p["excl"].read_text(encoding="utf-8").splitlines()
    assert lines == ["x", f"visdrone/train/{DUP_STEM}"]


def _pool_scene_leak(tmp_path: Path) -> dict[str, Path]:
    """UD-pool scene CLIPtrain (3 frames) whose frame 2 is a copy of test frame CLIPtest_1."""
    p = _setup(tmp_path)
    uit = tmp_path / "uit" / "images"
    shutil.copy(uit / "test" / "CLIPtest_1.jpg", uit / "train" / "CLIPtrain_2.jpg")
    r = CliRunner().invoke(app, ["manifest", "build", "--datasets", str(p["datasets"]), "--out",
                                 str(p["manifest"]), "--exclusions", str(p["excl"])])
    assert r.exit_code == 0, r.output
    return p


def test_leakage_frame_level_exclusions_and_scene_summary(tmp_path: Path):
    p = _pool_scene_leak(tmp_path)
    scenes_csv = tmp_path / "scenes.csv"
    r = CliRunner().invoke(app, _leak_args(p, "--write-exclusions", str(p["excl"]),
                                           "--scenes-out", str(scenes_csv)))
    assert r.exit_code == 1
    assert read_exclusions(p["excl"]) == {"uitdrone7/train/CLIPtrain_2"}
    assert "uitdrone7/CLIPtrain" in r.output  # scene summary printed
    sc = pd.read_csv(scenes_csv)
    assert sc.to_dict("records") == [{
        "train_scene": "uitdrone7/CLIPtrain", "train_set": "UD-pool",
        "test_scene": "uitdrone7/CLIPtest", "test_set": "T-VN", "pairs": 1,
        "train_images_flagged": 1, "train_scene_images": 3,
    }]


def test_leakage_exclude_scenes_writes_whole_scene(tmp_path: Path):
    p = _pool_scene_leak(tmp_path)
    r = CliRunner().invoke(app, _leak_args(p, "--write-exclusions", str(p["excl"]),
                                           "--exclude-scenes",
                                           "--scenes-out", str(tmp_path / "scenes.csv")))
    assert r.exit_code == 1
    assert read_exclusions(p["excl"]) == {f"uitdrone7/train/CLIPtrain_{i}" for i in (1, 2, 3)}
    # after excluding the scene and rebuilding, the check passes
    r = CliRunner().invoke(app, ["manifest", "build", "--datasets", str(p["datasets"]), "--out",
                                 str(p["manifest"]), "--exclusions", str(p["excl"])])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, _leak_args(p, "--scenes-out", str(tmp_path / "scenes.csv")))
    assert r.exit_code == 0, r.output


# --- manifest build ------------------------------------------------------------------------

def test_manifest_build_warns_about_unmatched_exclusions(tmp_path: Path):
    p = _setup(tmp_path)
    p["excl"].write_text("visdrone/train/nope\nvisdrone/train/also_nope\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["manifest", "build", "--datasets", str(p["datasets"]), "--out",
                                 str(p["manifest"]), "--exclusions", str(p["excl"])])
    assert r.exit_code == 0, r.output
    assert "warning: 2 exclusion ids not found in the manifest" in r.output


# --- eval ----------------------------------------------------------------------------------

def test_eval_refuses_leaky_manifest(tmp_path: Path):
    p = _setup(tmp_path, dup=True)
    r = CliRunner().invoke(app, _eval_args(p, "--model", "oracle", "--sets", "T-VD"))
    assert r.exit_code == 1
    assert "near-duplicate a test image" in r.stderr
    assert not p["results"].exists()


def test_eval_unknown_model_or_set_fails_clearly(tmp_path: Path):
    p = _setup(tmp_path)
    r = CliRunner().invoke(app, _eval_args(p, "--model", "nope", "--sets", "T-VD"))
    assert r.exit_code != 0 and "'nope' not in models config" in r.output
    r = CliRunner().invoke(app, _eval_args(p, "--model", "oracle", "--sets", "T-XX"))
    assert r.exit_code != 0 and "T-XX" in r.output and "missing from the manifest" in r.output


def test_eval_map_classes_pre_declared(tmp_path: Path):
    p = _setup(tmp_path)
    r = CliRunner().invoke(app, _eval_args(p, "--model", "oracle", "--sets", "T-VD,T-VN",
                                           "--map-classes", "motorbike, car"))
    assert r.exit_code == 0, r.output
    df = pd.read_csv(p["results"])
    assert (df.notes == "mAP classes: motorbike,car (pre-declared)").all()
    assert (df["slice"] == "all").all() and df.set_hash.astype(str).str.len().eq(8).all()
    assert {f"AP50/{c}" for c in ("motorbike", "car", "pedestrian", "bus")} <= set(df.metric)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")
    r = CliRunner().invoke(app, ["report", "--results", str(p["results"]), "--readme", str(readme)])
    assert r.exit_code == 0, r.output
    assert "- T-VN: mAP over motorbike,car (pre-declared)" in readme.read_text()


def test_eval_map_classes_rejects_unknown_names(tmp_path: Path):
    p = _setup(tmp_path)
    r = CliRunner().invoke(app, _eval_args(p, "--model", "oracle", "--map-classes", "car,boat"))
    assert r.exit_code == 2 and "boat" in r.output


# --- prediction cache ----------------------------------------------------------------------

class _CountingDetector:
    name, cache_tag = "fake", "fake-v1"

    def __init__(self, calls: list[int]):
        self.calls = calls

    def predict_paths(self, paths):
        self.calls.append(len(paths))
        for _ in paths:
            yield Detections.empty()


def test_prediction_cache_survives_manifest_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = _setup(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(run_mod, "build_detector", lambda *a, **k: _CountingDetector(calls))
    kw = dict(models_cfg={"fake": {"arch": "fake"}}, manifest_dir=p["manifest"],
              hash_cache=p["hash"], pred_cache=p["preds"])
    first = [r for rows in evaluate_model("fake", ["T-VD"], 640, 5, 0, 0.01, **kw) for r in rows]
    assert calls == [3]

    # exclude one TRAIN image -> new manifest hash, same T-VD images
    images, boxes = build_manifest(load_yaml(p["datasets"]), set())
    train_id = images[images.set_name.isin(["VD-train", "VD-tune"])].image_id.iloc[0]
    images, boxes = build_manifest(load_yaml(p["datasets"]), {train_id})
    save_manifest(images, boxes, p["manifest"])

    second = [r for rows in evaluate_model("fake", ["T-VD"], 640, 5, 0, 0.01, **kw) for r in rows]
    assert calls == [3]  # T-VD predictions came from the cache
    assert first[0]["manifest_hash"] != second[0]["manifest_hash"]
    assert first[0]["set_hash"] == second[0]["set_hash"]


# --- pseudo-labels and YOLO export ---------------------------------------------------------

def _flat(text: str) -> str:
    """CLI output with colour codes, panel borders and line wraps removed (Rich colours its output
    on CI runners, which would split the words a test looks for)."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return " ".join(text.replace("\u2502", " ").split())


def _with_pool(tmp_path: Path, n: int = 2) -> dict[str, Path]:
    """_setup plus `n` P-train frames (copies of VD-train images under new paths)."""
    p = _setup(tmp_path)
    images, boxes, _ = load_manifest(p["manifest"])
    src = images[images.set_name == "VD-train"].head(n).reset_index(drop=True)
    rows = []
    for i, r in src.iterrows():
        dst = tmp_path / "pool" / f"v1_{i}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(r.path, dst)
        rows.append({**r.to_dict(), "image_id": f"pool/v1_{i}", "path": str(dst),
                     "source": "pool", "scene_id": "pool/v1", "set_name": "P-train"})
    images = pd.concat([images, pd.DataFrame(rows)], ignore_index=True)
    save_manifest(images, boxes, p["manifest"])
    return p


def _fake_pseudo(out: Path, ids, manifest_hash: str) -> None:
    frames = pd.DataFrame({"image_id": list(ids), "n_boxes": 0, "disagreement": 0.0})
    save_pseudo(pd.DataFrame(columns=PRED_COLS), frames, {"manifest_hash": manifest_hash}, out)


def test_yolo_export_checks_pseudo_label_coverage_and_manifest(tmp_path: Path):
    p = _with_pool(tmp_path)
    images, _, mhash = load_manifest(p["manifest"])
    pool = images[images.set_name == "P-train"].image_id.tolist()
    pseudo = tmp_path / "pseudo" / "fused"
    args = ["yolo", "export", "--sets", "P-train", "--pseudo", str(pseudo), "--manifest",
            str(p["manifest"]), "--out", str(tmp_path / "yolo")]
    _fake_pseudo(pseudo, pool[:1], mhash)
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 2
    assert "do not cover 1 images of P-train" in _flat(r.output)
    assert not (tmp_path / "yolo" / "P-train-fused.txt").exists()
    _fake_pseudo(pseudo, pool, "deadbeef")
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert f"warning: pseudo-labels were made against manifest deadbeef, current is {mhash}" in r.stderr
    assert len((tmp_path / "yolo" / "P-train-fused.txt").read_text().split()) == 2


def _thr(p: dict[str, Path], model: str, out: Path) -> None:
    r = CliRunner().invoke(app, ["thresholds", "--model", model, "--set", "VD-tune", "--out", str(out),
                                 "--models", str(p["models"]), "--manifest", str(p["manifest"]),
                                 "--hash-cache", str(p["hash"]), "--pred-cache", str(p["preds"])])
    assert r.exit_code == 0, r.output


def test_pseudolabel_member_thresholds_and_threshold_file_checks(tmp_path: Path):
    p = _with_pool(tmp_path)
    p["models"].write_text(yaml.safe_dump({
        "oracle": {"arch": "oracle"}, "fused": {"arch": "fused", "members": ["oracle", "oracle"]},
    }))
    thr, members = tmp_path / "thr" / "fused.yaml", tmp_path / "thr" / "members"
    _thr(p, "fused", thr)
    out = tmp_path / "pseudo" / "fused"
    args = ["pseudolabel", "--model", "fused", "--sets", "P-train", "--thresholds", str(thr),
            "--member-thresholds-dir", str(members), "--out", str(out),
            "--models", str(p["models"]), "--manifest", str(p["manifest"]),
            "--hash-cache", str(p["hash"]), "--pred-cache", str(p["preds"])]
    r = CliRunner().invoke(app, args)  # no member thresholds yet
    assert r.exit_code == 2 and "cvt thresholds --model oracle" in _flat(r.output)
    assert not out.exists()

    _thr(p, "oracle", members / "oracle.yaml")
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert [m["name"] for m in meta["members"]] == ["oracle", "oracle"]
    assert meta["members"][0]["thresholds_file"] == str(members / "oracle.yaml")
    frames = pd.read_parquet(out / "frames.parquet")
    assert list(frames.columns) == ["image_id", "n_boxes", "disagreement", "n_boxes_m0", "n_boxes_m1"]
    assert len(frames) == 2

    good = yaml.safe_load(thr.read_text(encoding="utf-8"))
    thr.write_text(yaml.safe_dump({**good, "cache_tag": "other"}))
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 2 and "different detector configuration" in _flat(r.output)

    th = dict(good["thresholds"])
    th.pop("bus")
    th["boat"] = 0.5
    thr.write_text(yaml.safe_dump({**good, "thresholds": th}))
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 2 and "boat" in r.output and "bus" in r.output

    thr.write_text(yaml.safe_dump(good))
    m = yaml.safe_load((members / "oracle.yaml").read_text(encoding="utf-8"))
    (members / "oracle.yaml").write_text(yaml.safe_dump({**m, "cache_tag": "other"}))
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 2 and "different detector configuration" in _flat(r.output)
