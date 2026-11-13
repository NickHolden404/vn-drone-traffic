from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cvt.cli import app
from cvt.data.manifest import build_manifest, save_manifest
from cvt.data.schema import CLASSES
from cvt.detect.composite import FusedDetector
from cvt.detect.oracle import OracleDetector
from cvt.eval.report import compare_table, inject_readme
from cvt.eval.score import score_set
from tests.helpers import make_uitdrone7, make_visdrone


def _setup(tmp_path: Path) -> dict:
    make_visdrone(tmp_path / "vd" / "train", n_images=4, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=4, seed0=100)
    make_uitdrone7(tmp_path / "uit")
    cfg = {"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd/train"), "val_dir": str(tmp_path / "vd/val"),
                         "rider_merge": True, "tune_frac": 0.25},
        "uitdrone7": {"images_dir": str(tmp_path / "uit/images"),
                      "labels_dir": str(tmp_path / "uit/annotations")},
    }}
    images, boxes = build_manifest(cfg, exclusions=set())
    save_manifest(images, boxes, tmp_path / "m")
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump({"oracle": {"arch": "oracle"}, "oracle2": {"arch": "oracle"}}))
    return {"images": images, "boxes": boxes, "models": models, "m": tmp_path / "m",
            "results": tmp_path / "results.csv", "ph": tmp_path / "ph.parquet",
            "preds": tmp_path / "preds"}


def test_score_set_oracle_is_perfect(tmp_path: Path):
    s = _setup(tmp_path)
    im = s["images"][s["images"].set_name == "T-VD"].reset_index(drop=True)
    gt = s["boxes"][s["boxes"].image_id.isin(im.image_id) & ~s["boxes"].is_ignore]
    dt = gt.assign(score=1.0)[["image_id", "x1", "y1", "x2", "y2", "score", "label"]]
    sc = score_set(im, s["boxes"], dt)
    assert sc.metrics()["mAP50"] == 1.0
    assert list(sc.image_ids) == list(im.image_id) and len(sc.scenes) == len(im)


def test_compare_identical_models_gives_zero_delta(tmp_path: Path):
    s = _setup(tmp_path)
    r = CliRunner().invoke(app, [
        "compare", "--a", "oracle", "--b", "oracle2", "--sets", "T-VD", "--n-boot", "30",
        "--models", str(s["models"]), "--manifest", str(s["m"]), "--hash-cache", str(s["ph"]),
        "--pred-cache", str(s["preds"]), "--results", str(s["results"]),
    ])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(s["results"])
    d = df[df.metric == "delta/mAP50_95"].iloc[0]
    assert d.value == 0.0 and d.ci_lo == 0.0 and d.ci_hi == 0.0
    assert d.notes.startswith("vs oracle2@1280")


def test_compare_detects_missing_class(tmp_path: Path, monkeypatch):
    s = _setup(tmp_path)
    import cvt.eval.run as run
    from cvt.detect.base import Detections

    real = run.build_detector

    class NoPedestrians:
        def __init__(self, inner):
            self.inner, self.name, self.cache_tag = inner, "noped", "noped"

        def predict_paths(self, paths):
            for d in self.inner.predict_paths(paths):
                k = d.labels != 5
                yield Detections(d.boxes[k], d.scores[k], d.labels[k])

    def fake(name, *a, **kw):
        det = real("oracle" if name == "oracle2" else name, *a, **kw)
        return NoPedestrians(det) if name == "oracle2" else det

    monkeypatch.setattr(run, "build_detector", fake)
    rows = [r for batch in run.compare_models(
        "oracle", "oracle2", ["T-VD"], 1280, 1280, 30, 0, 0.01,
        models_cfg=yaml.safe_load(s["models"].read_text()), manifest_dir=s["m"],
        hash_cache=s["ph"], pred_cache=s["preds"]) for r in batch]
    ped = next(r for r in rows if r["metric"] == "delta/AP50_95/pedestrian")
    assert ped["value"] == pytest.approx(1.0) and ped["ci_lo"] == pytest.approx(1.0)


def test_compare_table_and_named_markers():
    df = pd.DataFrame([{
        "timestamp": "t", "model": "a", "imgsz": 1280, "eval_set": "T-VN", "slice": "all",
        "metric": "delta/mAP50_95", "value": 0.031, "ci_lo": 0.01, "ci_hi": 0.05,
        "notes": "vs b@1280; mAP classes: motorbike",
    }])
    md = compare_table(df)
    assert "| a@1280 | vs b@1280 |" in md and "+3.1 [+1.0, +5.0]" in md
    text = "x\n<!-- compare:start -->\nold\n<!-- compare:end -->\n"
    assert "NEW" in inject_readme(text, "NEW", name="compare")
    with pytest.raises(ValueError, match="markers"):
        inject_readme("none", "x", name="compare")


def _delta(**kw) -> dict:
    row = {"timestamp": "2026-10-05T00:00:00", "model": "a", "imgsz": 1280, "eval_set": "T-VN",
           "slice": "all", "metric": "delta/mAP50_95", "value": 0.031, "ci_lo": 0.01,
           "ci_hi": 0.05, "notes": "vs b@1280; mAP classes: motorbike"}
    row.update(kw)
    return row


def test_compare_table_latest_timestamp_wins():
    df = pd.DataFrame([_delta(timestamp="2026-10-06T00:00:00", value=0.02, ci_lo=0.0, ci_hi=0.04),
                       _delta(value=0.5)])  # older row appended later
    md = compare_table(df)
    assert "+2.0 [+0.0, +4.0]" in md and "+50.0" not in md


def test_compare_table_nan_and_missing_notes():
    md = compare_table(pd.DataFrame([_delta(value=float("nan"), ci_lo=float("nan"),
                                            ci_hi=float("nan"))]))
    assert "| a@1280 | vs b@1280 | – |" in md
    md = compare_table(pd.DataFrame([_delta(ci_lo=float("nan"), ci_hi=float("nan"))]))
    assert "| a@1280 | vs b@1280 | +3.1 |" in md
    md = compare_table(pd.DataFrame([_delta(notes=None)]))
    assert "| a@1280 | vs ? | +3.1 [+1.0, +5.0] |" in md


def test_compare_validates_models_and_refuses_self_comparison(tmp_path: Path):
    s = _setup(tmp_path)
    import cvt.eval.run as run

    cfg = yaml.safe_load(s["models"].read_text())
    kw = dict(models_cfg=cfg, manifest_dir=tmp_path / "no-manifest", hash_cache=s["ph"],
              pred_cache=s["preds"])
    with pytest.raises(KeyError, match="'nope' not in models config"):
        next(run.compare_models("oracle", "nope", ["T-VD"], 1280, 1280, 10, 0, 0.01, **kw))
    with pytest.raises(KeyError, match="'nope' not in models config"):
        next(run.compare_models("nope", "oracle", ["T-VD"], 1280, 1280, 10, 0, 0.01, **kw))
    with pytest.raises(ValueError, match="itself"):
        next(run.compare_models("oracle", "oracle", ["T-VD"], 1280, 1280, 10, 0, 0.01, **kw))
    msgs: list[str] = []
    kw["manifest_dir"] = s["m"]
    rows = next(run.compare_models("oracle", "oracle", ["T-VD"], 1280, 640, 10, 0, 0.01,
                                   log=msgs.append, **kw))
    assert rows and sum("matching" in m for m in msgs) == 2


class _Member:
    """Wraps the oracle under its own name and cache tag, counting predict_paths calls."""

    def __init__(self, name: str, inner, calls: list[str]):
        self.name, self.cache_tag, self.inner, self.calls = name, f"{name}-tag", inner, calls

    def predict_paths(self, paths):
        self.calls.append(self.name)
        return self.inner.predict_paths(paths)


def test_fused_set_predictions_caches_members_and_fusion(tmp_path: Path):
    s = _setup(tmp_path)
    import cvt.eval.run as run

    im = s["images"][s["images"].set_name == "T-VD"].reset_index(drop=True)
    oracle = OracleDetector(s["images"], s["boxes"])
    calls: list[str] = []
    fused = FusedDetector("fused", [_Member("m1", oracle, calls), _Member("m2", oracle, calls)])
    first = run.set_predictions(fused, "fused", im, "T-VD", s["preds"])
    assert sorted(calls) == ["m1", "m2"]
    names = sorted(p.name for p in s["preds"].iterdir())
    assert [n.split("_")[0] for n in names] == ["fused", "m1", "m2"]
    assert all(n.endswith(".parquet") for n in names)
    second = run.set_predictions(fused, "fused", im, "T-VD", s["preds"])
    assert sorted(calls) == ["m1", "m2"]  # fused cache hit: members not called again
    pd.testing.assert_frame_equal(first, second)
    assert len(first) and (first.score == 1.0).all()


def _thr_args(s: dict, *extra: str) -> list[str]:
    return ["thresholds", "--model", "oracle", "--models", str(s["models"]), "--manifest",
            str(s["m"]), "--hash-cache", str(s["ph"]), "--pred-cache", str(s["preds"]), *extra]


def test_thresholds_cli(tmp_path: Path):
    s = _setup(tmp_path)
    out = tmp_path / "thr.yaml"
    r = CliRunner().invoke(app, _thr_args(s, "--set", "T-VD", "--out", str(out)))
    assert r.exit_code == 2 and not out.exists()
    r = CliRunner().invoke(app, _thr_args(s, "--set", "VD-tune", "--out", str(out)))
    assert r.exit_code == 0, r.output
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert list(doc) == ["model", "set", "cache_tag", "imgsz", "conf", "thresholds"]
    assert doc["model"] == "oracle" and doc["set"] == "VD-tune"
    assert doc["cache_tag"] == "oracle" and doc["imgsz"] == 1280 and doc["conf"] == 0.01
    assert list(doc["thresholds"]) == list(CLASSES)
    tune = s["images"][s["images"].set_name == "VD-tune"].image_id
    gt = s["boxes"][s["boxes"].image_id.isin(tune) & ~s["boxes"].is_ignore]
    present = {CLASSES[c] for c in gt.label}
    assert present  # the fixture has GT on VD-tune
    for name, t in doc["thresholds"].items():
        assert t == (1.0 if name in present else 0.5), (name, t)
