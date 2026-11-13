"""End-to-end CLI on tiny synthetic data with the oracle detector (CPU, seconds)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from cvt.cli import app
from tests.helpers import make_uitdrone7, make_visdrone

pytestmark = pytest.mark.smoke


def test_manifest_leakage_eval_report(tmp_path: Path):
    make_visdrone(tmp_path / "vd" / "train", n_images=4, seed0=1)
    make_visdrone(tmp_path / "vd" / "val", n_images=3, seed0=100)
    make_uitdrone7(tmp_path / "uit")
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text(yaml.safe_dump({"sources": {
        "visdrone_det": {"train_dir": str(tmp_path / "vd/train"), "val_dir": str(tmp_path / "vd/val"),
                         "rider_merge": True, "tune_frac": 0.1},
        "uitdrone7": {"images_dir": str(tmp_path / "uit/images"),
                      "labels_dir": str(tmp_path / "uit/annotations")},
    }}))
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump({"oracle": {"arch": "oracle"}}))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")
    m, results = tmp_path / "manifest", tmp_path / "results.csv"
    run = CliRunner().invoke

    r = run(app, ["manifest", "build", "--datasets", str(datasets), "--out", str(m),
                  "--exclusions", str(tmp_path / "none.txt")])
    assert r.exit_code == 0, r.output
    assert "T-VN" in r.output

    r = run(app, ["leakage", "check", "--manifest", str(m), "--cache", str(tmp_path / "ph.parquet"),
                  "--out", str(tmp_path / "pairs.csv")])
    assert r.exit_code == 0, r.output

    r = run(app, ["eval", "--model", "oracle", "--sets", "T-VD,T-VN", "--n-boot", "50",
                  "--models", str(models), "--manifest", str(m), "--hash-cache",
                  str(tmp_path / "ph.parquet"), "--pred-cache", str(tmp_path / "preds"),
                  "--results", str(results)])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(results)
    top = df[df.metric.isin(["mAP50", "mAP50_95"])]
    assert set(top.eval_set) == {"T-VD", "T-VN"} and (top.value == 1.0).all()

    r = run(app, ["report", "--results", str(results), "--readme", str(readme)])
    assert r.exit_code == 0, r.output
    assert "100.0" in readme.read_text()
    assert b"\r\n" not in readme.read_bytes()
