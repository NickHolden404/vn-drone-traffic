from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from cvt.cli import app
from cvt.eval.report import compare_table, inject_readme, results_table
from cvt.eval.results import RESULT_COLS, append_results


def _row(**kw):
    base = {c: "" for c in RESULT_COLS}
    base.update(timestamp="2026-10-05T00:00:00", model="m", imgsz=1280, eval_set="T-VN",
                metric="mAP50", value=0.5, ci_lo=0.4, ci_hi=0.6)
    base.update(kw)
    return base


def test_append_writes_header_once(tmp_path: Path):
    p = tmp_path / "r.csv"
    append_results(p, [_row()])
    append_results(p, [_row(value=0.7, timestamp="2026-10-06T00:00:00")])
    df = pd.read_csv(p)
    assert len(df) == 2 and list(df.columns) == RESULT_COLS


def test_table_uses_latest_row_and_formats_ci(tmp_path: Path):
    p = tmp_path / "r.csv"
    append_results(p, [_row(), _row(value=0.7, ci_lo=0.65, ci_hi=0.75, timestamp="2026-10-06")])
    md = results_table(pd.read_csv(p), metrics=("mAP50",))
    assert "| m | 1280 | 70.0 [65.0, 75.0] |" in md
    assert "T-VN mAP50" in md


def test_table_tie_on_timestamp_keeps_last_appended(tmp_path: Path):
    p = tmp_path / "r.csv"
    append_results(p, [_row(value=0.1)])
    append_results(p, [_row(value=0.2)])
    md = results_table(pd.read_csv(p), metrics=("mAP50",))
    assert "| m | 1280 | 20.0 [40.0, 60.0] |" in md


def test_inject_readme_replaces_between_markers():
    text = "a\n<!-- results:start -->\nold\n<!-- results:end -->\nb\n"
    out = inject_readme(text, "NEW")
    assert "NEW" in out and "old" not in out and out.startswith("a\n") and out.endswith("b\n")
    with pytest.raises(ValueError, match="markers"):
        inject_readme("no markers", "x")


def test_append_rejects_header_mismatch(tmp_path: Path):
    p = tmp_path / "r.csv"
    old = [c for c in RESULT_COLS if c not in ("slice", "set_hash")]
    p.write_text(",".join(old) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="slice"):
        append_results(p, [_row()])
    assert p.read_text(encoding="utf-8") == ",".join(old) + "\n"  # nothing appended


def test_results_cols_has_slice_and_set_hash():
    assert RESULT_COLS[RESULT_COLS.index("eval_set") + 1] == "slice"
    assert RESULT_COLS[RESULT_COLS.index("manifest_hash") + 1] == "set_hash"


def test_table_ignores_non_all_slices(tmp_path: Path):
    p = tmp_path / "r.csv"
    append_results(p, [_row(slice="all", value=0.5),
                       _row(slice="night", value=0.9, timestamp="2026-10-07")])
    md = results_table(pd.read_csv(p), metrics=("mAP50",))
    assert "| m | 1280 | 50.0 [40.0, 60.0] |" in md and "90.0" not in md


def test_table_footnotes_map_classes_per_set(tmp_path: Path):
    p = tmp_path / "r.csv"
    append_results(p, [
        _row(notes="mAP classes: motorbike,car"),
        _row(model="m2", notes="mAP classes: car (pre-declared)"),
        _row(eval_set="T-VD", notes="mAP classes: car,bus"),
        _row(eval_set="T-VD", model="m2", notes="mAP classes: car,bus"),
    ])
    md = results_table(pd.read_csv(p), metrics=("mAP50",))
    assert "- T-VD: mAP over car,bus" in md
    vn = next(line for line in md.splitlines() if line.startswith("- T-VN:"))
    assert "motorbike,car (m)" in vn and "car (pre-declared) (m2)" in vn
    assert "T-VN has 12" not in md
    assert "With few clips (see n_scenes in reports/results.csv) the CIs are approximate." in md


def test_main_table_keeps_detection_sets_and_mot_sets_go_to_tracking(tmp_path: Path):
    p = tmp_path / "r.csv"
    det = [_row(eval_set=s, notes="mAP classes: car") for s in ("T-VD", "T-VN", "VD-tune")]
    mot = [_row(eval_set=s, value=0.33, ci_lo=0.3, ci_hi=0.36, notes="mAP classes: car")
           for s in ("MOT-tune", "T-MOT", "T-UAVDT")]
    append_results(p, det + mot)
    only = tmp_path / "only.csv"
    append_results(only, det)
    expected = results_table(pd.read_csv(only))
    assert results_table(pd.read_csv(p), sets=("T-VD", "T-VN", "VD-tune")) == expected
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- tracking:start -->\n<!-- tracking:end -->\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["report", "--results", str(p), "--readme", str(readme)])
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    main = text.split("<!-- results:end -->")[0]
    assert main == f"<!-- results:start -->\n{expected}\n"
    assert "MOT" not in main and "UAVDT" not in main and "33.0" not in main
    track = text.split("<!-- tracking:start -->\n")[1].split("\n<!-- tracking:end -->")[0]
    assert track.startswith("Detection mAP on tracking sets")
    assert "MOT-tune mAP50" in track and "T-UAVDT mAP50" in track and "T-VD" not in track
    assert "| m | 1280 | 33.0 [30.0, 36.0] |" in track


def test_report_cli_separates_results_and_deltas(tmp_path: Path):
    p = tmp_path / "r.csv"
    normal = [_row(), _row(metric="mAP50_95", value=0.3, notes="mAP classes: car"),
              _row(eval_set="T-VD", notes="mAP classes: car")]
    deltas = [_row(metric="delta/mAP50_95", value=0.02, ci_lo=-0.01, ci_hi=0.05,
                   notes="vs m2@1280; mAP classes: car", slice="all")]
    append_results(p, normal + deltas)
    only = tmp_path / "only.csv"
    append_results(only, normal)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- compare:start -->\n<!-- compare:end -->\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["report", "--results", str(p), "--readme", str(readme)])
    assert r.exit_code == 0, r.output
    text = readme.read_text(encoding="utf-8")
    expected = results_table(pd.read_csv(only))
    assert results_table(pd.read_csv(p)) == expected
    assert f"<!-- results:start -->\n{expected}\n<!-- results:end -->" in text
    block = text.split("<!-- compare:start -->\n")[1].split("\n<!-- compare:end -->")[0]
    assert block == compare_table(pd.read_csv(p))
    assert "| m@1280 | vs m2@1280 | +2.0 [-1.0, +5.0] |" in block and "delta" not in expected
