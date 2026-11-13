from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from cvt.cli import app
from cvt.eval.report import (
    analytics_table,
    follow_table,
    is_analytics,
    results_table,
    tracking_table,
)
from cvt.eval.results import RESULT_COLS, append_results

ROOT = Path(__file__).resolve().parents[1]


def _arow(**kw):
    base = {c: "" for c in RESULT_COLS}
    base.update(eval_set="A-LONG", metric="anomaly/auc", imgsz=1536, slice="all",
                timestamp="2026-11-12T00:00:00", model="p+events-both", value=0.5,
                ci_lo=0.4, ci_hi=0.6)
    base.update(kw)
    return base


def _rows():
    rows = [_arow(slice=s, value=v, ci_lo=v - 0.05, ci_hi=v + 0.05, n_scenes=n)
            for s, v, n in (("all", 0.71, 16), ("trained", 0.70, 11), ("held-out", 0.74, 5))]
    rows.append(_arow(model="p", eval_set="T-MOT", metric="od/count_pct_err", value=0.25,
                      ci_lo=0.1, ci_hi=0.4, n_scenes=4,
                      notes="4 sequences, 11 arms, 37 complete GT tracks"))
    rows.append(_arow(model="p", eval_set="T-MOT", metric="od/complete_ratio", value=1.25,
                      ci_lo=float("nan"), ci_hi=float("nan"), n_scenes=4,
                      notes="4 sequences, 11 arms, 37 complete GT tracks"))
    return rows


def _clips():
    return pd.DataFrame([{"clip": "DJI_0143", "group": "trained", "height_m": 61.0,
                          "mpp_telemetry": 0.045, "mpp_car": 0.037, "auc_both": float("nan")}])


def test_analytics_table_and_exclusion():
    assert analytics_table(pd.DataFrame(columns=RESULT_COLS), None) == "(no analytics results yet)"
    df = pd.DataFrame(_rows())
    md = analytics_table(df, _clips())
    assert "71.0 [66.0, 76.0]" in md and "held-out (5 clips)" in md
    assert "25.0 [10.0, 40.0]" in md and "37 complete GT tracks" in md
    assert "| DJI_0143 | trained | 61.0 | 4.5 | 3.7 |" in md and "unvalidated" in md
    assert "complete tracks, ours / GT %" in md and "| 125.0 |" in md and "recall" not in md
    assert "| cm/px car length | tracks |" in md and "can exceed 100 %" in md
    assert "shorter than 1 s" in md and "MOT-tune" in md and "JPG" in md
    assert "| – |" in md  # a missing clip column
    assert "not comparable" in md and "11 of the 16" in md
    assert is_analytics(df).all()
    for table in (results_table, tracking_table, follow_table):
        assert "events-both" not in table(df) and "od/" not in table(df)


def test_report_fills_the_analytics_block(tmp_path):
    res = tmp_path / "r.csv"
    append_results(res, _rows())
    clips = tmp_path / "clips.csv"
    _clips().to_csv(clips, index=False)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n"
                      "<!-- tracking:start -->\n<!-- tracking:end -->\n"
                      "<!-- follow:start -->\n<!-- follow:end -->\n"
                      "<!-- analytics:start -->\n(no analytics results yet)\n"
                      "<!-- analytics:end -->\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["report", "--results", str(res), "--readme", str(readme),
                                 "--clips-csv", str(clips)])
    assert r.exit_code == 0, r.output
    main, rest = readme.read_text(encoding="utf-8").split("<!-- results:end -->")
    block = rest.split("<!-- analytics:start -->")[1]
    assert "events-both" not in main and "od/" not in rest.split("<!-- analytics:start -->")[0]
    assert "71.0 [66.0, 76.0]" in block and "| DJI_0143 | trained |" in block
    assert "(no analytics results yet)" not in block


def test_results_page_has_the_analytics_block():
    text = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    assert "### Traffic analytics on full-length clips" in text
    assert "VisDrone-MOT / UAVDT ground-truth tracks" in text and "our own ground" not in text
    assert "\x07" not in text and "`scripts\\analytics.ps1`" in text
    assert text.index("<!-- follow:end -->") < text.index("<!-- analytics:start -->") < text.index(
        "### Deployment")
    assert "<!-- analytics:end -->" in text
    assert "reports/demo/analytics_DJI_0143.jpg" in text
    assert "reports/demo/analytics_DJI_0143_speed.jpg" in text
