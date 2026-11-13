import numpy as np

from cvt.eval.bootstrap import bootstrap_ci, paired_bootstrap, scene_weights


def test_scene_weights_constant_within_scene():
    codes = np.array([0, 0, 1, 1, 1, 2])
    w = scene_weights(codes, 3, np.random.default_rng(0))
    assert w[0] == w[1] and w[2] == w[3] == w[4]
    counts = np.array([w[0], w[2], w[5]])
    assert counts.sum() == 3


def test_bootstrap_ci_brackets_true_mean():
    values = np.arange(20, dtype=float)
    scenes = np.repeat(np.arange(10), 2)

    def fn(w):
        w = np.ones_like(values) if w is None else w
        return {"mean": float((values * w).sum() / w.sum())}

    lo, hi = bootstrap_ci(fn, scenes, n_boot=500, seed=1)["mean"]
    assert lo < values.mean() < hi
    assert bootstrap_ci(fn, scenes, n_boot=500, seed=1)["mean"] == (lo, hi)


def test_paired_bootstrap_identical_models_zero_delta():
    scenes = np.repeat(np.arange(5), 3)

    def fn(w):
        return {"m": 0.5 if w is None else float(w.mean())}

    delta, lo, hi = paired_bootstrap(fn, fn, scenes, n_boot=100)["m"]
    assert delta == lo == hi == 0.0


def test_interval_is_nan_when_too_many_resamples_invalid():
    scenes = np.arange(12)

    def fn(w):
        return {"m": float("nan") if (w is not None and w[0] == 0) else 1.0}

    lo, hi = bootstrap_ci(fn, scenes, n_boot=200, seed=0)["m"]  # scene 0 missing ~28% of draws
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = bootstrap_ci(fn, scenes, n_boot=200, seed=0, min_valid=0.5)["m"]
    assert lo == hi == 1.0


def test_paired_bootstrap_survives_vanishing_keys():
    scenes = np.arange(12)

    def fa(w):
        out = {"m": 1.0}
        if w is None or w[0] > 0:
            out["rare"] = 0.5
        return out

    res = paired_bootstrap(fa, fa, scenes, n_boot=100, seed=0)
    assert res["m"] == (0.0, 0.0, 0.0)
    assert res["rare"][0] == 0.0 and np.isnan(res["rare"][1])
