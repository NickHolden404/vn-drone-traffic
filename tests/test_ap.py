import contextlib
import io

import numpy as np
import pandas as pd
from pycocotools.cocoeval import COCOeval

from cvt.eval.ap import (
    AREA_NAMES,
    ap_from_table,
    build_tables,
    configure,
    eligible_classes,
    summarize,
)
from cvt.eval.coco_bridge import to_coco

CLASS_IDS = list(range(6))


def synthetic(seed: int = 0, n_img: int = 12):
    rng = np.random.default_rng(seed)
    images = pd.DataFrame(
        {"image_id": [f"im{i}" for i in range(n_img)], "width": 1920, "height": 1080,
         "scene_id": [f"s{i // 3}" for i in range(n_img)]}
    )
    gt, dt = [], []
    for iid in images.image_id:
        for _ in range(rng.integers(0, 8)):
            x, y = rng.uniform(0, 1800), rng.uniform(0, 1000)
            w, h = rng.uniform(4, 80, size=2)
            lab = int(rng.integers(0, 6))
            gt.append((iid, x, y, x + w, y + h, lab, False))
            if rng.random() < 0.8:
                j = rng.normal(0, 3, 4)
                x1, y1 = x + j[0], y + j[1]
                dt.append((iid, x1, y1, max(x + w + j[2], x1 + 1), max(y + h + j[3], y1 + 1),
                           float(rng.random()), lab))
        for _ in range(rng.integers(0, 3)):
            x, y = rng.uniform(0, 1800), rng.uniform(0, 1000)
            dt.append((iid, x, y, x + 30, y + 30, float(rng.random()), int(rng.integers(0, 6))))
        if rng.random() < 0.3:
            gt.append((iid, 0, 0, 200, 200, -1, True))
    gt = pd.DataFrame(gt, columns=["image_id", "x1", "y1", "x2", "y2", "label", "is_ignore"])
    dt = pd.DataFrame(dt, columns=["image_id", "x1", "y1", "x2", "y2", "score", "label"])
    return images, gt, dt


def test_unweighted_ap_matches_pycocotools_exactly():
    images, gt, dt = synthetic()
    cgt, cdt, ids = to_coco(images, gt, dt, CLASS_IDS)
    tables = build_tables(cgt, cdt, ids, CLASS_IDS)
    E = COCOeval(cgt, cdt, "bbox")
    configure(E, ids, CLASS_IDS)
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
    prec = E.eval["precision"]  # [T, R, K, A, M]
    for k, c in enumerate(CLASS_IDS):
        for a, area in enumerate(AREA_NAMES):
            ours = ap_from_table(tables[(c, area)])
            ref = prec[:, :, k, a, 0]
            if ref[0, 0] == -1:
                assert np.isnan(ours).all()
            else:
                np.testing.assert_allclose(ours, ref.mean(axis=1), atol=1e-12)


def test_integer_weights_equal_duplicating_images():
    images, gt, dt = synthetic(seed=3)
    cgt, cdt, ids = to_coco(images, gt, dt, CLASS_IDS)
    tables = build_tables(cgt, cdt, ids, CLASS_IDS)
    w = np.ones(len(images))
    w[0], w[2] = 2, 0

    keep = images.image_id != "im2"
    dup = images[images.image_id == "im0"].assign(image_id="im0_dup")
    images2 = pd.concat([images[keep], dup], ignore_index=True)
    gt2 = pd.concat([gt[gt.image_id != "im2"], gt[gt.image_id == "im0"].assign(image_id="im0_dup")])
    dt2 = pd.concat([dt[dt.image_id != "im2"], dt[dt.image_id == "im0"].assign(image_id="im0_dup")])
    cgt2, cdt2, ids2 = to_coco(images2, gt2, dt2, CLASS_IDS)
    tables2 = build_tables(cgt2, cdt2, ids2, CLASS_IDS)
    for key in tables:
        np.testing.assert_allclose(
            ap_from_table(tables[key], w), ap_from_table(tables2[key]), atol=1e-12
        )


def test_summarize_keys_and_perfect_detector():
    images, gt, _ = synthetic(seed=5)
    obj = gt[~gt.is_ignore]
    perfect = obj.assign(score=1.0)[["image_id", "x1", "y1", "x2", "y2", "score", "label"]]
    cgt, cdt, ids = to_coco(images, gt, perfect, CLASS_IDS)
    m = summarize(build_tables(cgt, cdt, ids, CLASS_IDS), CLASS_IDS)
    assert m["mAP50"] == 1.0 and m["mAP50_95"] == 1.0
    assert "AP50_95/motorbike" in m and "mAP50_95@tiny" in m


def _tables(seed=0):
    images, gt, dt = synthetic(seed=seed)
    cgt, cdt, ids = to_coco(images, gt, dt, CLASS_IDS)
    return images, gt, build_tables(cgt, cdt, ids, CLASS_IDS)


def test_eligible_classes_require_four_scenes():
    images, gt, tables = _tables()
    scenes = images.scene_id.to_numpy()
    elig = eligible_classes(tables, CLASS_IDS, scenes)
    for c in CLASS_IDS:
        ids = set(gt[(gt.label == c) & ~gt.is_ignore].image_id)
        n = images[images.image_id.isin(ids)].scene_id.nunique()
        assert (c in elig) == (n >= min(4, images.scene_id.nunique()))
    one_scene = np.zeros(len(images), int)
    assert eligible_classes(tables, CLASS_IDS, one_scene) == [
        c for c in CLASS_IDS if tables[(c, "all")].npig.sum() > 0
    ]


def test_summarize_always_emits_per_class_keys_and_strict_mode():
    images, gt, tables = _tables()
    obj = gt[~gt.is_ignore]
    c = int(obj.label.value_counts().index[-1])  # rarest class present
    has_c = images.image_id.isin(set(obj[obj.label == c].image_id)).to_numpy()
    w = np.where(has_c, 0.0, 1.0)  # resample in which class c vanishes
    loose = summarize(tables, CLASS_IDS, w, areas=("all",))
    strict = summarize(tables, CLASS_IDS, w, areas=("all",), strict=True)
    from cvt.data.schema import CLASSES
    assert np.isnan(loose[f"AP50/{CLASSES[c]}"])
    assert not np.isnan(loose["mAP50"]) and np.isnan(strict["mAP50"])
