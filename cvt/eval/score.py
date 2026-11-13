"""Score one set of predictions against the manifest: match once, then any metric/bootstrap."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cvt.data.schema import CLASSES
from cvt.eval.ap import AREA_NAMES, MatchTable, build_tables, eligible_classes, summarize
from cvt.eval.coco_bridge import to_coco


@dataclass(eq=False)
class SetScore:
    image_ids: np.ndarray  # image order shared by tables, scenes and bootstrap weights
    scenes: np.ndarray
    tables: dict[tuple[int, str], MatchTable]
    class_ids: list[int]
    map_classes: list[int]

    def metrics(
        self, weights: np.ndarray | None = None, strict: bool = False,
        areas: Sequence[str] = AREA_NAMES,
    ) -> dict[str, float]:
        return summarize(self.tables, self.class_ids, weights, areas=areas,
                         map_classes=self.map_classes, strict=strict)

    def resample_metrics(self, weights: np.ndarray | None) -> dict[str, float]:
        """Metric function for the bootstrap: area 'all', fixed class set, strict."""
        return self.metrics(weights, strict=True, areas=("all",))


def score_set(
    images: pd.DataFrame, boxes: pd.DataFrame, dt: pd.DataFrame,
    map_classes: Sequence[int] | None = None,
) -> SetScore:
    """Match `dt` against the GT of `images` (in their order). By default mAP averages the
    classes eligible on this set (GT in >= min(4, #scenes) scenes)."""
    class_ids = list(range(len(CLASSES)))
    gt = boxes[boxes.image_id.isin(images.image_id)]
    cgt, cdt, ids = to_coco(images, gt, dt, class_ids)
    tables = build_tables(cgt, cdt, ids, class_ids)
    scenes = images.scene_id.to_numpy()
    mc = (eligible_classes(tables, class_ids, scenes) if map_classes is None
          else sorted(set(map_classes)))
    return SetScore(images.image_id.to_numpy(), scenes, tables, class_ids, mc)
