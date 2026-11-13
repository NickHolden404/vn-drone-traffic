"""Raw dataset ids and model class names -> our 6-class schema."""

from __future__ import annotations

from cvt.data.schema import CLASS_ID, CLASSES

IGNORE = -1

# VisDrone raw category -> our id. 2 ("people") includes riders; adapters run rider
# merge on raw ids before applying this map.
VISDRONE_RAW: dict[int, int] = {
    0: IGNORE,  # ignored region
    1: CLASS_ID["pedestrian"],
    2: CLASS_ID["pedestrian"],  # people
    3: CLASS_ID["bicycle"],
    4: CLASS_ID["car"],
    5: CLASS_ID["car"],  # van
    6: CLASS_ID["truck"],
    7: IGNORE,  # tricycle
    8: IGNORE,  # awning-tricycle
    9: CLASS_ID["bus"],
    10: CLASS_ID["motorbike"],
    11: IGNORE,  # others
}
# Tricycles (7, 8) are ignore regions; with rider merge, people sitting on them are folded
# into the tricycle box and so become part of that ignore region (UIT has no tricycle class).
VISDRONE_PEOPLE = 2
VISDRONE_RIDEABLE = (3, 10)
VISDRONE_TRICYCLES = (7, 8)  # people on these are folded into the (ignored) tricycle box

# UIT-Drone7 raw id -> our id (verified visually 2026-10-05; motorbike/bicycle boxes include riders).
UITDRONE7_RAW: dict[int, int] = {
    0: CLASS_ID["pedestrian"],
    1: CLASS_ID["motorbike"],
    2: CLASS_ID["car"],
    3: CLASS_ID["bus"],
    4: CLASS_ID["car"],  # van
    5: CLASS_ID["truck"],
    6: CLASS_ID["bicycle"],
}

# Class *names* emitted by a model -> our class name (None = drop the prediction).
NAME_MAPS: dict[str, dict[str, str | None]] = {
    "visdrone": {
        "pedestrian": "pedestrian",
        "people": "pedestrian",
        "bicycle": "bicycle",
        "car": "car",
        "van": "car",
        "truck": "truck",
        "tricycle": None,
        "awning-tricycle": None,
        "bus": "bus",
        "motor": "motorbike",
        "others": None,  # VisDrone catch-all category (an ignore class in GT)
    },
}
NAME_MAPS["cvt"] = {c: c for c in CLASSES}  # our own models emit the 6 classes directly

# Per name-map: (rider class name, rideable vehicle class names) used for prediction rider merge.
RIDER_NAMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "visdrone": ("people", ("bicycle", "motor")),
}
