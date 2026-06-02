"""Remapping and paths for the 3-class merged dataset pipeline."""

from __future__ import annotations

from pathlib import Path

FINAL_CLASS_NAMES = ["tank", "armored_vehicle", "civilian_vehicle"]
FINAL_NC = 3

TANK = 0
ARMORED_VEHICLE = 1
CIVILIAN_VEHICLE = 2

RUSSIAN_MILITARY_REMAP: dict[int, int] = {i: ARMORED_VEHICLE for i in range(6)}

ABOBA_REMAP: dict[int, int] = {
    1: ARMORED_VEHICLE,
    2: ARMORED_VEHICLE,
    4: ARMORED_VEHICLE,
    9: TANK,
}
ABOBA_DROP: set[int] = {0, 3, 5, 6, 7, 8, 10, 11}

VISDRONE_REMAP: dict[int, int] = {
    2: CIVILIAN_VEHICLE,
    3: CIVILIAN_VEHICLE,
    4: CIVILIAN_VEHICLE,
    5: CIVILIAN_VEHICLE,
}
VISDRONE_DROP: set[int] = {0, 1}

DATASET_SOURCES = {
    "russian_military": {
        "root": Path("datasets/russian_military/Russian-military-annotated"),
        "remap": RUSSIAN_MILITARY_REMAP,
        "drop": set(),
        "keep_all": True,
        "prefix": "rm",
    },
    "aboba": {
        "root": Path("datasets/aboba/aboba"),
        "remap": ABOBA_REMAP,
        "drop": ABOBA_DROP,
        "keep_all": True,
        "prefix": "ab",
    },
    "visdrone": {
        "root": Path("datasets/visdrone/VisDrone_Veri"),
        "remap": VISDRONE_REMAP,
        "drop": VISDRONE_DROP,
        "keep_all": False,
        "prefix": "vd",
        "civilian_target_min": 6000,
        "civilian_target_max": 8000,
    },
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
