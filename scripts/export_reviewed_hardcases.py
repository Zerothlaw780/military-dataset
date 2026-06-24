#!/usr/bin/env python3
"""Export reviewed hard-case crops into a YOLO-format dataset.

Reads the decisions produced by ``review_hardcases.py`` (``review.csv``) and
turns every crop confirmed as armored or tank into a ready-to-train YOLO
sample. Because each crop is a tight box around a single vehicle, the whole
image is the object, so every label is a full-frame box centered at
(0.5, 0.5) with width/height 1.0.

Class mapping (matches the project's final classes):
    tank            -> 0
    armored_vehicle -> 1

Inputs:
    review.csv        decisions (crop_file, decision, ...)
    crops/            crops left in place (e.g. civilian picks)
    hardcase_armored/ crops the reviewer MOVED here when marking armored
    metadata.csv      optional; only used to locate crops if present

Outputs:
    export/images/    copied crop images
    export/labels/    one YOLO .txt per image

Example:
    python scripts/export_reviewed_hardcases.py \
        --crops hardcase_mining/crops \
        --review-csv hardcase_mining/review.csv \
        --armored-dir hardcase_mining/hardcase_armored \
        --output hardcase_mining/export
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# decision string (lowercased) -> (class_id, class_name)
TANK = (0, "tank")
ARMORED = (1, "armored_vehicle")
DECISION_TO_CLASS: dict[str, tuple[int, str]] = {
    "tank": TANK,
    "t": TANK,
    "0": TANK,
    "armored_vehicle": ARMORED,
    "armored": ARMORED,
    "a": ARMORED,
    "1": ARMORED,
}

# Full-image box: the crop IS the object.
FULL_BOX = (0.5, 0.5, 1.0, 1.0)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_review_rows(review_csv: Path) -> list[dict]:
    if not review_csv.exists():
        _fail(f"review.csv not found: {review_csv}")
    with review_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_crop(name: str, search_dirs: list[Path]) -> Path | None:
    """Locate a crop image by filename across the candidate directories."""
    for d in search_dirs:
        cand = d / name
        if cand.is_file():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Export reviewed armored/tank crops to a YOLO dataset.")
    ap.add_argument("--crops", type=Path, default=Path("hardcase_mining/crops"), help="Folder of crops.")
    ap.add_argument("--review-csv", type=Path, default=None, help="Decisions file (default: <crops>/../review.csv).")
    ap.add_argument("--armored-dir", type=Path, default=None, help="Where armored picks were moved (default: <crops>/../hardcase_armored).")
    ap.add_argument("--tank-dir", type=Path, default=None, help="Optional folder where tank picks were moved.")
    ap.add_argument("--metadata", type=Path, default=None, help="Optional metadata CSV (used only to locate crops).")
    ap.add_argument("--output", type=Path, default=None, help="Output dataset root (default: <crops>/../export).")
    args = ap.parse_args()

    base = args.crops.parent
    review_csv = args.review_csv or (base / "review.csv")
    armored_dir = args.armored_dir or (base / "hardcase_armored")
    tank_dir = args.tank_dir or (base / "hardcase_tank")
    output_root = args.output or (base / "export")

    # Candidate directories to find each crop file, in priority order.
    search_dirs: list[Path] = []
    for d in (args.crops, armored_dir, tank_dir, base):
        if d.is_dir() and d not in search_dirs:
            search_dirs.append(d)

    rows = read_review_rows(review_csv)

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"tank": 0, "armored_vehicle": 0}
    missing: list[str] = []
    skipped_other = 0
    seen: set[str] = set()

    for row in rows:
        name = (row.get("crop_file") or "").strip()
        decision = (row.get("decision") or "").strip().lower()
        if not name:
            continue
        target = DECISION_TO_CLASS.get(decision)
        if target is None:
            skipped_other += 1
            continue
        if name in seen:
            continue
        seen.add(name)

        src = find_crop(name, search_dirs)
        if src is None:
            missing.append(name)
            continue

        cls_id, cls_name = target
        dst_img = images_dir / src.name
        shutil.copy2(src, dst_img)

        cx, cy, w, h = FULL_BOX
        label_path = labels_dir / f"{src.stem}.txt"
        label_path.write_text(f"{cls_id} {cx} {cy} {w} {h}\n", encoding="utf-8")

        counts[cls_name] += 1

    total = counts["tank"] + counts["armored_vehicle"]
    print("\n===== Export summary =====")
    print(f"  tank (class 0):            {counts['tank']}")
    print(f"  armored_vehicle (class 1): {counts['armored_vehicle']}")
    print(f"  total exported:            {total}")
    if skipped_other:
        print(f"  skipped (civilian/delete/other decisions): {skipped_other}")
    if missing:
        print(f"  WARNING: {len(missing)} crop(s) not found on disk (searched {', '.join(str(d) for d in search_dirs)}):")
        for n in missing[:10]:
            print(f"    - {n}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    print(f"  Images -> {images_dir}")
    print(f"  Labels -> {labels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
