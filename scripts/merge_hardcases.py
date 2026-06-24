#!/usr/bin/env python3
"""Merge an exported hard-case dataset into the main merged_dataset.

Takes the YOLO dataset produced by ``export_reviewed_hardcases.py``
(``<export>/images`` + ``<export>/labels``) and copies every image/label pair
into ``merged_dataset/train`` so the reviewed armored/tank crops join the
training split.

Filenames are prefixed and de-duplicated so newly added crops can never
overwrite existing training files.

Class ids follow the merged dataset (0 = tank, 1 = armored_vehicle).

Example:
    python scripts/merge_hardcases.py \
        --export hardcase_mining/export \
        --merged merged_dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "tank", 1: "armored_vehicle", 2: "civilian_vehicle"}


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def label_class_ids(label_path: Path) -> list[int]:
    ids: list[int] = []
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(int(float(line.split()[0])))
            except (ValueError, IndexError):
                continue
    except OSError:
        pass
    return ids


def unique_stem(stem: str, images_dir: Path, labels_dir: Path) -> str:
    """Return a stem guaranteed not to collide with existing image/label files."""
    def taken(s: str) -> bool:
        if (labels_dir / f"{s}.txt").exists():
            return True
        return any((images_dir / f"{s}{ext}").exists() for ext in IMG_EXTS)

    if not taken(stem):
        return stem
    i = 1
    while taken(f"{stem}_{i}"):
        i += 1
    return f"{stem}_{i}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge exported hard-case crops into merged_dataset/train.")
    ap.add_argument("--export", type=Path, default=Path("hardcase_mining/export"), help="Hard-case export folder (images/ + labels/).")
    ap.add_argument("--merged", type=Path, default=Path("merged_dataset"), help="Target merged dataset root.")
    ap.add_argument("--split", default="train", help="Destination split (default: train).")
    ap.add_argument("--prefix", default="hc_", help="Filename prefix for added crops (collision safety).")
    args = ap.parse_args()

    src_images = args.export / "images"
    src_labels = args.export / "labels"
    if not src_images.is_dir():
        _fail(f"export images folder not found: {src_images}")
    if not src_labels.is_dir():
        _fail(f"export labels folder not found: {src_labels}")

    dst_images = args.merged / args.split / "images"
    dst_labels = args.merged / args.split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        _fail(f"no images found in {src_images}")

    added = {"tank": 0, "armored_vehicle": 0}
    skipped_no_label = 0
    other_class = 0
    total = 0

    for img in images:
        label = src_labels / f"{img.stem}.txt"
        if not label.exists():
            skipped_no_label += 1
            continue

        ids = label_class_ids(label)
        prefixed = f"{args.prefix}{img.stem}"
        stem = unique_stem(prefixed, dst_images, dst_labels)

        shutil.copy2(img, dst_images / f"{stem}{img.suffix.lower()}")
        shutil.copy2(label, dst_labels / f"{stem}.txt")
        total += 1

        for cid in ids:
            name = CLASS_NAMES.get(cid)
            if name in added:
                added[name] += 1
            else:
                other_class += 1

    print("\n===== Hard-case merge summary =====")
    print(f"  Added tank instances:     {added['tank']}")
    print(f"  Added armored instances:  {added['armored_vehicle']}")
    print(f"  Total images added:       {total}")
    if skipped_no_label:
        print(f"  Skipped (no label file):  {skipped_no_label}")
    if other_class:
        print(f"  Other-class instances:    {other_class}")
    print(f"  Images -> {dst_images}")
    print(f"  Labels -> {dst_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
