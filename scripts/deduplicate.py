#!/usr/bin/env python3
"""Remove near-duplicate hard-case crops using perceptual hashing.

Drone footage produces many almost-identical crops of the same vehicle across
consecutive frames. This collapses those into a single representative per
near-duplicate cluster so the review grid (and any future re-labeling) stays
diverse.

Example:
    python scripts/deduplicate.py \
        --crops hardcase_mining/crops \
        --metadata hardcase_mining/metadata.csv \
        --threshold 5
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _import_deps():
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        _fail("Pillow and imagehash are required. Install with: pip install Pillow imagehash")
    import imagehash
    from PIL import Image

    return imagehash, Image


def load_confidences(metadata_path: Path | None) -> dict[str, float]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    conf: dict[str, float] = {}
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("crop_file")
            if not name:
                continue
            try:
                conf[name] = float(row.get("confidence", 0.0) or 0.0)
            except ValueError:
                conf[name] = 0.0
    return conf


def cluster_crops(
    crop_paths: list[Path],
    *,
    imagehash,
    Image,
    threshold: int,
    hash_size: int,
) -> list[list[Path]]:
    """Greedy clustering: assign each crop to the first cluster whose
    representative hash is within ``threshold`` Hamming distance."""
    clusters: list[list[Path]] = []
    cluster_hashes: list = []

    for p in crop_paths:
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
        except Exception as e:  # unreadable/corrupt crop -> keep as its own cluster
            print(f"  WARNING: could not hash {p.name}: {e}")
            clusters.append([p])
            cluster_hashes.append(None)
            continue

        placed = False
        for i, ch in enumerate(cluster_hashes):
            if ch is not None and (h - ch) <= threshold:
                clusters[i].append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
            cluster_hashes.append(h)
    return clusters


def pick_representative(group: list[Path], confidences: dict[str, float]) -> Path:
    """Highest-confidence crop wins; ties break on larger file then name."""
    def key(p: Path):
        return (confidences.get(p.name, -1.0), p.stat().st_size, p.name)

    return max(group, key=key)


def main() -> int:
    ap = argparse.ArgumentParser(description="Deduplicate hard-case crops via perceptual hashing.")
    ap.add_argument("--crops", required=True, type=Path, help="Folder of crop images.")
    ap.add_argument("--metadata", type=Path, default=None, help="Optional metadata.csv (used for ranking + filtered output).")
    ap.add_argument("--threshold", type=int, default=5, help="Max Hamming distance for near-duplicates.")
    ap.add_argument("--hash-size", type=int, default=8, help="pHash size (8 -> 64-bit hash).")
    ap.add_argument(
        "--action",
        choices=("move", "delete", "none"),
        default="move",
        help="What to do with duplicates: move aside, delete, or report only.",
    )
    ap.add_argument(
        "--dup-dir",
        type=Path,
        default=None,
        help="Destination for duplicates when --action move (default: <crops>/../crops_duplicates).",
    )
    args = ap.parse_args()

    if not args.crops.is_dir():
        _fail(f"crops folder not found: {args.crops}")

    imagehash, Image = _import_deps()

    crop_paths = sorted(p for p in args.crops.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not crop_paths:
        _fail(f"no crop images found in {args.crops}")

    confidences = load_confidences(args.metadata)

    print(f"Hashing {len(crop_paths)} crop(s) (threshold={args.threshold})...")
    clusters = cluster_crops(
        crop_paths,
        imagehash=imagehash,
        Image=Image,
        threshold=args.threshold,
        hash_size=args.hash_size,
    )

    kept: list[Path] = []
    duplicates: list[Path] = []
    for group in clusters:
        rep = pick_representative(group, confidences)
        kept.append(rep)
        duplicates.extend(p for p in group if p != rep)

    print(f"Clusters: {len(clusters)} | kept: {len(kept)} | duplicates: {len(duplicates)}")

    if duplicates and args.action == "move":
        dup_dir = args.dup_dir or (args.crops.parent / "crops_duplicates")
        dup_dir.mkdir(parents=True, exist_ok=True)
        for p in duplicates:
            shutil.move(str(p), str(dup_dir / p.name))
        print(f"Moved {len(duplicates)} duplicate(s) -> {dup_dir}")
    elif duplicates and args.action == "delete":
        for p in duplicates:
            p.unlink(missing_ok=True)
        print(f"Deleted {len(duplicates)} duplicate(s)")
    elif duplicates:
        print("Reporting only (no files changed). Use --action move|delete to apply.")

    # Write a filtered metadata file alongside the originals.
    if args.metadata and args.metadata.exists():
        kept_names = {p.name for p in kept}
        out_csv = args.metadata.with_name(args.metadata.stem + "_dedup.csv")
        with args.metadata.open("r", newline="", encoding="utf-8") as fin, out_csv.open(
            "w", newline="", encoding="utf-8"
        ) as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                if row.get("crop_file") in kept_names:
                    writer.writerow(row)
        print(f"Deduplicated metadata -> {out_csv}")

    print("Next: python scripts/create_grid.py --crops {} --metadata {}".format(
        args.crops,
        (args.metadata.with_name(args.metadata.stem + "_dedup.csv") if args.metadata else "''"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
