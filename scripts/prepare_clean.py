"""Clean, remap, deduplicate, and stage a single source dataset."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dataset_config import IMG_EXTS
from phash_dedup import cluster_by_phash, pick_representatives
from yolo_io import YoloBox, discover_label_image_pairs, parse_label_file, remap_boxes, write_label_file


@dataclass
class SampleRecord:
    label_src: Path
    image_src: Path
    split: str
    boxes: list[YoloBox]
    vehicle_type_mask: int = 0


@dataclass
class CleanStats:
    pairs_seen: int = 0
    empty_labels_removed: int = 0
    no_boxes_after_remap: int = 0
    invalid_boxes_dropped: int = 0
    duplicates_removed: int = 0
    subset_removed: int = 0
    kept: int = 0


def _vehicle_mask(raw_boxes) -> int:
    mask = 0
    for b in raw_boxes:
        if b.cls in (2, 3, 4, 5):
            mask |= 1 << (b.cls - 2)
    return mask


def build_records(root: Path, remap: dict[int, int], drop: set[int]) -> tuple[list[SampleRecord], CleanStats]:
    stats = CleanStats()
    records: list[SampleRecord] = []

    for label_path, image_path, split in discover_label_image_pairs(root):
        stats.pairs_seen += 1
        raw = parse_label_file(label_path)
        if not raw:
            stats.empty_labels_removed += 1
            continue

        before_valid = len(raw)
        remapped = remap_boxes(raw, remap, drop)
        stats.invalid_boxes_dropped += before_valid - len(remapped) - sum(1 for b in raw if b.cls in drop)

        if not remapped:
            stats.no_boxes_after_remap += 1
            continue

        records.append(
            SampleRecord(
                label_src=label_path,
                image_src=image_path,
                split=split,
                boxes=remapped,
                vehicle_type_mask=_vehicle_mask(raw),
            )
        )
    return records, stats


def dedupe_records(records: list[SampleRecord], stats: CleanStats, *, max_hamming: int = 8) -> list[SampleRecord]:
    path_to_rec = {r.image_src: r for r in records}

    def score(p: Path) -> float:
        r = path_to_rec[p]
        return len(r.boxes) + 2.0 * bin(r.vehicle_type_mask).count("1")

    clusters = cluster_by_phash([r.image_src for r in records], max_hamming=max_hamming)
    kept_paths = set(pick_representatives(clusters, score))
    kept = [r for r in records if r.image_src in kept_paths]
    stats.duplicates_removed = len(records) - len(kept)
    return kept


def select_visdrone_subset(
    records: list[SampleRecord],
    *,
    target_min: int = 6000,
    target_max: int = 8000,
) -> list[SampleRecord]:
    if not records:
        return []

    def score(r: SampleRecord) -> float:
        types = bin(r.vehicle_type_mask).count("1")
        n = len(r.boxes)
        return types * 100.0 - abs(n - 15) * 0.3

    ordered = sorted(records, key=score, reverse=True)
    selected: list[SampleRecord] = []
    total = 0

    for r in ordered:
        n = len(r.boxes)
        if total >= target_min and total + n > target_max:
            continue
        if total + n > target_max and total >= target_min:
            break
        selected.append(r)
        total += n
        if total >= target_max:
            break

    if total < target_min:
        rest = sorted([r for r in ordered if r not in selected], key=lambda x: len(x.boxes))
        for r in rest:
            n = len(r.boxes)
            if total + n > target_max:
                continue
            selected.append(r)
            total += n
            if total >= target_min:
                break
    return selected


def write_staging(records: list[SampleRecord], staging_dir: Path, prefix: str) -> Counter:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    images_dir = staging_dir / "images"
    labels_dir = staging_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    inst = Counter()
    manifest: list[dict] = []

    for i, r in enumerate(records):
        ext = r.image_src.suffix.lower()
        if ext not in IMG_EXTS:
            ext = ".jpg"
        stem = f"{prefix}_{r.image_src.stem}_{i:06d}"
        out_img = images_dir / f"{stem}{ext}"
        out_lbl = labels_dir / f"{stem}.txt"
        shutil.copy2(r.image_src, out_img)
        write_label_file(out_lbl, r.boxes)
        for b in r.boxes:
            inst[b.cls] += 1
        manifest.append(
            {
                "image": out_img.name,
                "label": out_lbl.name,
                "source_image": str(r.image_src),
                "instances": len(r.boxes),
            }
        )

    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return inst


def clean_source(
    root: Path,
    staging_dir: Path,
    remap: dict[int, int],
    drop: set[int],
    prefix: str,
    *,
    keep_all: bool = True,
    civilian_target_min: int = 6000,
    civilian_target_max: int = 8000,
    max_hamming: int = 8,
) -> tuple[CleanStats, Counter]:
    records, stats = build_records(root, remap, drop)
    records = dedupe_records(records, stats, max_hamming=max_hamming)

    if not keep_all:
        before = len(records)
        records = select_visdrone_subset(
            records,
            target_min=civilian_target_min,
            target_max=civilian_target_max,
        )
        stats.subset_removed = before - len(records)

    stats.kept = len(records)
    inst = write_staging(records, staging_dir, prefix)
    return stats, inst
