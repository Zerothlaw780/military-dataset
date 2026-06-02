#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclasses.dataclass(frozen=True)
class YoloBox:
    cls: int
    x: float
    y: float
    w: float
    h: float

    def is_normalized(self) -> bool:
        return (
            0.0 <= self.x <= 1.0
            and 0.0 <= self.y <= 1.0
            and 0.0 <= self.w <= 1.0
            and 0.0 <= self.h <= 1.0
        )

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def aspect_ratio(self) -> float:
        if self.h <= 0:
            return math.inf
        return self.w / self.h


def _sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _iter_image_files(images_dir: Path) -> Iterable[Path]:
    if not images_dir.exists():
        return []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _corresponding_label_path(labels_dir: Path, image_path: Path, images_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def _parse_yolo_label_file(label_path: Path) -> tuple[list[YoloBox], list[str]]:
    boxes: list[YoloBox] = []
    errors: list[str] = []
    try:
        raw = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return [], [f"read_error:{type(e).__name__}"]

    for i, line in enumerate(raw, start=1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            errors.append(f"line_{i}:too_few_columns")
            continue
        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            errors.append(f"line_{i}:parse_error")
            continue
        boxes.append(YoloBox(cls=cls, x=x, y=y, w=w, h=h))
    return boxes, errors


def _percentiles(values: list[float], ps: list[int]) -> dict[str, float]:
    if not values:
        return {f"p{p}": float("nan") for p in ps}
    xs = sorted(values)
    n = len(xs)
    out: dict[str, float] = {}
    for p in ps:
        if n == 1:
            out[f"p{p}"] = xs[0]
            continue
        k = (p / 100) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[f"p{p}"] = xs[int(k)]
        else:
            out[f"p{p}"] = xs[f] * (c - k) + xs[c] * (k - f)
    return out


def _read_names_from_yaml(dataset_root: Path) -> dict:
    """
    Best-effort extraction of `nc` and `names` from a YOLO data yaml.
    Implemented without external deps to keep this repo lightweight.
    """
    yaml_candidates = [
        dataset_root / "data.yaml",
        dataset_root / "dataset.yaml",
        dataset_root / "visdrone.yaml",
    ]
    yaml_candidates.extend(sorted(dataset_root.glob("*.yaml")))
    yaml_candidates.extend(sorted(dataset_root.glob("*.yml")))

    for p in yaml_candidates:
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Very small parser: expects `nc:` and `names:` on one line (as in Roboflow exports).
        nc = None
        names = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("nc:"):
                try:
                    nc = int(s.split(":", 1)[1].strip())
                except Exception:
                    pass
            if s.startswith("names:"):
                rhs = s.split(":", 1)[1].strip()
                if rhs.startswith("[") and rhs.endswith("]"):
                    body = rhs[1:-1].strip()
                    parts = [x.strip() for x in body.split(",")] if body else []
                    cleaned: list[str] = []
                    for item in parts:
                        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                            cleaned.append(item[1:-1])
                        else:
                            cleaned.append(item)
                    names = cleaned

        if nc is not None or names is not None:
            return {"yaml_path": str(p), "nc": nc, "names": names}
    return {}


def _discover_splits(root: Path) -> dict[str, tuple[Path, Path]]:
    """
    Returns split -> (images_dir, labels_dir).
    Supports:
      - Roboflow layout: train|valid|test/{images,labels}
      - VisDrone YAML layout already materialized similarly
    """
    candidates = {}
    for split in ("train", "valid", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if images_dir.exists() and labels_dir.exists():
            key = "valid" if split == "val" else split
            candidates[key] = (images_dir, labels_dir)
    # Some datasets place train/val directly under root (VisDrone)
    # e.g. root/VisDrone2019-DET-train/{images,labels}
    for p in root.iterdir() if root.exists() else []:
        if not p.is_dir():
            continue
        name = p.name.lower()
        if "train" in name and (p / "images").exists() and (p / "labels").exists():
            candidates.setdefault("train", (p / "images", p / "labels"))
        if ("val" in name or "valid" in name) and (p / "images").exists() and (p / "labels").exists():
            candidates.setdefault("valid", (p / "images", p / "labels"))
        if "test" in name and (p / "images").exists() and (p / "labels").exists():
            candidates.setdefault("test", (p / "images", p / "labels"))
    return candidates


def analyze_dataset(name: str, root: Path, *, hash_images: bool) -> dict:
    splits = _discover_splits(root)
    if not splits:
        return {"name": name, "root": str(root), "error": "no_splits_found"}

    yaml_info = _read_names_from_yaml(root)

    summary = {
        "name": name,
        "root": str(root),
        "yolo_yaml": yaml_info,
        "splits": {},
        "totals": {},
        "label_health": {},
        "classes": {},
        "bboxes": {},
        "duplicates": {},
    }

    cls_counts_total: Counter[int] = Counter()
    img_hashes: dict[str, list[str]] = defaultdict(list)  # sha1 -> [relpaths]
    label_hashes: dict[str, list[str]] = defaultdict(list)

    bbox_areas: list[float] = []
    bbox_ar: list[float] = []
    invalid_boxes = 0
    negative_cls = 0
    missing_labels = 0
    empty_labels = 0
    parse_error_lines = 0
    total_images = 0
    total_labels_present = 0
    total_boxes = 0

    for split, (images_dir, labels_dir) in splits.items():
        split_images = list(_iter_image_files(images_dir))
        total_images += len(split_images)

        split_stats = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "num_images": len(split_images),
            "num_label_files_present": 0,
            "num_images_missing_label_file": 0,
            "num_empty_label_files": 0,
            "num_boxes": 0,
            "class_counts": {},
        }
        split_cls_counts: Counter[int] = Counter()

        for img_path in split_images:
            lbl_path = _corresponding_label_path(labels_dir, img_path, images_dir)
            if not lbl_path.exists():
                missing_labels += 1
                split_stats["num_images_missing_label_file"] += 1
                continue

            total_labels_present += 1
            split_stats["num_label_files_present"] += 1

            boxes, errors = _parse_yolo_label_file(lbl_path)
            if not boxes and not errors:
                empty_labels += 1
                split_stats["num_empty_label_files"] += 1

            if errors:
                for e in errors:
                    if "too_few_columns" in e or "parse_error" in e or "read_error" in e:
                        parse_error_lines += 1

            for b in boxes:
                total_boxes += 1
                split_stats["num_boxes"] += 1
                if b.cls < 0:
                    negative_cls += 1
                    continue
                if not b.is_normalized() or b.w <= 0 or b.h <= 0:
                    invalid_boxes += 1
                    continue
                split_cls_counts[b.cls] += 1
                cls_counts_total[b.cls] += 1
                bbox_areas.append(b.area())
                bbox_ar.append(b.aspect_ratio())

            # label duplicate signal: identical label files (common after copy/rename)
            try:
                h_lbl = _sha1_file(lbl_path)
                label_hashes[h_lbl].append(str(lbl_path.relative_to(root)))
            except Exception:
                pass

            if hash_images:
                try:
                    h_img = _sha1_file(img_path)
                    img_hashes[h_img].append(str(img_path.relative_to(root)))
                except Exception:
                    pass

        split_stats["class_counts"] = dict(split_cls_counts.most_common())
        summary["splits"][split] = split_stats

    summary["totals"] = {
        "num_images": total_images,
        "num_label_files_present": total_labels_present,
        "num_images_missing_label_file": missing_labels,
        "num_boxes": total_boxes,
    }

    summary["label_health"] = {
        "missing_label_files": missing_labels,
        "empty_label_files": empty_labels,
        "invalid_boxes": invalid_boxes,
        "negative_class_ids": negative_cls,
        "parse_error_lines": parse_error_lines,
    }

    summary["classes"] = {
        "observed_class_ids": sorted(cls_counts_total.keys()),
        "class_counts": dict(cls_counts_total.most_common()),
        "num_observed_classes": len(cls_counts_total),
    }

    summary["bboxes"] = {
        "area_percentiles": _percentiles(bbox_areas, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
        "aspect_ratio_percentiles": _percentiles(bbox_ar, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
        "num_valid_boxes_used_for_stats": len(bbox_areas),
    }

    dup_labels = {h: paths for h, paths in label_hashes.items() if len(paths) > 1}
    dup_imgs = {h: paths for h, paths in img_hashes.items() if len(paths) > 1} if hash_images else {}

    # Provide a few "largest duplicate groups" for investigation.
    largest_label_dup_groups = sorted(dup_labels.values(), key=len, reverse=True)[:5]
    largest_img_dup_groups = sorted(dup_imgs.values(), key=len, reverse=True)[:5]

    summary["duplicates"] = {
        "label_file_duplicate_groups": len(dup_labels),
        "label_file_duplicates_total": sum(len(v) for v in dup_labels.values()),
        "image_duplicate_groups": len(dup_imgs),
        "image_duplicates_total": sum(len(v) for v in dup_imgs.values()),
        "largest_label_duplicate_groups_samples": largest_label_dup_groups,
        "largest_image_duplicate_groups_samples": largest_img_dup_groups,
        "note": "Image duplicates are only computed when --hash-images is enabled.",
    }

    return summary


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def write_markdown_report(out_path: Path, summaries: list[dict]) -> None:
    lines: list[str] = []
    lines.append("## Dataset Analysis Report")
    lines.append("")
    lines.append("This report summarizes dataset structure, label health, class distributions, and basic bounding-box statistics.")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s.get('name','(unknown)')}")
        lines.append("")
        if "error" in s:
            lines.append(f"**Error**: {_md_escape(str(s['error']))}")
            lines.append("")
            continue

        yaml_info = s.get("yolo_yaml") or {}
        totals = s["totals"]
        health = s["label_health"]
        classes = s["classes"]
        bboxes = s["bboxes"]
        dups = s["duplicates"]

        lines.append(f"- **Root**: `{s['root']}`")
        if yaml_info:
            lines.append(f"- **YAML**: `{yaml_info.get('yaml_path','')}`")
            if yaml_info.get("nc") is not None:
                lines.append(f"- **YAML nc**: {yaml_info['nc']}")
            if yaml_info.get("names"):
                lines.append(f"- **YAML names**: {yaml_info['names']}")
        lines.append(f"- **Images**: {totals['num_images']}")
        lines.append(f"- **Label files present**: {totals['num_label_files_present']}")
        lines.append(f"- **Images missing label file**: {totals['num_images_missing_label_file']}")
        lines.append(f"- **Total boxes (raw)**: {totals['num_boxes']}")
        lines.append("")

        lines.append("### Splits")
        for split, st in s["splits"].items():
            lines.append(f"- **{split}**: {st['num_images']} images, {st['num_label_files_present']} label files, {st['num_boxes']} boxes")
        lines.append("")

        lines.append("### Label health")
        lines.append(f"- **Empty label files**: {health['empty_label_files']}")
        lines.append(f"- **Invalid boxes** (out of range / non-positive w/h): {health['invalid_boxes']}")
        lines.append(f"- **Negative class ids**: {health['negative_class_ids']}")
        lines.append(f"- **Parse error lines**: {health['parse_error_lines']}")
        lines.append("")

        lines.append("### Observed classes (by id)")
        lines.append(f"- **Observed class ids**: {classes['observed_class_ids']}")
        lines.append(f"- **Num observed classes**: {classes['num_observed_classes']}")
        lines.append("")
        if classes["class_counts"]:
            lines.append("| class_id | count |")
            lines.append("|---:|---:|")
            for cid, cnt in classes["class_counts"].items():
                lines.append(f"| {cid} | {cnt} |")
            lines.append("")

        lines.append("### Bounding boxes (normalized YOLO, valid boxes only)")
        lines.append(f"- **Valid boxes used**: {bboxes['num_valid_boxes_used_for_stats']}")
        ap = bboxes["area_percentiles"]
        rp = bboxes["aspect_ratio_percentiles"]
        lines.append("")
        lines.append("| metric | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            "| area (w*h) | "
            + " | ".join(f"{ap[f'p{p}']:.6g}" if not math.isnan(ap[f"p{p}"]) else "NA" for p in [1, 5, 10, 25, 50, 75, 90, 95, 99])
            + " |"
        )
        lines.append(
            "| aspect ratio (w/h) | "
            + " | ".join(f"{rp[f'p{p}']:.6g}" if not math.isnan(rp[f"p{p}"]) else "NA" for p in [1, 5, 10, 25, 50, 75, 90, 95, 99])
            + " |"
        )
        lines.append("")

        lines.append("### Duplicate signals")
        lines.append(f"- **Duplicate label-file groups** (identical content): {dups['label_file_duplicate_groups']} (total files in groups: {dups['label_file_duplicates_total']})")
        lines.append(f"- **Duplicate image groups** (identical bytes): {dups['image_duplicate_groups']} (total files in groups: {dups['image_duplicates_total']})")
        if dups.get("largest_label_duplicate_groups_samples"):
            lines.append("")
            lines.append("- **Largest label-duplicate groups (samples)**:")
            for group in dups["largest_label_duplicate_groups_samples"]:
                sample = ", ".join(f"`{_md_escape(p)}`" for p in group[:3])
                more = f" (+{len(group) - 3} more)" if len(group) > 3 else ""
                lines.append(f"  - {len(group)} files: {sample}{more}")
        lines.append(f"- **Note**: {dups['note']}")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze YOLO-format datasets and write a detailed report.")
    ap.add_argument("--out-dir", default="reports", help="Output directory for report artifacts.")
    ap.add_argument(
        "--hash-images",
        action="store_true",
        help="Compute SHA1 hashes for all images to detect exact duplicates (can be slow).",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Russian Military", repo / "datasets" / "russian_military" / "Russian-military-annotated"),
        ("Aboba", repo / "datasets" / "aboba" / "aboba"),
        ("VisDrone", repo / "datasets" / "visdrone" / "VisDrone_Veri"),
    ]

    summaries = [analyze_dataset(name, root, hash_images=args.hash_images) for name, root in datasets]

    (out_dir / "dataset_analysis.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_markdown_report(out_dir / "dataset_report.md", summaries)

    print(f"Wrote: {out_dir / 'dataset_report.md'}")
    print(f"Wrote: {out_dir / 'dataset_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

