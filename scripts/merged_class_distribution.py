#!/usr/bin/env python3
"""Class distribution report (markdown + CSV) for the merged 3-class dataset."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from dataset_config import FINAL_CLASS_NAMES
from yolo_io import parse_label_file

SPLITS = ("train", "val", "test")
SOURCE_PREFIX = {"rm": "russian_military", "ab": "aboba", "vd": "visdrone"}


def _source_of(filename: str) -> str:
    return SOURCE_PREFIX.get(filename.split("_", 1)[0], "unknown")


def collect(merged_root: Path) -> dict:
    per_split_inst: dict[str, Counter] = {s: Counter() for s in SPLITS}
    per_split_imgs: dict[str, Counter] = {s: Counter() for s in SPLITS}
    per_split_count: dict[str, int] = {s: 0 for s in SPLITS}
    source_inst: dict[str, Counter] = defaultdict(Counter)

    for split in SPLITS:
        labels_dir = merged_root / split / "labels"
        if not labels_dir.is_dir():
            continue
        for lbl in labels_dir.glob("*.txt"):
            per_split_count[split] += 1
            boxes = parse_label_file(lbl)
            present = set()
            src = _source_of(lbl.name)
            for b in boxes:
                per_split_inst[split][b.cls] += 1
                source_inst[src][b.cls] += 1
                present.add(b.cls)
            for c in present:
                per_split_imgs[split][c] += 1

    return {
        "per_split_inst": per_split_inst,
        "per_split_imgs": per_split_imgs,
        "per_split_count": per_split_count,
        "source_inst": source_inst,
    }


def write_markdown(data: dict, out_path: Path) -> None:
    psi = data["per_split_inst"]
    psm = data["per_split_imgs"]
    psc = data["per_split_count"]
    src = data["source_inst"]

    totals = Counter()
    for s in SPLITS:
        totals.update(psi[s])

    lines = ["## Merged dataset class distribution", ""]
    lines.append(f"- **Classes**: {FINAL_CLASS_NAMES}")
    lines.append(f"- **Total images**: {sum(psc.values())}")
    lines.append("")

    lines.append("### Totals (all splits)")
    lines.append("")
    lines.append("| class_id | class | instances |")
    lines.append("|---:|---|---:|")
    for i, name in enumerate(FINAL_CLASS_NAMES):
        lines.append(f"| {i} | {name} | {totals.get(i, 0)} |")
    lines.append("")

    for split in SPLITS:
        lines.append(f"### {split} ({psc[split]} images)")
        lines.append("")
        lines.append("| class_id | class | instances | images_with_class |")
        lines.append("|---:|---|---:|---:|")
        for i, name in enumerate(FINAL_CLASS_NAMES):
            lines.append(f"| {i} | {name} | {psi[split].get(i, 0)} | {psm[split].get(i, 0)} |")
        lines.append("")

    lines.append("### Instances by source dataset")
    lines.append("")
    lines.append("| source | " + " | ".join(FINAL_CLASS_NAMES) + " | total |")
    lines.append("|---|" + "---:|" * (len(FINAL_CLASS_NAMES) + 1))
    for source in ("russian_military", "aboba", "visdrone"):
        counts = src.get(source, Counter())
        row = [str(counts.get(i, 0)) for i in range(len(FINAL_CLASS_NAMES))]
        lines.append(f"| {source} | " + " | ".join(row) + f" | {sum(counts.values())} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(data: dict, out_path: Path) -> None:
    psi = data["per_split_inst"]
    psm = data["per_split_imgs"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class_id", "class_name", "instances", "images_with_class"])
        for split in SPLITS:
            for i, name in enumerate(FINAL_CLASS_NAMES):
                w.writerow([split, i, name, psi[split].get(i, 0), psm[split].get(i, 0)])
        totals = Counter()
        for s in SPLITS:
            totals.update(psi[s])
        for i, name in enumerate(FINAL_CLASS_NAMES):
            w.writerow(["TOTAL", i, name, totals.get(i, 0), ""])


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    merged = repo / "merged_dataset"
    data = collect(merged)
    md = repo / "reports" / "merged_class_distribution.md"
    csv_path = repo / "reports" / "merged_class_distribution.csv"
    write_markdown(data, md)
    write_csv(data, csv_path)
    print(f"Wrote {md}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
