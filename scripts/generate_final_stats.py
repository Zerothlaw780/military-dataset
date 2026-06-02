"""Generate final statistics report for merged dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dataset_config import FINAL_CLASS_NAMES
from yolo_io import parse_label_file


def compute_split_stats(split_dir: Path) -> dict:
    labels_dir = split_dir / "labels"
    inst: Counter = Counter()
    images_with_class: Counter = Counter()
    n_images = 0
    n_empty = 0

    if not labels_dir.is_dir():
        return {"images": 0, "instances": {}, "images_with_class": {}, "empty_labels": 0}

    for lbl in labels_dir.glob("*.txt"):
        n_images += 1
        boxes = parse_label_file(lbl)
        if not boxes:
            n_empty += 1
            continue
        present = {b.cls for b in boxes}
        for b in boxes:
            inst[b.cls] += 1
        for c in present:
            images_with_class[c] += 1

    return {
        "images": n_images,
        "instances": dict(inst),
        "images_with_class": dict(images_with_class),
        "empty_labels": n_empty,
    }


def write_final_report(merged_root: Path, report_path: Path, pipeline_meta: dict | None = None) -> None:
    lines = [
        "## Final merged dataset statistics",
        "",
        f"- **Dataset root**: `{merged_root.resolve()}`",
        f"- **Classes**: {FINAL_CLASS_NAMES}",
        "",
    ]
    totals_inst: Counter = Counter()
    totals_img = 0

    for split in ("train", "val", "test"):
        st = compute_split_stats(merged_root / split)
        lines.append(f"### {split}")
        lines.append(f"- **Images**: {st['images']}")
        lines.append(f"- **Empty label files**: {st['empty_labels']}")
        lines.append("")
        lines.append("| class | instances | images_with_class |")
        lines.append("|---|---:|---:|")
        for i, name in enumerate(FINAL_CLASS_NAMES):
            inst = st["instances"].get(i, 0)
            iwc = st["images_with_class"].get(i, 0)
            lines.append(f"| {name} | {inst} | {iwc} |")
            totals_inst[i] += inst
        lines.append("")
        totals_img += st["images"]

    lines.append("### TOTAL (all splits)")
    lines.append(f"- **Images**: {totals_img}")
    lines.append("")
    lines.append("| class | instances |")
    lines.append("|---|---:|")
    for i, name in enumerate(FINAL_CLASS_NAMES):
        lines.append(f"| {name} | {totals_inst[i]} |")
    lines.append("")

    if pipeline_meta:
        lines.append("### Pipeline metadata")
        for k, v in pipeline_meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
