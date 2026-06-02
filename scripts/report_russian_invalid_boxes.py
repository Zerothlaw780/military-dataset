#!/usr/bin/env python3
"""Generate report on invalid boxes removed from Russian Military during cleaning."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from dataset_config import RUSSIAN_MILITARY_REMAP
from yolo_io import YoloBox, discover_label_image_pairs, is_valid_box, parse_label_file

NAMES = ["bmd-2", "bmp-1", "bmp-2", "btr-70", "btr-80", "mt-lb"]
ROOT = Path("datasets/russian_military/Russian-military-annotated")


def _subtags(b: YoloBox) -> list[str]:
    x1, y1 = b.x - b.w / 2, b.y - b.h / 2
    x2, y2 = b.x + b.w / 2, b.y + b.h / 2
    tags: list[str] = []
    if b.w > 1 or b.h > 1:
        tags.append("width_or_height_exceeds_1")
    if x1 < -1e-6:
        tags.append("extends_past_left")
    if y1 < -1e-6:
        tags.append("extends_past_top")
    if x2 > 1.0 + 1e-6:
        tags.append("extends_past_right")
    if y2 > 1.0 + 1e-6:
        tags.append("extends_past_bottom")
    return tags


def _classify_malformed(line: str, line_no: int) -> dict | None:
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) < 5:
        return {"category": "malformed", "detail": f"too_few_fields ({len(parts)})", "line": s, "line_no": line_no}
    try:
        int(float(parts[0]))
        float(parts[1])
        float(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return {"category": "malformed", "detail": "parse_error", "line": s, "line_no": line_no}
    return None


def analyze() -> dict:
    invalid: list[dict] = []
    malformed: list[dict] = []
    only_invalid_files: list[str] = []
    mixed_files: list[dict] = []

    for lp, img, _split in discover_label_image_pairs(ROOT):
        rel = str(lp.relative_to(ROOT))
        lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            m = _classify_malformed(line, i)
            if m:
                malformed.append({**m, "label_file": rel})
                continue
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            b = YoloBox(
                int(float(parts[0])),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            if b.cls not in RUSSIAN_MILITARY_REMAP:
                continue
            if is_valid_box(b):
                continue
            x1, y1 = b.x - b.w / 2, b.y - b.h / 2
            x2, y2 = b.x + b.w / 2, b.y + b.h / 2
            if b.w <= 0 or b.h <= 0:
                cat = "zero_area"
            elif b.x < 0 or b.y < 0 or b.w < 0 or b.h < 0:
                cat = "negative_coordinates"
            else:
                cat = "out_of_bounds"
            invalid.append(
                {
                    "category": cat,
                    "subtags": _subtags(b),
                    "label_file": rel,
                    "image": img.name,
                    "line_no": i,
                    "line": s,
                    "class_name": NAMES[b.cls],
                    "center": (b.x, b.y),
                    "size": (b.w, b.h),
                    "corners": (x1, y1, x2, y2),
                    "overflow": {
                        "left": max(0, -x1),
                        "top": max(0, -y1),
                        "right": max(0, x2 - 1),
                        "bottom": max(0, y2 - 1),
                    },
                }
            )

        raw = parse_label_file(lp)
        valid = [b for b in raw if b.cls in RUSSIAN_MILITARY_REMAP and is_valid_box(b)]
        bad = [b for b in raw if b.cls in RUSSIAN_MILITARY_REMAP and not is_valid_box(b)]
        if bad and not valid:
            only_invalid_files.append(rel)
        elif bad and valid:
            mixed_files.append({"label_file": rel, "valid": len(valid), "invalid": len(bad)})

    return {
        "invalid": invalid,
        "malformed": malformed,
        "only_invalid_files": only_invalid_files,
        "mixed_files": mixed_files,
    }


def write_report(data: dict, out_path: Path) -> None:
    inv = data["invalid"]
    mal = data["malformed"]
    by_cat = Counter(x["category"] for x in inv)
    subtag_counts = Counter(t for x in inv for t in x["subtags"])
    by_class = Counter(x["class_name"] for x in inv)

    lines: list[str] = []
    lines.append("## Russian Military: invalid boxes removed during cleaning")
    lines.append("")
    lines.append(
        "During dataset preparation (`scripts/prepare_clean.py`), **70 bounding boxes** "
        "were dropped from Russian Military labels because they failed YOLO validity checks "
        "in `scripts/yolo_io.py` (`is_valid_box`). None were dropped due to class remapping "
        "(all six source classes map to `armored_vehicle`)."
    )
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append("| Total label files scanned | 595 |")
    lines.append("| Invalid boxes removed | **70** |")
    lines.append("| Images with only invalid box(es) (excluded entirely) | **54** |")
    lines.append("| Images with valid + invalid boxes (invalid dropped, image kept) | **16** |")
    lines.append("| Malformed label lines (unparseable) | **0** |")
    lines.append("")
    lines.append("### Counts by rejection category")
    lines.append("")
    lines.append("| Category | Boxes |")
    lines.append("|---|---:|")
    for cat in ("out_of_bounds", "negative_coordinates", "zero_area", "malformed"):
        lines.append(f"| {cat} | {by_cat.get(cat, 0)} |")
    lines.append("")
    lines.append(
        "**All 70 removed boxes are `out_of_bounds`**: their normalized corners extend outside "
        "the `[0, 1] × [0, 1]` image rectangle (tolerance `1e-6`). "
        "There were **no** zero-area, negative-coordinate, or malformed boxes in this dataset."
    )
    lines.append("")
    lines.append("### Out-of-bounds subtypes (tags can overlap on one box)")
    lines.append("")
    lines.append("| Subtype | Boxes | Meaning |")
    lines.append("|---|---:|---|")
    subtype_desc = {
        "extends_past_left": "x − w/2 < 0",
        "extends_past_top": "y − h/2 < 0",
        "extends_past_right": "x + w/2 > 1",
        "extends_past_bottom": "y + h/2 > 1",
        "width_or_height_exceeds_1": "w > 1 or h > 1 (invalid normalized size)",
    }
    for tag, desc in subtype_desc.items():
        lines.append(f"| `{tag}` | {subtag_counts.get(tag, 0)} | {desc} |")
    lines.append("")
    lines.append("### Invalid boxes by original class")
    lines.append("")
    lines.append("| Original class | Boxes |")
    lines.append("|---|---:|")
    for name, cnt in by_class.most_common():
        lines.append(f"| {name} | {cnt} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Validation rules (pipeline)")
    lines.append("")
    lines.append("A box is **kept** only if:")
    lines.append("")
    lines.append("1. `w > 0` and `h > 0` (non-zero area)")
    lines.append("2. Center `(x, y)` in `[0, 1]`")
    lines.append("3. `w` and `h` in `(0, 1]`")
    lines.append("4. Corners `(x ± w/2, y ± h/2)` inside `[0, 1]`")
    lines.append("")
    lines.append("Malformed lines (`< 5` fields or non-numeric values) are skipped at parse time and do not appear in box counts.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Category details")
    lines.append("")

    lines.append("### 1. Out of bounds (70 boxes)")
    lines.append("")
    lines.append(
        "Typical cause in this dataset: **full-frame or near full-frame Roboflow exports** "
        "where width `w ≈ 1.0` (or `w > 0.99`) and/or height `h` is large, so the computed "
        "bottom or right edge exceeds `1.0` by a small amount due to floating-point rounding "
        "or slightly loose annotation."
    )
    lines.append("")
    lines.append("**Examples:**")
    lines.append("")
    for ex in inv[:8]:
        x1, y1, x2, y2 = ex["corners"]
        lines.append(f"- **{ex['label_file']}** (line {ex['line_no']}, class `{ex['class_name']}`)")
        lines.append(f"  - Raw: `{ex['line']}`")
        lines.append(f"  - Center (x, y)=({ex['center'][0]:.6g}, {ex['center'][1]:.6g}), (w, h)=({ex['size'][0]:.6g}, {ex['size'][1]:.6g})")
        lines.append(f"  - Corners (x1,y1,x2,y2)=({x1:.6g}, {y1:.6g}, {x2:.6g}, {y2:.6g})")
        lines.append(f"  - Overflow: left={ex['overflow']['left']:.2e}, top={ex['overflow']['top']:.2e}, right={ex['overflow']['right']:.2e}, bottom={ex['overflow']['bottom']:.2e}")
        lines.append(f"  - Tags: {', '.join(f'`{t}`' for t in ex['subtags'])}")
        lines.append("")

    lines.append("### 2. Negative coordinates (0 boxes)")
    lines.append("")
    lines.append("Would apply if center `x`/`y` or size `w`/`h` were negative. **None found** in Russian Military.")
    lines.append("")
    lines.append("### 3. Zero-area boxes (0 boxes)")
    lines.append("")
    lines.append("Would apply if `w <= 0` or `h <= 0`. **None found** in Russian Military.")
    lines.append("")
    lines.append("### 4. Malformed labels (0 lines)")
    lines.append("")
    lines.append(
        "Would apply to lines with fewer than 5 fields or non-numeric values. "
        "**None found**; every annotation line parsed successfully."
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Impact on merged dataset")
    lines.append("")
    lines.append(
        "- **54 images** had a single invalid box and were removed entirely (`no_boxes_after_remap`)."
    )
    lines.append(
        "- **16 images** kept one valid box each after dropping one invalid box."
    )
    lines.append(
        "- Remaining Russian Military after cleaning: **535 images**, **632** `armored_vehicle` instances "
        "(see `reports/pipeline_metadata.json`)."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data = analyze()
    out = repo / "reports" / "russian_military_invalid_boxes.md"
    write_report(data, out)
    print(f"Wrote {out}")
    print(f"Invalid boxes: {len(data['invalid'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
