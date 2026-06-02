#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def _read_yaml_names(dataset_root: Path) -> tuple[Path | None, list[str] | None]:
    """
    Best-effort extraction of `names:` from a YOLO yaml without dependencies.
    Expects inline list syntax: names: ['a','b',...]
    """
    candidates: list[Path] = []
    candidates.extend(sorted(dataset_root.glob("*.yaml")))
    candidates.extend(sorted(dataset_root.glob("*.yml")))
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            s = line.strip()
            if not s.startswith("names:"):
                continue
            rhs = s.split(":", 1)[1].strip()
            if not (rhs.startswith("[") and rhs.endswith("]")):
                continue
            body = rhs[1:-1].strip()
            parts = [x.strip() for x in body.split(",")] if body else []
            names: list[str] = []
            for item in parts:
                if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                    names.append(item[1:-1])
                else:
                    names.append(item)
            return p, names
    return None, None


def _discover_splits(root: Path) -> dict[str, Path]:
    """
    Returns split -> labels_dir.
    Supports Roboflow layout (train/valid/test) and VisDrone style folders.
    """
    splits: dict[str, Path] = {}
    for split in ("train", "valid", "val", "test"):
        p = root / split / "labels"
        if p.exists():
            splits["valid" if split == "val" else split] = p

    # VisDrone style: root/VisDrone2019-DET-train/labels, etc.
    for p in root.iterdir() if root.exists() else []:
        if not p.is_dir():
            continue
        name = p.name.lower()
        lp = p / "labels"
        if not lp.exists():
            continue
        if "train" in name:
            splits.setdefault("train", lp)
        elif "val" in name or "valid" in name:
            splits.setdefault("valid", lp)
        elif "test" in name:
            splits.setdefault("test", lp)
    return splits


def _iter_label_files(labels_dir: Path) -> list[Path]:
    return [p for p in labels_dir.rglob("*.txt") if p.is_file()]


def _parse_label_classes(label_path: Path) -> tuple[list[int], bool]:
    """
    Returns (class_ids, ok). ok=False if we hit parse problems.
    """
    try:
        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return [], False
    out: list[int] = []
    ok = True
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 1:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            ok = False
            continue
        out.append(cls)
    return out, ok


def _class_name(cid: int, names: list[str] | None) -> str:
    if names and 0 <= cid < len(names):
        return names[cid]
    return ""


def analyze_dataset(dataset_name: str, dataset_root: Path) -> dict:
    yaml_path, names = _read_yaml_names(dataset_root)
    splits = _discover_splits(dataset_root)
    if not splits:
        return {"name": dataset_name, "root": str(dataset_root), "error": "no_labels_found"}

    instances_total: Counter[int] = Counter()
    images_total: Counter[int] = Counter()
    split_instances: dict[str, Counter[int]] = {}
    split_images: dict[str, Counter[int]] = {}
    totals = {"label_files": 0, "empty_label_files": 0, "parse_error_files": 0}

    for split, labels_dir in splits.items():
        inst = Counter()
        imgs = Counter()
        label_files = _iter_label_files(labels_dir)
        totals["label_files"] += len(label_files)
        for lf in label_files:
            class_ids, ok = _parse_label_classes(lf)
            if not ok:
                totals["parse_error_files"] += 1
            if not class_ids:
                totals["empty_label_files"] += 1
                continue

            # instances
            inst.update(class_ids)
            instances_total.update(class_ids)

            # images containing each class (unique per file)
            for cid in set(class_ids):
                imgs[cid] += 1
                images_total[cid] += 1

        split_instances[split] = inst
        split_images[split] = imgs

    observed = sorted(set(instances_total.keys()) | set(images_total.keys()))
    return {
        "name": dataset_name,
        "root": str(dataset_root),
        "yaml_path": str(yaml_path) if yaml_path else None,
        "names": names,
        "splits": {k: str(v) for k, v in splits.items()},
        "totals": totals,
        "observed_class_ids": observed,
        "instances_total": dict(instances_total),
        "images_total": dict(images_total),
        "split_instances": {k: dict(v) for k, v in split_instances.items()},
        "split_images": {k: dict(v) for k, v in split_images.items()},
    }


def write_csv_instances(path: Path, summary: dict) -> None:
    names = summary.get("names")
    observed: list[int] = summary["observed_class_ids"]
    splits = list(summary["split_instances"].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "class_id", "class_name", "split", "instances"])
        for cid in observed:
            cname = _class_name(cid, names)
            for split in splits:
                cnt = int(summary["split_instances"][split].get(str(cid), summary["split_instances"][split].get(cid, 0)))
                w.writerow([summary["name"], cid, cname, split, cnt])
            total = int(summary["instances_total"].get(str(cid), summary["instances_total"].get(cid, 0)))
            w.writerow([summary["name"], cid, cname, "TOTAL", total])


def write_csv_images(path: Path, summary: dict) -> None:
    names = summary.get("names")
    observed: list[int] = summary["observed_class_ids"]
    splits = list(summary["split_images"].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "class_id", "class_name", "split", "images_with_class"])
        for cid in observed:
            cname = _class_name(cid, names)
            for split in splits:
                cnt = int(summary["split_images"][split].get(str(cid), summary["split_images"][split].get(cid, 0)))
                w.writerow([summary["name"], cid, cname, split, cnt])
            total = int(summary["images_total"].get(str(cid), summary["images_total"].get(cid, 0)))
            w.writerow([summary["name"], cid, cname, "TOTAL", total])


def write_markdown_report(path: Path, summaries: list[dict]) -> None:
    lines: list[str] = []
    lines.append("## Class distribution report")
    lines.append("")
    lines.append("For each dataset, this report includes:")
    lines.append("")
    lines.append("- **Instances per original class** (count of labeled boxes)")
    lines.append("- **Images containing each class** (count of label files where the class appears at least once)")
    lines.append("")

    for s in summaries:
        lines.append(f"## {s.get('name','(unknown)')}")
        lines.append("")
        if "error" in s:
            lines.append(f"**Error**: {s['error']}")
            lines.append("")
            continue

        lines.append(f"- **Root**: `{s['root']}`")
        if s.get("yaml_path"):
            lines.append(f"- **YAML**: `{s['yaml_path']}`")
        if s.get("names"):
            lines.append(f"- **Original class names**: {s['names']}")
        lines.append(f"- **Label files scanned**: {s['totals']['label_files']}")
        lines.append(f"- **Empty label files**: {s['totals']['empty_label_files']}")
        lines.append(f"- **Parse-error label files**: {s['totals']['parse_error_files']}")
        lines.append("")

        observed: list[int] = s["observed_class_ids"]
        names = s.get("names")

        lines.append("### Instances per class (TOTAL)")
        lines.append("| class_id | class_name | instances |")
        lines.append("|---:|---|---:|")
        for cid in observed:
            cname = _class_name(cid, names)
            total = int(s["instances_total"].get(str(cid), s["instances_total"].get(cid, 0)))
            lines.append(f"| {cid} | {cname} | {total} |")
        lines.append("")

        lines.append("### Images containing class (TOTAL)")
        lines.append("| class_id | class_name | images_with_class |")
        lines.append("|---:|---|---:|")
        for cid in observed:
            cname = _class_name(cid, names)
            total = int(s["images_total"].get(str(cid), s["images_total"].get(cid, 0)))
            lines.append(f"| {cid} | {cname} | {total} |")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create class distribution report (instances + images-per-class).")
    ap.add_argument("--out-dir", default="reports", help="Output directory (default: reports).")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Russian Military", repo / "datasets" / "russian_military" / "Russian-military-annotated"),
        ("Aboba", repo / "datasets" / "aboba" / "aboba"),
        ("VisDrone", repo / "datasets" / "visdrone" / "VisDrone_Veri"),
    ]

    summaries = [analyze_dataset(name, root) for name, root in datasets]

    # Per-dataset CSVs
    for s in summaries:
        if "error" in s:
            continue
        slug = s["name"].lower().replace(" ", "_")
        write_csv_instances(out_dir / f"class_distribution_{slug}_instances.csv", s)
        write_csv_images(out_dir / f"class_distribution_{slug}_images.csv", s)

    # Combined markdown
    write_markdown_report(out_dir / "class_distribution.md", summaries)
    print(f"Wrote: {out_dir / 'class_distribution.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

