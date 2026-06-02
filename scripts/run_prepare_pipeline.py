#!/usr/bin/env python3
"""Run full dataset preparation pipeline (no training)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataset_config import DATASET_SOURCES
from generate_final_stats import write_final_report
from prepare_clean import clean_source
from prepare_merge import (
    collect_staged_samples,
    split_samples,
    write_data_yaml,
    write_merged_dataset,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare merged 3-class YOLO dataset.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--phash-threshold", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-visdrone", action="store_true", help="Skip VisDrone (debug only)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    staging = repo / "merged_dataset" / "_staging"
    output = repo / "merged_dataset"
    reports = repo / "reports"

    all_stats: dict = {}

    for name, cfg in DATASET_SOURCES.items():
        if name == "visdrone" and args.skip_visdrone:
            continue
        root = (repo / cfg["root"]).resolve()
        if not root.is_dir():
            print(f"ERROR: missing {name} at {root}")
            return 1

        print(f"\n=== Cleaning {name} ===")
        kw = dict(
            root=root,
            staging_dir=staging / name,
            remap=cfg["remap"],
            drop=cfg["drop"],
            prefix=cfg["prefix"],
            keep_all=cfg.get("keep_all", True),
            max_hamming=args.phash_threshold,
        )
        if not cfg.get("keep_all", True):
            kw["civilian_target_min"] = cfg.get("civilian_target_min", 6000)
            kw["civilian_target_max"] = cfg.get("civilian_target_max", 8000)

        stats, inst = clean_source(**kw)
        all_stats[name] = {
            "pairs_seen": stats.pairs_seen,
            "empty_labels_removed": stats.empty_labels_removed,
            "no_boxes_after_remap": stats.no_boxes_after_remap,
            "invalid_boxes_dropped": stats.invalid_boxes_dropped,
            "duplicates_removed": stats.duplicates_removed,
            "subset_removed": stats.subset_removed,
            "kept_images": stats.kept,
            "instances": dict(inst),
        }
        print(f"  kept {stats.kept} images | instances {dict(inst)}")

    print("\n=== Merge + split 70/20/10 ===")
    samples = collect_staged_samples(staging)
    print(f"  staged samples: {len(samples)}")
    splits = split_samples(samples, seed=args.seed)
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")

    inst_merged = write_merged_dataset(splits, output)
    yaml_path = write_data_yaml(output)
    print(f"  wrote {yaml_path}")

    meta = {
        "staging": str(staging),
        "output": str(output),
        "seed": args.seed,
        "phash_threshold": args.phash_threshold,
        "per_source": all_stats,
        "merged_instances": dict(inst_merged),
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "pipeline_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_final_report(
        output,
        reports / "final_statistics.md",
        pipeline_meta={
            "split": "70/20/10 stratified by dominant class",
            "visdrone_civilian_target": "6000-8000 instances",
            "merged_instances": dict(inst_merged),
        },
    )
    print(f"\nWrote {reports / 'final_statistics.md'}")
    print("Done (training not started).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
