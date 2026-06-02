#!/usr/bin/env python3
"""Detect train/val/test leakage via exact and perceptual image hashes."""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from phash_dedup import cluster_by_phash, compute_phash

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def sha1_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def collect_images(merged_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for split in SPLITS:
        img_dir = merged_root / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.append((split, p))
    return out


def find_exact_duplicates(items: list[tuple[str, Path]]) -> dict[str, list[tuple[str, str]]]:
    """sha1 -> [(split, filename), ...]"""
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, path in items:
        by_hash[sha1_file(path)].append((split, path.name))
    return {h: v for h, v in by_hash.items() if len(v) > 1}


def cross_split_only(groups: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, str]]]:
    leaked = {}
    for key, members in groups.items():
        splits_present = {m[0] for m in members}
        if len(splits_present) > 1:
            leaked[key] = members
    return leaked


def find_near_duplicates_cross_split(
    items: list[tuple[str, Path]],
    *,
    max_hamming: int = 8,
) -> list[dict]:
    """
    Cluster by pHash; return clusters that span multiple splits.
    """
    paths = [p for _, p in items]
    split_of = {p: s for s, p in items}

    clusters = cluster_by_phash(paths, max_hamming=max_hamming)
    cross_clusters: list[dict] = []

    for cluster in clusters:
        splits_in = {split_of[p] for p in cluster}
        if len(splits_in) <= 1:
            continue
        members = []
        for p in cluster:
            try:
                ph = str(compute_phash(p))
            except Exception:
                ph = "error"
            members.append(
                {
                    "split": split_of[p],
                    "file": p.name,
                    "path": str(p),
                    "phash": ph,
                }
            )
        cross_clusters.append(
            {
                "size": len(cluster),
                "splits": sorted(splits_in),
                "members": members,
            }
        )
    return cross_clusters


def write_report(
    out_path: Path,
    *,
    merged_root: Path,
    total_by_split: dict[str, int],
    exact_all: dict[str, list[tuple[str, str]]],
    exact_cross: dict[str, list[tuple[str, str]]],
    near_cross: list[dict],
    max_hamming: int,
) -> None:
    lines: list[str] = []
    lines.append("## Train / Val / Test leakage report")
    lines.append("")
    lines.append(f"- **Dataset**: `{merged_root.resolve()}`")
    lines.append(f"- **Splits checked**: {', '.join(SPLITS)}")
    lines.append(f"- **Near-duplicate threshold**: Hamming distance ≤ {max_hamming} (pHash)")
    lines.append("")

    lines.append("### Image counts per split")
    lines.append("")
    lines.append("| split | images |")
    lines.append("|---|---:|")
    for s in SPLITS:
        lines.append(f"| {s} | {total_by_split.get(s, 0)} |")
    lines.append(f"| **total** | {sum(total_by_split.values())} |")
    lines.append("")

    lines.append("### 1. Exact duplicates (SHA-1 byte hash)")
    lines.append("")
    lines.append(
        f"- **Duplicate groups (same bytes, any split)**: {len(exact_all)} "
        f"({sum(len(v) for v in exact_all.values())} files in groups)"
    )
    lines.append(
        f"- **Cross-split exact duplicate groups (leakage)**: **{len(exact_cross)}**"
    )
    lines.append("")

    if exact_cross:
        lines.append("#### Images appearing in multiple splits (exact match)")
        lines.append("")
        for i, (h, members) in enumerate(sorted(exact_cross.items(), key=lambda x: -len(x[1])), 1):
            splits = sorted({m[0] for m in members})
            lines.append(f"**Group {i}** — SHA-1 `{h[:12]}…`, splits: {', '.join(splits)}")
            lines.append("")
            lines.append("| split | filename |")
            lines.append("|---|---|")
            for split, name in sorted(members):
                lines.append(f"| {split} | `{name}` |")
            lines.append("")
    else:
        lines.append("No exact duplicate images were found across different splits.")
        lines.append("")

    lines.append("### 2. Near-duplicates (perceptual hash)")
    lines.append("")
    lines.append(
        f"- **Cross-split near-duplicate clusters**: **{len(near_cross)}**"
    )
    if near_cross:
        total_files = sum(c["size"] for c in near_cross)
        lines.append(f"- **Files involved**: {total_files}")
    lines.append("")

    if near_cross:
        lines.append("#### Clusters spanning multiple splits")
        lines.append("")
        for i, cluster in enumerate(
            sorted(near_cross, key=lambda c: (-c["size"], c["splits"])), 1
        ):
            lines.append(
                f"**Cluster {i}** — {cluster['size']} images, splits: {', '.join(cluster['splits'])}"
            )
            lines.append("")
            lines.append("| split | filename | phash |")
            lines.append("|---|---|---|")
            for m in sorted(cluster["members"], key=lambda x: (x["split"], x["file"])):
                lines.append(f"| {m['split']} | `{m['file']}` | `{m['phash']}` |")
            lines.append("")
    else:
        lines.append("No near-duplicate clusters spanned more than one split.")
        lines.append("")

    lines.append("### 3. Summary")
    lines.append("")
    if not exact_cross and not near_cross:
        lines.append(
            "**No cross-split leakage detected** under exact SHA-1 matching and "
            f"pHash near-duplicate clustering (≤{max_hamming})."
        )
    else:
        if exact_cross:
            lines.append(
                f"- **{len(exact_cross)}** exact duplicate group(s) share the same image bytes across splits."
            )
        if near_cross:
            lines.append(
                f"- **{len(near_cross)}** near-duplicate cluster(s) link visually similar images across splits."
            )
        lines.append(
            "- Review the groups above; consider removing duplicates from val/test or "
            "re-splitting so related frames stay in one split."
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Check train/val/test leakage.")
    ap.add_argument("--merged-root", type=Path, default=None)
    ap.add_argument("--max-hamming", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    merged = (args.merged_root or repo / "merged_dataset").resolve()
    out = (args.out or repo / "reports" / "leakage_report.md").resolve()

    items = collect_images(merged)
    if not items:
        print(f"No images found under {merged}")
        return 1

    total_by_split: dict[str, int] = Counter_split(items)
    print(f"Scanning {len(items)} images...")

    exact_all = find_exact_duplicates(items)
    exact_cross = cross_split_only(exact_all)

    print("Computing perceptual hashes (may take a few minutes)...")
    near_cross = find_near_duplicates_cross_split(items, max_hamming=args.max_hamming)

    write_report(
        out,
        merged_root=merged,
        total_by_split=total_by_split,
        exact_all=exact_all,
        exact_cross=exact_cross,
        near_cross=near_cross,
        max_hamming=args.max_hamming,
    )
    print(f"Wrote {out}")
    print(f"Exact cross-split groups: {len(exact_cross)}")
    print(f"Near-duplicate cross-split clusters: {len(near_cross)}")
    return 0


def Counter_split(items: list[tuple[str, Path]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for s, _ in items:
        c[s] += 1
    return dict(c)


if __name__ == "__main__":
    raise SystemExit(main())
