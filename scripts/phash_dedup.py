"""Perceptual-hash duplicate detection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imagehash
from PIL import Image


def compute_phash(image_path: Path) -> imagehash.ImageHash:
    with Image.open(image_path) as im:
        return imagehash.phash(im.convert("RGB"))


def cluster_by_phash(paths: list[Path], *, max_hamming: int = 8) -> list[list[Path]]:
    clusters: list[list[Path]] = []
    cluster_hashes: list[imagehash.ImageHash] = []

    for p in paths:
        try:
            h = compute_phash(p)
        except Exception:
            clusters.append([p])
            cluster_hashes.append(imagehash.phash(Image.new("RGB", (8, 8))))
            continue
        placed = False
        for i, ch in enumerate(cluster_hashes):
            if h - ch <= max_hamming:
                clusters[i].append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
            cluster_hashes.append(h)
    return clusters


def pick_representatives(
    clusters: list[list[Path]],
    score_fn: Callable[[Path], float],
) -> list[Path]:
    return [max(group, key=score_fn) for group in clusters]
