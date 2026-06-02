"""YOLO label parsing, validation, and filesystem helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_config import IMG_EXTS


@dataclass(frozen=True)
class YoloBox:
    cls: int
    x: float
    y: float
    w: float
    h: float


def is_valid_box(b: YoloBox) -> bool:
    if b.w <= 0 or b.h <= 0:
        return False
    if not (0.0 <= b.x <= 1.0 and 0.0 <= b.y <= 1.0):
        return False
    if not (0.0 < b.w <= 1.0 and 0.0 < b.h <= 1.0):
        return False
    x1, y1 = b.x - b.w / 2, b.y - b.h / 2
    x2, y2 = b.x + b.w / 2, b.y + b.h / 2
    return x1 >= -1e-6 and y1 >= -1e-6 and x2 <= 1.0 + 1e-6 and y2 <= 1.0 + 1e-6


def parse_label_file(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append(
                YoloBox(
                    cls=int(float(parts[0])),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                )
            )
        except ValueError:
            continue
    return boxes


def write_label_file(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls} {b.x:.6f} {b.y:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def remap_boxes(boxes: list[YoloBox], remap: dict[int, int], drop: set[int]) -> list[YoloBox]:
    out: list[YoloBox] = []
    for b in boxes:
        if b.cls in drop:
            continue
        if b.cls not in remap:
            continue
        if not is_valid_box(b):
            continue
        out.append(YoloBox(cls=remap[b.cls], x=b.x, y=b.y, w=b.w, h=b.h))
    return out


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        c = images_dir / f"{stem}{ext}"
        if c.is_file():
            return c
    for ext in IMG_EXTS:
        for c in images_dir.rglob(f"{stem}{ext}"):
            if c.is_file():
                return c
    return None


def discover_label_image_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    seen_labels: set[Path] = set()

    def add_from(labels_dir: Path, images_dir: Path, split: str) -> None:
        if not labels_dir.is_dir() or not images_dir.is_dir():
            return
        for lp in labels_dir.rglob("*.txt"):
            if not lp.is_file():
                continue
            key = lp.resolve()
            if key in seen_labels:
                continue
            img = find_image_for_label(lp, images_dir)
            if img is not None:
                seen_labels.add(key)
                pairs.append((lp, img, split))

    for split in ("train", "valid", "val", "test"):
        ld, idir = root / split / "labels", root / split / "images"
        if ld.exists():
            add_from(ld, idir, "valid" if split == "val" else split)

    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if not p.is_dir():
            continue
        name = p.name.lower()
        # Skip standard split dirs already handled above
        if name in ("train", "valid", "val", "test"):
            continue
        ld, idir = p / "labels", p / "images"
        if not ld.exists():
            continue
        if "train" in name:
            add_from(ld, idir, "train")
        elif "val" in name or "valid" in name:
            add_from(ld, idir, "valid")
        elif "test" in name:
            add_from(ld, idir, "test")

    return pairs
