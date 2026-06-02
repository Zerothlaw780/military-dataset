"""Merge staged sources, split 70/20/10, write data.yaml."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

from dataset_config import FINAL_CLASS_NAMES, FINAL_NC
from yolo_io import parse_label_file


def collect_staged_samples(staging_root: Path) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    for source_dir in sorted(staging_root.iterdir()):
        if not source_dir.is_dir():
            continue
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for lbl in labels_dir.glob("*.txt"):
            stem = lbl.stem
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                c = images_dir / f"{stem}{ext}"
                if c.is_file():
                    img = c
                    break
            if img is not None:
                samples.append((img, lbl))
    return samples


def dominant_class(label_path: Path) -> int:
    boxes = parse_label_file(label_path)
    if not boxes:
        return -1
    return Counter(b.cls for b in boxes).most_common(1)[0][0]


def split_samples(
    samples: list[tuple[Path, Path]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, Path]]] = {}
    for img, lbl in samples:
        by_class.setdefault(dominant_class(lbl), []).append((img, lbl))

    train, val, test = [], [], []
    for group in by_class.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = max(0, n - n_train - n_val)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def write_merged_dataset(
    splits: dict[str, list[tuple[Path, Path]]],
    output_root: Path,
) -> Counter:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = output_root / split / kind
            if d.exists():
                shutil.rmtree(d)

    inst = Counter()
    for split_name, items in splits.items():
        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        for img_src, lbl_src in items:
            shutil.copy2(img_src, img_out / img_src.name)
            shutil.copy2(lbl_src, lbl_out / lbl_src.name)
            for b in parse_label_file(lbl_src):
                inst[b.cls] += 1
    return inst


def write_data_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        f"""# Auto-generated merged 3-class vehicle dataset
path: {output_root.resolve()}
train: train/images
val: val/images
test: test/images

nc: {FINAL_NC}
names: {FINAL_CLASS_NAMES}
""",
        encoding="utf-8",
    )
    return yaml_path
