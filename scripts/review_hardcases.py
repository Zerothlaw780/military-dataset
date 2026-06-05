#!/usr/bin/env python3
"""Interactive review of mined hard-case crops (OpenCV keyboard UI).

Shows each crop large with its filename + confidence and a progress counter,
then records your decision. Designed to follow ``mine_hardcases.py`` /
``deduplicate.py``.

Keyboard controls:
    A  -> label as armored_vehicle  (crop is MOVED into hardcase_armored/)
    C  -> label as civilian_vehicle (kept in place)
    D  -> delete / ignore           (crop moved into hardcase_deleted/)
    U  -> undo last decision
    Q / ESC -> quit (progress is saved; rerun to resume)

Decisions are appended to ``review.csv`` so sessions resume where you left off.

Example:
    python scripts/review_hardcases.py \
        --crops hardcase_mining/crops \
        --metadata hardcase_mining/metadata_dedup.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REVIEW_FIELDS = ["crop_file", "decision", "confidence", "timestamp"]

DECISION_ARMORED = "armored_vehicle"
DECISION_CIVILIAN = "civilian_vehicle"
DECISION_DELETE = "delete"


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _import_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError:
        _fail("OpenCV is required. Install with: pip install opencv-python")
    import cv2

    if not hasattr(cv2, "imshow"):
        _fail("This OpenCV build has no GUI support (imshow). Install 'opencv-python' (not headless).")
    return cv2


def load_confidences(metadata_path: Path | None) -> dict[str, float]:
    conf: dict[str, float] = {}
    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("crop_file")
                if not name:
                    continue
                try:
                    conf[name] = float(row.get("confidence", "") or "nan")
                except ValueError:
                    pass
    return conf


def confidence_from_name(name: str) -> float | None:
    """Fallback: parse '..._cNNN.jpg' (NNN = percent) written by mine_hardcases.py."""
    stem = Path(name).stem
    tail = stem.rsplit("_", 1)[-1]
    if tail.startswith("c") and tail[1:].isdigit():
        return int(tail[1:]) / 100.0
    return None


def load_reviewed(review_csv: Path) -> dict[str, str]:
    reviewed: dict[str, str] = {}
    if review_csv.exists():
        with review_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("crop_file")
                if name:
                    reviewed[name] = row.get("decision", "")
    return reviewed


def append_decision(review_csv: Path, crop_file: str, decision: str, confidence: float | None) -> None:
    new_file = not review_csv.exists()
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "crop_file": crop_file,
                "decision": decision,
                "confidence": "" if confidence is None else round(confidence, 4),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )


def rewrite_review(review_csv: Path, rows: list[dict]) -> None:
    """Rewrite the whole review.csv (used by undo)."""
    with review_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def read_review_rows(review_csv: Path) -> list[dict]:
    if not review_csv.exists():
        return []
    with review_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render(cv2, img, *, header_lines: list[str], max_w: int, max_h: int):
    import numpy as np

    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (disp_w, disp_h), interpolation=interp)

    header_h = 28 * len(header_lines) + 16
    canvas = np.full((header_h + max_h, max_w, 3), 30, dtype=np.uint8)

    off_x = (max_w - disp_w) // 2
    off_y = header_h + (max_h - disp_h) // 2
    canvas[off_y : off_y + disp_h, off_x : off_x + disp_w] = resized

    y = 24
    for line in header_lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        y += 28
    cv2.line(canvas, (0, header_h - 2), (max_w, header_h - 2), (80, 80, 80), 1)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactively review hard-case crops with OpenCV.")
    ap.add_argument("--crops", type=Path, default=Path("hardcase_mining/crops"), help="Folder of crop images.")
    ap.add_argument("--metadata", type=Path, default=None, help="Optional metadata CSV for confidences.")
    ap.add_argument("--review-csv", type=Path, default=None, help="Decisions file (default: <crops>/../review.csv).")
    ap.add_argument("--armored-dir", type=Path, default=None, help="Destination for armored picks (default: <crops>/../hardcase_armored).")
    ap.add_argument("--deleted-dir", type=Path, default=None, help="Destination for deleted/ignored (default: <crops>/../hardcase_deleted).")
    ap.add_argument("--max-width", type=int, default=1000, help="Max display width.")
    ap.add_argument("--max-height", type=int, default=760, help="Max display height.")
    args = ap.parse_args()

    if not args.crops.is_dir():
        _fail(f"crops folder not found: {args.crops}")

    cv2 = _import_cv2()
    try:
        import numpy  # noqa: F401
    except ImportError:
        _fail("NumPy is required. Install with: pip install numpy")
    import shutil

    base = args.crops.parent
    review_csv = args.review_csv or (base / "review.csv")
    armored_dir = args.armored_dir or (base / "hardcase_armored")
    deleted_dir = args.deleted_dir or (base / "hardcase_deleted")

    # Confidence: prefer metadata, then auto-detect dedup/full metadata, then filename.
    meta_path = args.metadata
    if meta_path is None:
        for cand in (base / "metadata_dedup.csv", base / "metadata.csv"):
            if cand.exists():
                meta_path = cand
                break
    confidences = load_confidences(meta_path)

    reviewed = load_reviewed(review_csv)
    all_crops = sorted(p for p in args.crops.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    pending = [p for p in all_crops if p.name not in reviewed]

    total_known = len(reviewed) + len(pending)
    if not pending:
        print(f"Nothing to review. {len(reviewed)} crop(s) already decided in {review_csv}.")
        return 0

    print(f"Reviewing {len(pending)} crop(s) | {len(reviewed)} already done | total {total_known}")
    print("Controls: [A]rmored  [C]ivilian  [D]elete/ignore  [U]ndo  [Q/ESC]uit")

    window = "Hard-case review"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def conf_for(name: str) -> float | None:
        if name in confidences:
            return confidences[name]
        return confidence_from_name(name)

    idx = 0
    undo_stack: list[tuple[Path, str, Path | None]] = []  # (final_path, decision, moved_to)

    while idx < len(pending):
        crop_path = pending[idx]
        if not crop_path.exists():
            idx += 1
            continue

        img = cv2.imread(str(crop_path))
        if img is None:
            print(f"  WARNING: could not read {crop_path.name}, skipping")
            idx += 1
            continue

        conf = conf_for(crop_path.name)
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        done = len(reviewed)
        header = [
            f"[{done + 1}/{total_known}]  remaining: {len(pending) - idx}",
            f"file: {crop_path.name}",
            f"confidence: {conf_str}   keys: A=armored  C=civilian  D=delete  U=undo  Q=quit",
        ]
        canvas = render(cv2, img, header_lines=header, max_w=args.max_width, max_h=args.max_height)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("Quit. Progress saved.")
            break

        if key in (ord("u"), ord("U")):
            if not undo_stack:
                print("  nothing to undo")
                continue
            final_path, decision, moved_to = undo_stack.pop()
            # move the file back if it was relocated
            if moved_to is not None and moved_to.exists():
                shutil.move(str(moved_to), str(final_path))
            # drop last review row for this crop
            rows = read_review_rows(review_csv)
            for i in range(len(rows) - 1, -1, -1):
                if rows[i].get("crop_file") == final_path.name:
                    del rows[i]
                    break
            rewrite_review(review_csv, rows)
            reviewed.pop(final_path.name, None)
            idx = max(0, idx - 1)
            print(f"  undid: {final_path.name}")
            continue

        if key in (ord("a"), ord("A")):
            decision = DECISION_ARMORED
            armored_dir.mkdir(parents=True, exist_ok=True)
            dest = armored_dir / crop_path.name
            shutil.move(str(crop_path), str(dest))
            moved_to = dest
        elif key in (ord("c"), ord("C")):
            decision = DECISION_CIVILIAN
            moved_to = None
        elif key in (ord("d"), ord("D")):
            decision = DECISION_DELETE
            deleted_dir.mkdir(parents=True, exist_ok=True)
            dest = deleted_dir / crop_path.name
            shutil.move(str(crop_path), str(dest))
            moved_to = dest
        else:
            # unknown key: redraw same crop
            continue

        append_decision(review_csv, crop_path.name, decision, conf)
        reviewed[crop_path.name] = decision
        undo_stack.append((crop_path, decision, moved_to))
        idx += 1

    cv2.destroyAllWindows()

    counts: dict[str, int] = {}
    for d in reviewed.values():
        counts[d] = counts.get(d, 0) + 1
    print("\nSummary:")
    for d in (DECISION_ARMORED, DECISION_CIVILIAN, DECISION_DELETE):
        print(f"  {d}: {counts.get(d, 0)}")
    print(f"Decisions saved to {review_csv}")
    print(f"Armored crops -> {armored_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
