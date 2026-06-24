#!/usr/bin/env python3
"""Hard-case mining for the military vehicle detector.

Runs an Ultralytics YOLO model over every video in a folder and saves the
candidate *hard cases* for review. Two modes are supported:

  civilian (default)  Save detections labeled ``civilian_vehicle`` with high
                      confidence. In drone footage the model tends to call
                      armored vehicles (BMP/BTR/APC/IFV, damaged armor, ...)
                      ``civilian_vehicle``, so confident civilian predictions
                      surface likely mislabels.
  hardcase_all        Also save the predictions the model is *uncertain* about:
                      confident ``civilian_vehicle`` PLUS low-confidence
                      ``armored_vehicle`` and ``tank`` (confidence below
                      ``--uncertain-conf``). This widens mining beyond civilian
                      predictions to every spot where the model hesitates.

Each saved crop's predicted class and confidence are recorded in metadata.csv.

To avoid flooding the review stage with the same vehicle across consecutive
frames, mining applies *temporal deduplication*: each saved crop's perceptual
hash and frame number are remembered, and a new detection whose hash is within
``--hash-threshold`` of a recently saved crop is skipped for the next
``--cooldown-frames`` frames. This typically removes the large majority of
repeated BMP/BTR/APC crops before they ever reach the review grid.

Outputs (under ``--output``):
  crops/                 cropped detection images (one per kept box)
  metadata.csv           one row per saved crop with full provenance

Example:
    python scripts/mine_hardcases.py \
        --model runs/detect/train/weights/best.pt \
        --videos videos \
        --output hardcase_mining \
        --conf 0.50 \
        --hash-threshold 10 \
        --cooldown-frames 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm", ".wmv", ".flv"}
DEFAULT_TARGET_CLASS = "civilian_vehicle"

METADATA_FIELDS = [
    "crop_file",
    "video",
    "frame_index",
    "time_sec",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_w",
    "box_h",
    "frame_w",
    "frame_h",
    "model",
]


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _import_deps():
    """Import heavy optional deps with actionable error messages."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        _fail("OpenCV is required. Install with: pip install opencv-python")
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        _fail("Ultralytics is required. Install with: pip install ultralytics")
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        _fail("Pillow and imagehash are required. Install with: pip install Pillow imagehash")
    import cv2
    import imagehash
    from PIL import Image
    from ultralytics import YOLO

    return cv2, YOLO, imagehash, Image


def compute_crop_phash(cv2, imagehash, Image, crop_bgr, hash_size: int = 8, resize_to: int = 128):
    """Perceptual hash of a BGR crop (as produced by OpenCV).

    The crop is first resized to a fixed ``resize_to`` x ``resize_to`` square so
    that the same vehicle produces a near-identical hash regardless of how large
    or tightly the detector boxed it. This makes temporal dedup robust to the
    bbox size/aspect jitter that is common across consecutive convoy frames.
    """
    if resize_to and resize_to > 0:
        h, w = crop_bgr.shape[:2]
        interp = cv2.INTER_AREA if (w >= resize_to and h >= resize_to) else cv2.INTER_LINEAR
        crop_bgr = cv2.resize(crop_bgr, (resize_to, resize_to), interpolation=interp)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb), hash_size=hash_size)


def iter_video_files(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        _fail(f"videos folder not found: {videos_dir}")
    return sorted(p for p in videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def find_class_id(names: dict[int, str], target_name: str) -> int | None:
    for cid, name in names.items():
        if str(name).lower() == target_name.lower():
            return int(cid)
    return None


def resolve_target_class_id(names: dict[int, str], target_name: str, override_id: int | None) -> int:
    if override_id is not None:
        return override_id
    cid = find_class_id(names, target_name)
    if cid is not None:
        return cid
    _fail(
        f"class '{target_name}' not found in model names {names}. "
        f"Pass --class-id to select it explicitly."
    )


def process_video(
    *,
    cv2,
    imagehash,
    Image,
    model,
    video_path: Path,
    crops_dir: Path,
    writer: "csv.DictWriter",
    accept_fn: "Callable[[int, float], bool]",
    predict_conf: float,
    class_names: dict[int, str],
    frame_stride: int,
    imgsz: int,
    device: str | None,
    min_crop_px: int,
    hash_threshold: int,
    cooldown_frames: int,
    hash_size: int,
    hash_resize: int,
    model_name: str,
) -> tuple[int, int, int, int, Counter]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path.name}, skipping")
        return 0, 0, 0, 0, Counter()

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-6:
        fps = 30.0  # fallback when container lacks fps metadata

    # Temporal dedup memory for THIS video: list of (frame_index, phash) for
    # recently saved crops. An entry blocks similar detections until it ages out
    # of the cooldown window, after which it is pruned.
    recent_saved: list[tuple[int, object]] = []

    saved = 0
    skipped_temporal = 0
    skipped_small = 0  # accepted by the mode rule but bbox too small to be useful
    considered = 0  # detections accepted by the mode rule + size (candidates for saving)
    saved_by_class: Counter = Counter()
    frame_index = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % frame_stride != 0:
            continue

        # Drop memory entries older than the cooldown window.
        if recent_saved:
            recent_saved = [
                (fi, hh) for (fi, hh) in recent_saved if frame_index - fi <= cooldown_frames
            ]

        results = model.predict(
            frame,
            conf=predict_conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if not results:
            continue
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        det_idx = 0
        for b in boxes:
            cls_id = int(b.cls[0])
            confidence = float(b.conf[0])
            if not accept_fn(cls_id, confidence):
                continue

            x1f, y1f, x2f, y2f = (float(v) for v in b.xyxy[0].tolist())
            x1 = max(0, min(int(round(x1f)), frame_w - 1))
            y1 = max(0, min(int(round(y1f)), frame_h - 1))
            x2 = max(0, min(int(round(x2f)), frame_w))
            y2 = max(0, min(int(round(y2f)), frame_h))
            # Reject tiny boxes: too small to review or to be useful for retraining.
            if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
                skipped_small += 1
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            considered += 1

            # Temporal dedup: skip if this crop matches a recently saved crop
            # (same vehicle across consecutive frames) within the cooldown window.
            crop_hash = compute_crop_phash(
                cv2, imagehash, Image, crop, hash_size=hash_size, resize_to=hash_resize
            )
            is_recent_duplicate = any(
                (crop_hash - hh) <= hash_threshold
                and (frame_index - fi) <= cooldown_frames
                for (fi, hh) in recent_saved
            )
            if is_recent_duplicate:
                skipped_temporal += 1
                continue

            crop_name = (
                f"{video_path.stem}_f{frame_index:06d}_d{det_idx:02d}"
                f"_c{int(round(confidence * 100)):03d}.jpg"
            )
            crop_path = crops_dir / crop_name
            if not cv2.imwrite(str(crop_path), crop):
                print(f"  WARNING: failed to write {crop_path.name}")
                continue

            writer.writerow(
                {
                    "crop_file": crop_name,
                    "video": video_path.name,
                    "frame_index": frame_index,
                    "time_sec": round(frame_index / fps, 3),
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, str(cls_id)),
                    "confidence": round(confidence, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_w": x2 - x1,
                    "box_h": y2 - y1,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "model": model_name,
                }
            )
            # Remember this crop so similar ones are suppressed during cooldown.
            recent_saved.append((frame_index, crop_hash))
            saved_by_class[class_names.get(cls_id, str(cls_id))] += 1
            det_idx += 1
            saved += 1

    cap.release()
    return saved, skipped_temporal, skipped_small, considered, saved_by_class


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine hard-case vehicle detections from videos.")
    ap.add_argument("--model", required=True, type=Path, help="Path to Ultralytics YOLO weights (.pt/.onnx/.engine).")
    ap.add_argument("--videos", type=Path, default=Path("videos"), help="Folder containing input videos.")
    ap.add_argument("--output", type=Path, default=Path("hardcase_mining"), help="Output root directory.")
    ap.add_argument(
        "--mode",
        choices=("civilian", "hardcase_all"),
        default="civilian",
        help="Mining mode. 'civilian' (default): confident civilian_vehicle predictions only. "
        "'hardcase_all': also collect low-confidence armored_vehicle and tank predictions "
        "(model is uncertain).",
    )
    ap.add_argument("--conf", type=float, default=0.50, help="Min confidence for civilian_vehicle detections.")
    ap.add_argument("--target-class", default=DEFAULT_TARGET_CLASS, help="Class name to mine in 'civilian' mode.")
    ap.add_argument("--class-id", type=int, default=None, help="Override target class id (skips name lookup).")
    ap.add_argument(
        "--uncertain-conf",
        type=float,
        default=0.60,
        help="hardcase_all: keep armored_vehicle/tank predictions with confidence BELOW this (uncertain).",
    )
    ap.add_argument(
        "--uncertain-floor",
        type=float,
        default=0.25,
        help="hardcase_all: ignore predictions below this confidence (noise floor).",
    )
    ap.add_argument("--frame-stride", type=int, default=15, help="Process every Nth frame (>=1).")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    ap.add_argument("--device", default=None, help="Inference device, e.g. 'cpu', '0', '0,1'. Default: auto.")
    ap.add_argument(
        "--min-crop-px",
        type=int,
        default=50,
        help="Reject crops whose bbox width OR height is below this many pixels. "
        "Tiny crops are unusable for review and retraining. Default: 50.",
    )
    ap.add_argument(
        "--dedup-mode",
        choices=("standard", "aggressive"),
        default="standard",
        help="Temporal dedup strength. 'aggressive' uses a looser hash threshold and longer "
        "cooldown to collapse convoy repeats harder. Explicit --hash-threshold/--cooldown-frames "
        "override the mode defaults.",
    )
    ap.add_argument(
        "--hash-threshold",
        type=int,
        default=None,
        help="Max pHash Hamming distance to treat a new crop as a temporal duplicate (skip it). "
        "Default: 10 (standard) / 18 (aggressive).",
    )
    ap.add_argument(
        "--cooldown-frames",
        type=int,
        default=None,
        help="After saving a crop, suppress similar crops for this many frames. "
        "Default: 100 (standard) / 250 (aggressive).",
    )
    ap.add_argument("--hash-size", type=int, default=8, help="pHash size (8 -> 64-bit hash).")
    ap.add_argument(
        "--hash-resize",
        type=int,
        default=128,
        help="Resize crops to this square size before hashing (scale/size-invariance). 0 disables.",
    )
    args = ap.parse_args()

    # Resolve mode-dependent defaults while honoring any explicit overrides.
    mode_defaults = {
        "standard": {"hash_threshold": 10, "cooldown_frames": 100},
        "aggressive": {"hash_threshold": 18, "cooldown_frames": 250},
    }[args.dedup_mode]
    hash_threshold = args.hash_threshold if args.hash_threshold is not None else mode_defaults["hash_threshold"]
    cooldown_frames = args.cooldown_frames if args.cooldown_frames is not None else mode_defaults["cooldown_frames"]

    if args.frame_stride < 1:
        _fail("--frame-stride must be >= 1")
    if cooldown_frames < 0:
        _fail("--cooldown-frames must be >= 0")
    if hash_threshold < 0:
        _fail("--hash-threshold must be >= 0")
    if args.hash_resize < 0:
        _fail("--hash-resize must be >= 0")
    if not (0.0 <= args.uncertain_floor < args.uncertain_conf <= 1.0):
        _fail("require 0 <= --uncertain-floor < --uncertain-conf <= 1")
    if not args.model.exists():
        _fail(f"model weights not found: {args.model}")

    cv2, YOLO, imagehash, Image = _import_deps()

    videos = iter_video_files(args.videos)
    if not videos:
        _fail(f"no video files found in {args.videos} (extensions: {sorted(VIDEO_EXTS)})")

    output_root = args.output
    crops_dir = output_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.csv"

    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))
    names = {int(k): str(v) for k, v in model.names.items()}

    # Build the per-detection acceptance rule and the model prediction floor.
    civilian_id = resolve_target_class_id(names, args.target_class, args.class_id)
    if args.mode == "civilian":
        predict_conf = args.conf

        def accept_fn(cls_id: int, conf: float) -> bool:
            return cls_id == civilian_id and conf >= args.conf

        print(f"Mode: civilian | target id={civilian_id} '{names.get(civilian_id, civilian_id)}' conf>={args.conf}")
    else:  # hardcase_all
        armored_id = find_class_id(names, "armored_vehicle")
        tank_id = find_class_id(names, "tank")
        if armored_id is None or tank_id is None:
            _fail(
                "hardcase_all mode needs 'armored_vehicle' and 'tank' in model names; "
                f"found {names}."
            )
        # Lower the model floor so uncertain armored/tank boxes are visible, while
        # still capturing confident civilian_vehicle predictions.
        predict_conf = min(args.conf, args.uncertain_floor)
        uncertain_ids = {armored_id, tank_id}

        def accept_fn(cls_id: int, conf: float) -> bool:
            if cls_id == civilian_id:
                return conf >= args.conf
            if cls_id in uncertain_ids:
                return args.uncertain_floor <= conf < args.uncertain_conf
            return False

        print(
            f"Mode: hardcase_all | civilian '{names.get(civilian_id, civilian_id)}' conf>={args.conf}; "
            f"armored/tank uncertain in [{args.uncertain_floor}, {args.uncertain_conf})"
        )

    print(f"Found {len(videos)} video(s). Processing every {args.frame_stride} frame(s).")
    print(
        f"Temporal dedup: mode={args.dedup_mode}, hash-threshold={hash_threshold}, "
        f"cooldown-frames={cooldown_frames}, hash-resize={args.hash_resize}"
    )

    total_saved = 0
    total_skipped = 0
    total_skipped_small = 0
    total_considered = 0
    saved_by_class: Counter = Counter()
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for vi, video_path in enumerate(videos, 1):
            print(f"[{vi}/{len(videos)}] {video_path.name}")
            saved, skipped, skipped_small, considered, by_class = process_video(
                cv2=cv2,
                imagehash=imagehash,
                Image=Image,
                model=model,
                video_path=video_path,
                crops_dir=crops_dir,
                writer=writer,
                accept_fn=accept_fn,
                predict_conf=predict_conf,
                class_names=names,
                frame_stride=args.frame_stride,
                imgsz=args.imgsz,
                device=args.device,
                min_crop_px=args.min_crop_px,
                hash_threshold=hash_threshold,
                cooldown_frames=cooldown_frames,
                hash_size=args.hash_size,
                hash_resize=args.hash_resize,
                model_name=args.model.name,
            )
            f.flush()
            total_saved += saved
            total_skipped += skipped
            total_skipped_small += skipped_small
            total_considered += considered
            saved_by_class.update(by_class)
            print(
                f"    detections: {considered} | saved: {saved} | "
                f"skipped small: {skipped_small} | skipped (temporal dedup): {skipped}"
            )

    reduction = (total_skipped / total_considered * 100.0) if total_considered else 0.0
    print("\n===== Hard-case mining summary =====")
    print(f"  Mining mode:                {args.mode}")
    print(f"  Total detections:           {total_considered}")
    print(f"  Saved crops:                {total_saved}")
    print(f"  Skipped (small <{args.min_crop_px}px):     {total_skipped_small}")
    print(f"  Skipped by temporal dedup:  {total_skipped}")
    print(f"  Reduction:                  {reduction:.1f}%")
    if saved_by_class:
        print("  Saved crops by predicted class:")
        for cname, cnt in sorted(saved_by_class.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {cname}: {cnt}")
    print(f"  Crops -> {crops_dir}")
    print(f"  Metadata -> {metadata_path}")
    if args.dedup_mode == "standard" and reduction < 60.0:
        print("  Hint: reduction < 60%. Try '--dedup-mode aggressive' for convoy footage.")
    print(f"Next: python scripts/deduplicate.py --crops {crops_dir} --metadata {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
