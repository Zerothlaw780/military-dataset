#!/usr/bin/env python3
"""Build a 10x10 review grid (montage) of hard-case crops.

The grid lets a human quickly scan the mined ``civilian_vehicle`` crops and flag
the ones that are actually armored vehicles. Each cell shows the crop plus its
confidence (when ``metadata.csv`` is provided).

Example:
    python scripts/create_grid.py \
        --crops hardcase_mining/crops \
        --metadata hardcase_mining/metadata_dedup.csv \
        --output hardcase_mining/hardcases_grid.jpg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _import_deps():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ImportError:
        _fail("Pillow is required. Install with: pip install Pillow")
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


def load_confidences(metadata_path: Path | None) -> dict[str, float]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    conf: dict[str, float] = {}
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("crop_file")
            if not name:
                continue
            try:
                conf[name] = float(row.get("confidence", 0.0) or 0.0)
            except ValueError:
                conf[name] = 0.0
    return conf


def order_crops(crops: list[Path], confidences: dict[str, float], sort: str) -> list[Path]:
    if sort == "name":
        return sorted(crops, key=lambda p: p.name)
    reverse = sort == "conf-desc"
    return sorted(crops, key=lambda p: (confidences.get(p.name, -1.0), p.name), reverse=reverse)


def build_grid(
    crops: list[Path],
    *,
    Image,
    ImageDraw,
    ImageFont,
    rows: int,
    cols: int,
    cell: int,
    pad: int,
    confidences: dict[str, float],
    bg=(20, 20, 20),
):
    per_page = rows * cols
    label_h = 16
    grid_w = cols * cell + (cols + 1) * pad
    grid_h = rows * (cell + label_h) + (rows + 1) * pad

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(canvas)

    for idx, crop_path in enumerate(crops[:per_page]):
        rr, cc = divmod(idx, cols)
        x0 = pad + cc * (cell + pad)
        y0 = pad + rr * (cell + label_h + pad)
        try:
            with Image.open(crop_path) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                off_x = x0 + (cell - im.width) // 2
                off_y = y0 + (cell - im.height) // 2
                canvas.paste(im, (off_x, off_y))
        except Exception as e:
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(120, 0, 0))
            draw.text((x0 + 2, y0 + 2), "ERR", fill=(255, 80, 80), font=font)
            print(f"  WARNING: could not render {crop_path.name}: {e}")

        conf = confidences.get(crop_path.name)
        label = f"{conf:.2f}" if conf is not None else crop_path.stem[:14]
        draw.text((x0 + 2, y0 + cell + 2), label, fill=(230, 230, 230), font=font)

    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a review grid of hard-case crops.")
    ap.add_argument("--crops", required=True, type=Path, help="Folder of crop images.")
    ap.add_argument("--metadata", type=Path, default=None, help="Optional metadata CSV for confidence labels/sorting.")
    ap.add_argument("--output", type=Path, default=Path("hardcase_mining/hardcases_grid.jpg"), help="Output grid image.")
    ap.add_argument("--rows", type=int, default=10, help="Grid rows.")
    ap.add_argument("--cols", type=int, default=10, help="Grid columns.")
    ap.add_argument("--cell", type=int, default=128, help="Cell size in pixels.")
    ap.add_argument("--pad", type=int, default=4, help="Padding between cells.")
    ap.add_argument(
        "--sort",
        choices=("conf-desc", "conf-asc", "name"),
        default="conf-desc",
        help="Ordering of crops (conf-desc surfaces the most confident misclassifications first).",
    )
    ap.add_argument("--all-pages", action="store_true", help="Emit multiple pages if crops exceed one grid.")
    args = ap.parse_args()

    if not args.crops.is_dir():
        _fail(f"crops folder not found: {args.crops}")
    if args.rows < 1 or args.cols < 1:
        _fail("--rows and --cols must be >= 1")

    Image, ImageDraw, ImageFont = _import_deps()

    crops = sorted(p for p in args.crops.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not crops:
        _fail(f"no crop images found in {args.crops}")

    confidences = load_confidences(args.metadata)
    crops = order_crops(crops, confidences, args.sort)

    per_page = args.rows * args.cols
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.all_pages or len(crops) <= per_page:
        grid = build_grid(
            crops,
            Image=Image,
            ImageDraw=ImageDraw,
            ImageFont=ImageFont,
            rows=args.rows,
            cols=args.cols,
            cell=args.cell,
            pad=args.pad,
            confidences=confidences,
        )
        grid.save(args.output, quality=90)
        shown = min(len(crops), per_page)
        print(f"Wrote {args.output} ({shown}/{len(crops)} crops, {args.rows}x{args.cols}).")
        if len(crops) > per_page:
            print(f"Note: {len(crops) - per_page} crop(s) not shown. Use --all-pages to emit more grids.")
        return 0

    pages = (len(crops) + per_page - 1) // per_page
    for page in range(pages):
        chunk = crops[page * per_page : (page + 1) * per_page]
        grid = build_grid(
            chunk,
            Image=Image,
            ImageDraw=ImageDraw,
            ImageFont=ImageFont,
            rows=args.rows,
            cols=args.cols,
            cell=args.cell,
            pad=args.pad,
            confidences=confidences,
        )
        out = args.output if pages == 1 else args.output.with_name(
            f"{args.output.stem}_{page + 1:03d}{args.output.suffix}"
        )
        grid.save(out, quality=90)
        print(f"Wrote {out} ({len(chunk)} crops).")
    print(f"Done. {pages} page(s) for {len(crops)} crop(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
