## Russian Military: invalid boxes removed during cleaning

During dataset preparation (`scripts/prepare_clean.py`), **70 bounding boxes** were dropped from Russian Military labels because they failed YOLO validity checks in `scripts/yolo_io.py` (`is_valid_box`). None were dropped due to class remapping (all six source classes map to `armored_vehicle`).

### Summary

| Metric | Count |
|---|---:|
| Total label files scanned | 595 |
| Invalid boxes removed | **70** |
| Images with only invalid box(es) (excluded entirely) | **54** |
| Images with valid + invalid boxes (invalid dropped, image kept) | **16** |
| Malformed label lines (unparseable) | **0** |

### Counts by rejection category

| Category | Boxes |
|---|---:|
| out_of_bounds | 70 |
| negative_coordinates | 0 |
| zero_area | 0 |
| malformed | 0 |

**All 70 removed boxes are `out_of_bounds`**: their normalized corners extend outside the `[0, 1] × [0, 1]` image rectangle (tolerance `1e-6`). There were **no** zero-area, negative-coordinate, or malformed boxes in this dataset.

### Out-of-bounds subtypes (tags can overlap on one box)

| Subtype | Boxes | Meaning |
|---|---:|---|
| `extends_past_left` | 21 | x − w/2 < 0 |
| `extends_past_top` | 3 | y − h/2 < 0 |
| `extends_past_right` | 28 | x + w/2 > 1 |
| `extends_past_bottom` | 21 | y + h/2 > 1 |
| `width_or_height_exceeds_1` | 0 | w > 1 or h > 1 (invalid normalized size) |

### Invalid boxes by original class

| Original class | Boxes |
|---|---:|
| btr-70 | 20 |
| btr-80 | 19 |
| bmp-2 | 11 |
| mt-lb | 10 |
| bmd-2 | 7 |
| bmp-1 | 3 |

---

## Validation rules (pipeline)

A box is **kept** only if:

1. `w > 0` and `h > 0` (non-zero area)
2. Center `(x, y)` in `[0, 1]`
3. `w` and `h` in `(0, 1]`
4. Corners `(x ± w/2, y ± h/2)` inside `[0, 1]`

Malformed lines (`< 5` fields or non-numeric values) are skipped at parse time and do not appear in box counts.

---

## Category details

### 1. Out of bounds (70 boxes)

Typical cause in this dataset: **full-frame or near full-frame Roboflow exports** where width `w ≈ 1.0` (or `w > 0.99`) and/or height `h` is large, so the computed bottom or right edge exceeds `1.0` by a small amount due to floating-point rounding or slightly loose annotation.

**Examples:**

- **train/labels/7_jpg.rf.GRG8agpR6MPEkIPlhsU8.txt** (line 1, class `mt-lb`)
  - Raw: `5 0.5 0.5725370370370371 1 0.8549444444444445`
  - Center (x, y)=(0.5, 0.572537), (w, h)=(1, 0.854944)
  - Corners (x1,y1,x2,y2)=(0, 0.145065, 1, 1.00001)
  - Overflow: left=0.00e+00, top=0.00e+00, right=0.00e+00, bottom=9.26e-06
  - Tags: `extends_past_bottom`

- **train/labels/64_jpg.rf.UBiLFk0RbwxdnRy5eP5r.txt** (line 1, class `btr-80`)
  - Raw: `4 0.3569 0.7839643652561247 0.7138166666666667 0.43207126948775054`
  - Center (x, y)=(0.3569, 0.783964), (w, h)=(0.713817, 0.432071)
  - Corners (x1,y1,x2,y2)=(-8.33333e-06, 0.567929, 0.713808, 1)
  - Overflow: left=8.33e-06, top=0.00e+00, right=0.00e+00, bottom=0.00e+00
  - Tags: `extends_past_left`

- **train/labels/51_png.rf.dbDUEG1XvnWabaNsbhqi.txt** (line 1, class `bmp-1`)
  - Raw: `1 0.5 0.6356944444444445 1 0.728638888888889`
  - Center (x, y)=(0.5, 0.635694), (w, h)=(1, 0.728639)
  - Corners (x1,y1,x2,y2)=(0, 0.271375, 1, 1.00001)
  - Overflow: left=0.00e+00, top=0.00e+00, right=0.00e+00, bottom=1.39e-05
  - Tags: `extends_past_bottom`

- **train/labels/41_jpg.rf.2ZU04ppMf6O6KIlu4pNd.txt** (line 1, class `btr-70`)
  - Raw: `3 0.501375 0.4338472222222222 0.9972604166666666 0.60225`
  - Center (x, y)=(0.501375, 0.433847), (w, h)=(0.99726, 0.60225)
  - Corners (x1,y1,x2,y2)=(0.00274479, 0.132722, 1.00001, 0.734972)
  - Overflow: left=0.00e+00, top=0.00e+00, right=5.21e-06, bottom=0.00e+00
  - Tags: `extends_past_right`

- **train/labels/34_jpg.rf.Y94EXFlXPEf0caVSA216.txt** (line 1, class `btr-80`)
  - Raw: `4 0.5019027777777778 0.4702291666666667 0.9962083333333334 0.24879166666666666`
  - Center (x, y)=(0.501903, 0.470229), (w, h)=(0.996208, 0.248792)
  - Corners (x1,y1,x2,y2)=(0.00379861, 0.345833, 1.00001, 0.594625)
  - Overflow: left=0.00e+00, top=0.00e+00, right=6.94e-06, bottom=0.00e+00
  - Tags: `extends_past_right`

- **train/labels/78_jpg.rf.8kVk10nbY2wS4qiqFzPn.txt** (line 1, class `mt-lb`)
  - Raw: `5 0.13644444444444442 0.4930952380952381 0.2729206349206349 0.38933333333333336`
  - Center (x, y)=(0.136444, 0.493095), (w, h)=(0.272921, 0.389333)
  - Corners (x1,y1,x2,y2)=(-1.5873e-05, 0.298429, 0.272905, 0.687762)
  - Overflow: left=1.59e-05, top=0.00e+00, right=0.00e+00, bottom=0.00e+00
  - Tags: `extends_past_left`

- **train/labels/97_jpg.rf.ehcKpLIGV0zI7J3JZSFS.txt** (line 1, class `btr-70`)
  - Raw: `3 0.5786067708333333 0.492783203125 0.8427994791666666 0.356650390625`
  - Center (x, y)=(0.578607, 0.492783), (w, h)=(0.842799, 0.35665)
  - Corners (x1,y1,x2,y2)=(0.157207, 0.314458, 1.00001, 0.671108)
  - Overflow: left=0.00e+00, top=0.00e+00, right=6.51e-06, bottom=0.00e+00
  - Tags: `extends_past_right`

- **train/labels/65_jpg.rf.oSlHWu3AAmOGJy9tyquu.txt** (line 1, class `bmp-2`)
  - Raw: `2 0.553725 0.6719887429643527 0.89255 0.6560412757973734`
  - Center (x, y)=(0.553725, 0.671989), (w, h)=(0.89255, 0.656041)
  - Corners (x1,y1,x2,y2)=(0.10745, 0.343968, 1, 1.00001)
  - Overflow: left=0.00e+00, top=0.00e+00, right=0.00e+00, bottom=9.38e-06
  - Tags: `extends_past_bottom`

### 2. Negative coordinates (0 boxes)

Would apply if center `x`/`y` or size `w`/`h` were negative. **None found** in Russian Military.

### 3. Zero-area boxes (0 boxes)

Would apply if `w <= 0` or `h <= 0`. **None found** in Russian Military.

### 4. Malformed labels (0 lines)

Would apply to lines with fewer than 5 fields or non-numeric values. **None found**; every annotation line parsed successfully.

---

## Impact on merged dataset

- **54 images** had a single invalid box and were removed entirely (`no_boxes_after_remap`).
- **16 images** kept one valid box each after dropping one invalid box.
- Remaining Russian Military after cleaning: **535 images**, **632** `armored_vehicle` instances (see `reports/pipeline_metadata.json`).
