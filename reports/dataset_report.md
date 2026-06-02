## Dataset Analysis Report

This report summarizes dataset structure, label health, class distributions, and basic bounding-box statistics.

## Russian Military

- **Root**: `/Users/aliaydin/military-dataset/datasets/russian_military/Russian-military-annotated`
- **YAML**: `/Users/aliaydin/military-dataset/datasets/russian_military/Russian-military-annotated/data.yaml`
- **YAML nc**: 6
- **YAML names**: ['bmd-2', 'bmp-1', 'bmp-2', 'btr-70', 'btr-80', 'mt-lb']
- **Images**: 595
- **Label files present**: 595
- **Images missing label file**: 0
- **Total boxes (raw)**: 708

### Splits
- **train**: 595 images, 595 label files, 708 boxes

### Label health
- **Empty label files**: 0
- **Invalid boxes** (out of range / non-positive w/h): 0
- **Negative class ids**: 0
- **Parse error lines**: 0

### Observed classes (by id)
- **Observed class ids**: [0, 1, 2, 3, 4, 5]
- **Num observed classes**: 6

| class_id | count |
|---:|---:|
| 4 | 130 |
| 5 | 125 |
| 1 | 121 |
| 2 | 115 |
| 0 | 114 |
| 3 | 103 |

### Bounding boxes (normalized YOLO, valid boxes only)
- **Valid boxes used**: 708

| metric | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| area (w*h) | 0.0100265 | 0.043974 | 0.0853521 | 0.245896 | 0.439436 | 0.615266 | 0.764241 | 0.830006 | 0.952097 |
| aspect ratio (w/h) | 0.447427 | 0.692697 | 0.843629 | 1.02368 | 1.25584 | 1.54212 | 1.86727 | 2.08685 | 2.53312 |

### Duplicate signals
- **Duplicate label-file groups** (identical content): 0 (total files in groups: 0)
- **Duplicate image groups** (identical bytes): 0 (total files in groups: 0)
- **Note**: Image duplicates are only computed when --hash-images is enabled.

## Aboba

- **Root**: `/Users/aliaydin/military-dataset/datasets/aboba/aboba`
- **YAML**: `/Users/aliaydin/military-dataset/datasets/aboba/aboba/data.yaml`
- **YAML nc**: 12
- **YAML names**: ['---', 'BMP-BMD', 'BTR', 'KAMAZ', 'MTLB', 'REB', 'RLS', 'RSZO', 'SAU', 'TANK', 'URAL', 'ZRK']
- **Images**: 4386
- **Label files present**: 4386
- **Images missing label file**: 0
- **Total boxes (raw)**: 5424

### Splits
- **train**: 4386 images, 4386 label files, 5424 boxes

### Label health
- **Empty label files**: 45
- **Invalid boxes** (out of range / non-positive w/h): 0
- **Negative class ids**: 0
- **Parse error lines**: 0

### Observed classes (by id)
- **Observed class ids**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- **Num observed classes**: 12

| class_id | count |
|---:|---:|
| 9 | 2934 |
| 1 | 1305 |
| 2 | 363 |
| 4 | 288 |
| 8 | 180 |
| 11 | 108 |
| 7 | 99 |
| 10 | 63 |
| 3 | 45 |
| 5 | 18 |
| 6 | 12 |
| 0 | 9 |

### Bounding boxes (normalized YOLO, valid boxes only)
- **Valid boxes used**: 5424

| metric | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| area (w*h) | 0.000349683 | 0.000788574 | 0.00144678 | 0.0032373 | 0.00836426 | 0.024939 | 0.0645381 | 0.108511 | 0.279223 |
| aspect ratio (w/h) | 0.492146 | 0.669755 | 0.772727 | 1.01231 | 1.32874 | 1.72222 | 2.10782 | 2.41176 | 3.07091 |

### Duplicate signals
- **Duplicate label-file groups** (identical content): 85 (total files in groups: 215)
- **Duplicate image groups** (identical bytes): 0 (total files in groups: 0)

- **Largest label-duplicate groups (samples)**:
  - 45 files: `train/labels/100_png.rf.69db2c108a78cee1f7d92449609615ee.txt`, `train/labels/183_png.rf.03c6ae19c488a568974d0c0df7d454e8.txt`, `train/labels/175_png.rf.4df8f4a23f076a715b4f43aee7abb7b6.txt` (+42 more)
  - 3 files: `train/labels/bandicam-2023-10-16-17-52-19-570_jpg.rf.45e7ad495f8cb64087282df955759419.txt`, `train/labels/bandicam-2023-10-16-17-52-19-570_jpg.rf.ac28168252cfe6bd533f039e9b802f0d.txt`, `train/labels/bandicam-2023-10-16-17-52-19-570_jpg.rf.6430bea09beedfa3115e7a30fc3bb132.txt`
  - 3 files: `train/labels/44_png.rf.a3d5d11c8bd32b773a45fcba3d9e56b3.txt`, `train/labels/44_png.rf.62c17c00ee4ee42662ff61c8a547044e.txt`, `train/labels/44_png.rf.61e352c73138c1b38b5feae8ba1e74b7.txt`
  - 2 files: `train/labels/137_png.rf.fd77b241170c9a219b0d0aa3002249f8.txt`, `train/labels/137_png.rf.273529125360d758144c219bea205275.txt`
  - 2 files: `train/labels/IMG_20231018_192700_471_0036-0_png.rf.1e5a1a12e9a0fedbb1f476a8b1bf25f1.txt`, `train/labels/IMG_20231018_192700_471_0036-0_png.rf.6403bfc96e663c66376875a417fc6e2f.txt`
- **Note**: Image duplicates are only computed when --hash-images is enabled.

## VisDrone

- **Root**: `/Users/aliaydin/military-dataset/datasets/visdrone/VisDrone_Veri`
- **YAML**: `/Users/aliaydin/military-dataset/datasets/visdrone/VisDrone_Veri/visdrone.yaml`
- **YAML nc**: 6
- **YAML names**: ['pedestrian', 'people', 'car', 'van', 'truck', 'bus']
- **Images**: 7019
- **Label files present**: 7019
- **Images missing label file**: 0
- **Total boxes (raw)**: 326029

### Splits
- **valid**: 548 images, 548 label files, 31009 boxes
- **train**: 6471 images, 6471 label files, 295020 boxes

### Label health
- **Empty label files**: 0
- **Invalid boxes** (out of range / non-positive w/h): 1
- **Negative class ids**: 0
- **Parse error lines**: 0

### Observed classes (by id)
- **Observed class ids**: [0, 1, 2, 3, 4, 5]
- **Num observed classes**: 6

| class_id | count |
|---:|---:|
| 2 | 158930 |
| 0 | 88181 |
| 1 | 32184 |
| 3 | 26931 |
| 4 | 13625 |
| 5 | 6177 |

### Bounding boxes (normalized YOLO, valid boxes only)
- **Valid boxes used**: 326028

| metric | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| area (w*h) | 1.83325e-05 | 4.35115e-05 | 7.0745e-05 | 0.000174034 | 0.000486126 | 0.00146849 | 0.00397342 | 0.00691668 | 0.0179224 |
| aspect ratio (w/h) | 0.165444 | 0.211081 | 0.249978 | 0.351555 | 0.575024 | 0.954937 | 1.41667 | 1.64998 | 2.08333 |

### Duplicate signals
- **Duplicate label-file groups** (identical content): 0 (total files in groups: 0)
- **Duplicate image groups** (identical bytes): 0 (total files in groups: 0)
- **Note**: Image duplicates are only computed when --hash-images is enabled.
