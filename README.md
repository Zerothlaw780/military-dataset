## Military & Civilian Vehicle Detection (Drone Imagery)

This repository contains a dataset engineering and training workspace to **detect military and civilian vehicles from drone imagery** using **YOLO11**.

## Classes

Final model classes (YOLO indices):

- `0`: `tank`
- `1`: `armored_vehicle`
- `2`: `car`
- `3`: `van`
- `4`: `truck`
- `5`: `bus`

## Datasets

Source datasets used in this project:

- **Russian Military**
- **Aboba**
- **VisDrone**

All raw/source datasets live under `datasets/` (ignored by git by default).

## Pipeline

End-to-end workflow implemented in this repo:

1. **Dataset analysis**
2. **Class remapping** (map dataset-specific labels → the 6 final classes above)
3. **Data cleaning**
4. **Duplicate removal**
5. **Dataset merging**
6. **Train/Validation/Test split**
7. **YOLO11 training**
8. **Evaluation**
9. **ONNX export**

## Project structure

```text
military-dataset/
├── datasets/           # raw source datasets (ignored)
│   ├── russian_military/
│   ├── aboba/
│   └── visdrone/
├── merged_dataset/     # merged/normalized dataset output (ignored)
├── scripts/            # dataset processing utilities
├── reports/            # analyses, metrics, and figures
├── training/           # training configs, experiment notes, helpers
├── requirements.txt
└── README.md
```

## Getting started

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training & export (expected outputs)

- **Training runs**: YOLO training outputs typically appear under `runs/` (ignored by git).
- **Model artifacts**: exported weights and formats are ignored by git:
  - PyTorch: `*.pt`
  - ONNX: `*.onnx`
  - TensorRT: `*.engine`

## Notes

- This repo intentionally keeps **code, configs, and documentation tracked**, while ignoring **data, training runs, and exported binaries**.
- If you add a public subset or sample images for documentation, place them in a dedicated tracked folder (e.g. `docs/`) and adjust `.gitignore` accordingly.

## Hard-case mining

The model sometimes labels armored vehicles (BMP, BTR, APC, IFV, damaged armor) as
`civilian_vehicle` in drone footage. The hard-case mining pipeline surfaces these
likely mistakes from videos so they can be reviewed and re-labeled.

It runs in three stages:

1. `scripts/mine_hardcases.py` — run a trained YOLO model over every video and save
   detections classified as `civilian_vehicle` with `confidence >= 0.50`. Saves one
   cropped image per detection plus a `metadata.csv` with full provenance
   (video, frame index, timestamp, confidence, box coordinates).
2. `scripts/deduplicate.py` — collapse near-duplicate crops (consecutive frames of the
   same vehicle) using perceptual hashing (`imagehash`), keeping the highest-confidence
   representative per cluster.
3. `scripts/create_grid.py` — assemble a 10×10 review grid (`hardcases_grid.jpg`) of the
   deduplicated crops, sorted by confidence, for fast human triage.

### Inputs

- A trained model: `runs/detect/<run>/weights/best.pt` (or any `.pt`/`.onnx`/`.engine`).
- A `videos/` folder containing the drone footage to mine.

### Usage

```bash
# 0. Install dependencies (adds ultralytics + opencv)
python -m pip install -r requirements.txt

# 1. Mine confident civilian_vehicle detections from all videos
python scripts/mine_hardcases.py \
    --model runs/detect/train/weights/best.pt \
    --videos videos \
    --output hardcase_mining \
    --conf 0.50 \
    --frame-stride 15

# 2. Remove near-duplicate crops
python scripts/deduplicate.py \
    --crops hardcase_mining/crops \
    --metadata hardcase_mining/metadata.csv \
    --threshold 5

# 3. Build the 10x10 review grid
python scripts/create_grid.py \
    --crops hardcase_mining/crops \
    --metadata hardcase_mining/metadata_dedup.csv \
    --output hardcase_mining/hardcases_grid.jpg
```

### Outputs (under `hardcase_mining/`)

- `crops/` — cropped hard-case detections (`<video>_f<frame>_d<det>_c<conf>.jpg`)
- `metadata.csv` — one row per crop; `metadata_dedup.csv` after deduplication
- `crops_duplicates/` — near-duplicates moved aside by `deduplicate.py`
- `hardcases_grid.jpg` — the review montage

The class name is resolved from the model's own labels; if your weights use different
names, pass `--class-id` to `mine_hardcases.py`. Mined crops/grids are images and are
ignored by git per the existing `.gitignore`.

