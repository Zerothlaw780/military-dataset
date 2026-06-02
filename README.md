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

