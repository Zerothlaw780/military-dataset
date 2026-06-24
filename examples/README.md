# examples/

Qualitative example detections from the final **YOLO11m** model, primarily on
real-world UAV footage.

Add annotated frames or short result clips here to showcase model behavior on:

- `tank` detections (e.g. T-72/T-90 in convoy),
- `armored_vehicle` detections (BMP/BTR/MT-LB/APC),
- `civilian_vehicle` detections, and
- mixed scenes containing all three classes.

### Generate examples

```bash
# Annotated frames/video from a clip
yolo detect predict model=models/best.pt source=videos/<clip>.mp4 save=True conf=0.5

# Then copy the most illustrative outputs here
cp runs/detect/predict/<frame>.jpg examples/
```

> Prefer a few high-quality, representative images over many near-duplicates.
> The two headline images used in the main README live in [`docs/`](../docs/).
