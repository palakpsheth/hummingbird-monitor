# Model Fine-Tuning Cookbook

This guide walks you through fine-tuning a YOLO model using your annotated hummingbird data.

## Prerequisites

- **Python 3.11+** with `uv` (or `pip`)
- **NVIDIA GPU** (recommended) or CPU-only mode
- **Annotated observations** with completed frame reviews
- **Docker** (optional, for containerized training)

## Step 1: Prepare Your Annotations

### 1.1 Annotate Observations

1. Navigate to the **Annotate** tab in the web UI
2. Click an observation to start annotation
3. For each frame:
   - Toggle "Bird present" if a bird is visible
   - Draw/adjust bounding boxes around birds
   - Mark false-positive boxes with the FP checkbox
4. Save each frame and continue to the next

### 1.2 Verify Annotation Completeness

Before training, ensure no observations are partially labeled:
- Go to `/annotate` and check for any "In Review" observations
- Complete or reset partial annotations
- Training is blocked if any observations have some (but not all) frames reviewed

## Step 2: Export the Dataset

### 2.1 Run the Export Script

```bash
# From the project root
python scripts/build_yolo_dataset.py --output-dir /data/exports/yolo/dataset

# Options:
#   --train-split 0.8       # Training fraction (default: 80%)
#   --seed 42               # Random seed for reproducibility
#   --include-hard-negatives # Export FP boxes as hard negatives
#   --dry-run               # Preview without writing files
```

### 2.2 Verify the Export

Check the output directory structure:

```
/data/exports/yolo/dataset/
├── images/
│   ├── train/     # Training images
│   └── val/       # Validation images
├── labels/
│   ├── train/     # YOLO format labels
│   └── val/
├── hard_negatives/ # FP crop images
├── dataset.yaml    # YOLO dataset config
└── export_stats.json  # Export statistics
```

## Step 3: Train the Model

### 3.1 Basic Training

```bash
# Install ultralytics if needed
pip install ultralytics

# Train from a pretrained YOLO model
yolo detect train \
    data=/data/exports/yolo/dataset/dataset.yaml \
    model=yolo11n.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    name=hbmon_finetune
```

### 3.2 Training with Hard Negatives

To incorporate hard negatives into training:

```bash
# Option 1: Add hard negatives to training set
cp /data/exports/yolo/dataset/hard_negatives/*.jpg /data/exports/yolo/dataset/images/train/

# Create empty labels for hard negatives (no bird = empty label)
for f in /data/exports/yolo/dataset/hard_negatives/*.jpg; do
    touch /data/exports/yolo/dataset/labels/train/$(basename "${f%.jpg}.txt")
done

# Then train as normal
```

### 3.3 Hyperparameter Tuning

Common parameters to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Training epochs |
| `imgsz` | 640 | Image size (pixels) |
| `batch` | 16 | Batch size |
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final learning rate factor |
| `conf` | 0.25 | Confidence threshold |
| `iou` | 0.7 | IoU threshold for NMS |

## Step 4: Evaluate the Model

### 4.1 Validation Metrics

```bash
yolo detect val \
    data=/data/exports/yolo/dataset/dataset.yaml \
    model=runs/detect/hbmon_finetune/weights/best.pt
```

Key metrics to check:
- **mAP50**: Mean average precision at IoU=0.5
- **mAP50-95**: Mean average precision averaged over IoU thresholds
- **Precision/Recall**: Trade-off depending on your use case

### 4.2 Test on Sample Images

```bash
yolo detect predict \
    model=runs/detect/hbmon_finetune/weights/best.pt \
    source=/path/to/test/images
```

## Step 5: Deploy the Model

### 5.1 Export to OpenVINO (Intel GPUs)

```bash
yolo export \
    model=runs/detect/hbmon_finetune/weights/best.pt \
    format=openvino \
    imgsz=640
```

### 5.2 Update hbmon Configuration

1. Copy the new model weights to `/data/yolo/`
2. Update `HBMON_YOLO_MODEL` environment variable:
   ```bash
   HBMON_YOLO_MODEL=/data/yolo/best.pt
   ```
3. Restart the worker:
   ```bash
   docker compose restart hbmon-worker
   ```

## Troubleshooting

### Training Issues

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch` size or `imgsz` |
| Slow training | Use GPU, reduce image size |
| Poor accuracy | Add more training data, increase epochs |
| Overfitting | Use augmentation, early stopping |

### Dataset Issues

| Problem | Solution |
|---------|----------|
| Empty dataset | Check annotation completion in `/annotate` |
| Missing labels | Verify YOLO labels were generated |
| Imbalanced classes | Add more true negative frames |

### Deployment Issues

| Problem | Solution |
|---------|----------|
| Model not loading | Check file path and permissions |
| Slow inference | Export to OpenVINO for Intel GPUs |
| High false positives | Lower confidence threshold |

## Best Practices

1. **Annotate diverse conditions**: Include day/night, different weather, various bird positions
2. **Mark false positives carefully**: FP boxes become hard negatives for training
3. **Keep training data balanced**: Aim for ~50% positive and ~50% negative frames
4. **Validate on held-out data**: Don't train on data you'll use for testing
5. **Version your models**: Keep track of dataset versions and training configs

## Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
- [OpenVINO Optimization](https://docs.openvino.ai/)
