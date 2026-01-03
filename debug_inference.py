from ultralytics import YOLO
import sys

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model
results = model(sys.argv[1])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(f"Detections in {sys.argv[1]}:")
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        print(f"  Class: {cls_id} ({result.names[cls_id]}), Conf: {conf:.4f}, Box: {xyxy}")
