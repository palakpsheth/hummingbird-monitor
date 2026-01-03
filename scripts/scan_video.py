import cv2
import json
import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=None, help="Inference size")
    args = parser.parse_args()
    
    # Load settings
    with open("data/config.json") as f:
        config = json.load(f)
    roi = config.get("roi")
    
    # Default imgsz to 640 if not set, or ensure it's an int
    imgsz = args.imgsz if args.imgsz else 640
    
    print(f"Scanning {args.video_path}...")
    
    # Load model
    model = YOLO(os.environ.get("HBMON_YOLO_MODEL", "yolo11n.pt"))
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return
        
    frame_count = 0
    detections = 0
    max_conf = 0.0
    best_frame = -1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Apply ROI
        if roi:
            x1 = int(roi["x1"] * w)
            y1 = int(roi["y1"] * h)
            x2 = int(roi["x2"] * w)
            y2 = int(roi["y2"] * h)
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            frame = frame[y1:y2, x1:x2]
            
        results = model.predict(frame, conf=args.conf, verbose=False, imgsz=imgsz)
        for r in results:
            for box in r.boxes:
                # Class 14 = bird
                if int(box.cls[0]) == 14:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    detections += 1
                    print(f"Frame {frame_count}: conf={conf:.3f} area={area} bbox={xyxy.tolist()}")
                    
                    if conf > max_conf:
                        max_conf = conf
                        best_frame = frame_count
                    
                    print(f"Frame {frame_count}: conf={conf:.3f} area={int(area)}")

                        
        if frame_count % 20 == 0:
            print(f"Frame {frame_count}: max_conf={max_conf:.3f}", end="\r")
            
    print(f"\nScanned {frame_count} frames.")
    print(f"Total Bird Detections: {detections}")
    print(f"Max Confidence: {max_conf:.4f} (Frame {best_frame})")

if __name__ == "__main__":
    main()
