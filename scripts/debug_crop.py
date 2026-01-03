
import cv2
import json
import sys
from pathlib import Path

def main():
    # Load config
    try:
        with open("data/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config not found")
        sys.exit(1)

    roi = config.get("roi", {})
    if not roi:
        print("No ROI in config")
        sys.exit(1)
        
    img_path = Path("data/media/debug_latest.jpg")
    if not img_path.exists():
        print(f"{img_path} not found")
        sys.exit(1)
        
    frame = cv2.imread(str(img_path))
    if frame is None:
        print("Failed to load image")
        sys.exit(1)
        
    h, w = frame.shape[:2]
    
    # Calculate coords
    x1 = int(roi["x1"] * w)
    y1 = int(roi["y1"] * h)
    x2 = int(roi["x2"] * w)
    y2 = int(roi["y2"] * h)
    
    # Clamp
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    
    crop = frame[y1:y2, x1:x2]
    
    out_path = "data/media/debug_crop.jpg"
    cv2.imwrite(out_path, crop)
    print(f"Saved crop to {out_path} ({crop.shape})")

if __name__ == "__main__":
    main()
