#!/usr/bin/env python3
"""
test_detection.py

Test YOLO detection on observation snapshots/frames and output detailed diagnostics.
Use this to verify the detection pipeline is working and to tune Config UI settings.

This script DIRECTLY tests YOLO on known bird images without RTSP overhead,
helping isolate whether detection issues are in the model or the stream.

Usage:
    # Test with latest observation snapshot
    uv run python scripts/test_detection.py

    # Test with specific observation ID
    uv run python scripts/test_detection.py --observation-id 28

    # Test with custom detection settings
    uv run python scripts/test_detection.py --conf 0.05 --min-area 400

    # Test all recent observations and generate a report
    uv run python scripts/test_detection.py --batch --limit 10

    # Save annotated output image
    uv run python scripts/test_detection.py --save-annotated /tmp/annotated.jpg

Requirements:
    - ultralytics (YOLO)
    - opencv-python

Output includes:
    - Detection count, confidence, bounding boxes, area
    - Recommendations for Config UI settings (detect_conf, min_box_area)
    - Comparison with original observation metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_db_url() -> str:
    """Get database URL from environment or default.
    
    Handles Docker hostnames by translating them to localhost.
    """
    url = os.getenv("HBMON_DB_ASYNC_URL", "")
    if not url:
        url = os.getenv("DATABASE_URL", "postgresql://hbmon:hbmon@localhost:5432/hbmon")
    # Remove async driver suffix
    url = url.replace("+asyncpg", "").replace("+psycopg", "")
    # Translate Docker hostnames to localhost for host-run scripts
    url = url.replace("@hbmon-db:", "@localhost:")
    return url


def get_media_dir() -> Path:
    """Get media directory from environment or default."""
    return Path(os.getenv("HBMON_MEDIA_DIR", "./data/media"))


def query_observation(obs_id: int | None = None, limit: int = 1) -> list[dict]:
    """Query observation(s) from the database."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    db_url = get_db_url()
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    
    conn_params = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "hbmon",
        "password": parsed.password or "hbmon",
        "dbname": parsed.path.lstrip("/") or "hbmon",
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        
        if obs_id:
            query = """
                SELECT id, ts, species_label, snapshot_path, video_path,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2, extra_json
                FROM observations WHERE id = %s
            """
            cur.execute(query, (obs_id,))
        else:
            query = f"""
                SELECT id, ts, species_label, snapshot_path, video_path,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2, extra_json
                FROM observations
                WHERE snapshot_path IS NOT NULL AND snapshot_path != ''
                ORDER BY id DESC LIMIT {limit}
            """
            cur.execute(query)
        
        rows = cur.fetchall()
        results = []
        for row in rows:
            extra = None
            if row[9]:
                try:
                    extra = json.loads(row[9]) if isinstance(row[9], str) else row[9]
                except Exception:
                    pass
            # Extract ROI path from snapshots metadata if available
            roi_path = None
            if extra and isinstance(extra, dict):
                snapshots = extra.get("snapshots", {})
                if isinstance(snapshots, dict) and snapshots.get("roi_path"):
                    roi_path = snapshots["roi_path"]
            results.append({
                "id": row[0],
                "ts": row[1],
                "species_label": row[2],
                "snapshot_path": row[3],
                "video_path": row[4],
                "original_bbox": (row[5], row[6], row[7], row[8]) if row[5] else None,
                "extra": extra,
                "roi_path": roi_path,
            })
        
        cur.close()
        conn.close()
        return results
        
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        sys.exit(1)


def load_yolo_model(model_name: str = "yolo11n.pt", device: str = "cpu"):
    """Load YOLO model for testing."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading YOLO model: {model_name} on {device}...")
    t0 = time.time()
    model = YOLO(model_name, task="detect")
    print(f"  Loaded in {time.time() - t0:.2f}s")
    
    # Get bird class ID
    bird_class_id = None
    names = getattr(model, 'names', None)
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).strip().lower() == 'bird':
                bird_class_id = int(k)
                break
    
    if bird_class_id is None:
        bird_class_id = 14  # COCO default
    
    print(f"  Bird class ID: {bird_class_id}")
    return model, bird_class_id


def run_detection(
    model,
    image_path: Path,
    conf: float,
    iou: float,
    bird_class_id: int,
    min_box_area: int,
    imgsz: int | str = "auto",
) -> dict:
    """Run YOLO detection on an image and return detailed results.
    
    Args:
        imgsz: Image size for YOLO. "auto" uses the image's dimensions snapped to 32-stride.
    """
    import cv2
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"Failed to load image: {image_path}"}
    
    h, w = img.shape[:2]
    
    # Auto-detect imgsz from image dimensions (snap to 32-stride like worker)
    if imgsz == "auto":
        # Snap to 32-pixel stride
        imgsz_h = ((h + 31) // 32) * 32
        imgsz_w = ((w + 31) // 32) * 32
        imgsz_val = (imgsz_h, imgsz_w)  # YOLO expects (height, width)
    else:
        imgsz_val = int(imgsz)
    
    # Run inference
    t0 = time.time()
    results = model.predict(
        img,
        conf=conf,
        iou=iou,
        classes=[bird_class_id],
        imgsz=imgsz_val,
        verbose=False,
    )
    inference_time = (time.time() - t0) * 1000
    
    # Process results
    detections = []
    r0 = results[0] if results else None
    
    if r0 and r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
            box_conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": box_conf,
                "area": area,
                "passes_min_area": area >= min_box_area,
            })
    
    # Sort by confidence
    detections.sort(key=lambda d: d["conf"], reverse=True)
    
    return {
        "image_path": str(image_path),
        "image_size": (w, h),
        "inference_time_ms": inference_time,
        "total_detections": len(detections),
        "detections_passing_min_area": sum(1 for d in detections if d["passes_min_area"]),
        "detections": detections,
        "settings_used": {
            "conf": conf,
            "iou": iou,
            "min_box_area": min_box_area,
            "imgsz": imgsz_val,
        },
    }


def print_detection_report(obs: dict, result: dict, show_recommendations: bool = True):
    """Print detailed detection report with diagnostics."""
    print("\n" + "=" * 70)
    print(f"OBSERVATION #{obs['id']}")
    print("=" * 70)
    
    print(f"\n📷 Image: {result['image_path']}")
    print(f"   Size: {result['image_size'][0]}x{result['image_size'][1]}")
    print(f"   Species (recorded): {obs['species_label']}")
    
    if obs.get('original_bbox'):
        bbox = obs['original_bbox']
        orig_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(f"   Original bbox: {bbox}")
        print(f"   Original area: {orig_area} px²")
    
    # Detection settings used
    settings = result['settings_used']
    print("\n⚙️  Settings used:")
    print(f"   detect_conf: {settings['conf']}")
    print(f"   detect_iou: {settings['iou']}")
    print(f"   min_box_area: {settings['min_box_area']}")
    print(f"   imgsz: {settings['imgsz']}")
    
    # Inference timing
    print(f"\n⏱️  Inference: {result['inference_time_ms']:.1f}ms")
    
    # Detection results
    print("\n🔍 Detections:")
    print(f"   Total raw detections: {result['total_detections']}")
    print(f"   Passing min_box_area filter: {result['detections_passing_min_area']}")
    
    if result['detections']:
        print(f"\n   {'#':<3} {'Confidence':<12} {'BBox':<25} {'Area':<10} {'Passes Filter':<15}")
        print("   " + "-" * 65)
        for i, det in enumerate(result['detections'], 1):
            bbox_str = f"({det['bbox'][0]},{det['bbox'][1]})-({det['bbox'][2]},{det['bbox'][3]})"
            passes = "✓ YES" if det['passes_min_area'] else "✗ NO"
            print(f"   {i:<3} {det['conf']:.4f}       {bbox_str:<25} {det['area']:<10} {passes:<15}")
    else:
        print("   ⚠️  NO DETECTIONS FOUND")
    
    # Recommendations
    if show_recommendations:
        print("\n💡 Config UI Recommendations:")
        
        if result['total_detections'] == 0:
            print("   ⚠️  No detections at current conf={:.2f}".format(settings['conf']))
            print("   → Try lowering 'Detection confidence' to 0.05 or 0.01")
            print("   → Check if image has a visible bird")
            print("   → Consider using a larger YOLO model (yolo11s.pt)")
        elif result['detections_passing_min_area'] == 0:
            smallest = min(d['area'] for d in result['detections'])
            print("   ⚠️  Detections found but all filtered by min_box_area")
            print(f"   → Smallest detection area: {smallest} px²")
            print(f"   → Current min_box_area: {settings['min_box_area']}")
            print(f"   → Try lowering 'Minimum box area' to {max(100, smallest - 100)}")
        else:
            top = result['detections'][0]
            print("   ✓ Detection successful!")
            print(f"   → Best detection: conf={top['conf']:.4f}, area={top['area']} px²")
            
            if top['conf'] < 0.15:
                print(f"   → Low confidence ({top['conf']:.2f}) - consider enabling verbose logging")
            if settings['conf'] > top['conf'] - 0.02:
                print("   → Detection is near threshold - consider lowering detect_conf slightly")
    
    return result['detections_passing_min_area'] > 0


def save_annotated_image(image_path: Path, detections: list, output_path: Path):
    """Save image with detection boxes drawn."""
    import cv2
    
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = (0, 255, 0) if det['passes_min_area'] else (0, 165, 255)  # Green or orange
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{det['conf']:.2f} ({det['area']}px²)"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(str(output_path), img)
    print(f"\n📸 Saved annotated image: {output_path}")
    return True


def print_result(result: dict):
    """Refactored print utility for single image result."""
    # Create a mock observation dict for the report
    mock_obs = {
        "id": "N/A",
        "species_label": "Unknown",
        "original_bbox": None,
    }
    print_detection_report(mock_obs, result, show_recommendations=True)

# Alias for compatibility with previous code edit
annotate_image = save_annotated_image


def run_confidence_sweep(
    model, 
    image_path: Path, 
    bird_class_id: int, 
    imgsz: int | str = "auto"
) -> list[dict]:
    """Test multiple confidence thresholds to find optimal setting."""
    sweep_data = []
    # Test a broad range of thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for conf in thresholds:
        res = run_detection(
            model,
            image_path,
            conf=conf,
            iou=0.45,  # Using standard IOU for sweep
            bird_class_id=bird_class_id,
            min_box_area=600,
            imgsz=imgsz
        )
        sweep_data.append({
            "conf": conf,
            "count": len(res["detections"]),
            "max_conf": res["max_conf"],
            "avg_conf": res["avg_conf"]
        })
    return sweep_data


def print_sweep_table(results: list[dict]):
    """Print sweep results in a formatted table."""
    print("\nConfidence Sweep Results:")
    print(f"{'Conf':<10} | {'Detections':<12} | {'Max Conf':<10} | {'Avg Conf':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['conf']:<10.2f} | {r['count']:<12} | {r['max_conf']:<10.3f} | {r['avg_conf']:<10.3f}")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test YOLO detection on observation snapshots with detailed diagnostics."
    )
    ap.add_argument(
        "--observation-id", "-o",
        type=int,
        default=None,
        help="Test specific observation by ID"
    )
    ap.add_argument(
        "--conf", "-c",
        type=float,
        default=0.1,
        help="Detection confidence threshold (default: 0.1)"
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold (default: 0.45)"
    )
    ap.add_argument(
        "--min-area", "-a",
        type=int,
        default=600,
        help="Minimum box area in pixels (default: 600)"
    )
    ap.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n.pt",
        help="YOLO model to use (default: yolo11n.pt)"
    )
    ap.add_argument(
        "--imgsz",
        type=str,
        default="auto",
        help="YOLO inference image size (default: auto - uses ROI dimensions)"
    )
    ap.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Test multiple observations"
    )
    ap.add_argument(
        "--limit", "-n",
        type=int,
        default=5,
        help="Number of observations to test in batch mode (default: 5)"
    )
    ap.add_argument(
        "--save-annotated", "-s",
        type=str,
        default=None,
        help="Save annotated image to this path"
    )
    ap.add_argument(
        "--sweep-conf",
        action="store_true",
        help="Test multiple confidence thresholds to find optimal setting"
    )
    ap.add_argument(
        "--image-path", "-p",
        type=str,
        default=None,
        help="Path to a specific image file to test (bypasses observation database)"
    )
    args = ap.parse_args()

    media_dir = get_media_dir()
    
    # Handle direct image path mode
    if args.image_path:
        model, bird_class_id = load_yolo_model(args.model)
        
        print("\n" + "=" * 70)
        print("SINGLE IMAGE TEST")
        print("=" * 70)
        print(f"Testing image: {args.image_path}")
        
        result = run_detection(
            model,
            Path(args.image_path),
            args.conf,
            args.iou,
            bird_class_id,
            args.min_area,
            imgsz=args.imgsz
        )
        print_result(result)
        
        if args.sweep_conf:
            print("\nRunning Confidence Sweep...")
            sweep_results = run_confidence_sweep(
                model, 
                Path(args.image_path), 
                bird_class_id, 
                imgsz=args.imgsz
            )
            print_sweep_table(sweep_results)
        
        if args.save_annotated and result["detections"]:
             annotate_image(
                 Path(args.image_path),
                 args.save_annotated,
                 result["detections"]
             )
        
        return 0

    # Query observations
    if args.observation_id:
        observations = query_observation(obs_id=args.observation_id)
    else:
        observations = query_observation(limit=args.limit if args.batch else 1)
    
    if not observations:
        print("ERROR: No observations found", file=sys.stderr)
        return 1
    
    # Load YOLO model
    model, bird_class_id = load_yolo_model(args.model)
    
    print("\n" + "=" * 70)
    print("DETECTION PIPELINE TEST")
    print("=" * 70)
    print(f"Testing {len(observations)} observation(s)")
    print(f"Media directory: {media_dir}")
    
    # Confidence sweep mode
    if args.sweep_conf and len(observations) == 1:
        obs = observations[0]
        
        # Prefer ROI image (what YOLO actually ran on) over full snapshot
        if obs.get("roi_path"):
            snapshot_path = media_dir / obs["roi_path"]
        else:
            snapshot_path = media_dir / obs["snapshot_path"]
        
        if not snapshot_path.exists():
            # Fall back to snapshot if ROI doesn't exist
            snapshot_path = media_dir / obs["snapshot_path"]
            
        if not snapshot_path.exists():
            print(f"ERROR: Image not found: {snapshot_path}")
            return 1
        
        print(f"\n📊 Confidence Threshold Sweep for Observation #{obs['id']}")
        print("-" * 70)
        
        conf_levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        print(f"{'Conf':<8} {'Detections':<12} {'Best Conf':<12} {'Best Area':<12}")
        print("-" * 50)
        
        for conf in conf_levels:
            result = run_detection(
                model, snapshot_path, conf, args.iou, bird_class_id, args.min_area, args.imgsz
            )
            n_det = result['detections_passing_min_area']
            if result['detections']:
                best = result['detections'][0]
                print(f"{conf:<8.2f} {n_det:<12} {best['conf']:<12.4f} {best['area']:<12}")
            else:
                print(f"{conf:<8.2f} {n_det:<12} {'N/A':<12} {'N/A':<12}")
        
        print("\n💡 Recommendation: Set detect_conf just below the lowest threshold")
        print("   where detections appear, with some margin for safety.")
        return 0
    
    # Normal test mode
    success_count = 0
    
    for obs in observations:
        # Prefer ROI image (what YOLO actually ran on) over full snapshot
        if obs.get("roi_path"):
            image_path = media_dir / obs["roi_path"]
        else:
            image_path = media_dir / obs["snapshot_path"]
        
        if not image_path.exists():
            # Fall back to snapshot if ROI doesn't exist
            image_path = media_dir / obs["snapshot_path"]
        
        if not image_path.exists():
            print(f"\n⚠️  Observation #{obs['id']}: Image not found at {image_path}")
            continue
        
        result = run_detection(
            model, image_path, args.conf, args.iou, bird_class_id, args.min_area, args.imgsz
        )
        
        success = print_detection_report(obs, result, show_recommendations=not args.batch)
        if success:
            success_count += 1
        
        if args.save_annotated and not args.batch:
            save_annotated_image(image_path, result['detections'], Path(args.save_annotated))
    
    # Summary for batch mode
    if args.batch:
        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print(f"Tested: {len(observations)} observations")
        print(f"Detected: {success_count} ({100*success_count/len(observations):.0f}%)")
        
        if success_count < len(observations):
            print(f"\n💡 {len(observations) - success_count} observation(s) had no detections.")
            print("   Consider:")
            print("   → Lowering detect_conf (currently {:.2f})".format(args.conf))
            print("   → Lowering min_box_area (currently {})".format(args.min_area))
            print("   → Running with --sweep-conf on individual observations")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
