import sys
import os
import logging
import cv2
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("test_obs_53_extended")

# Force env vars for testing
os.environ["HBMON_ANNOTATOR_DEBUG"] = "1"
os.environ["HBMON_ANNOTATION_CONFIDENCE"] = "0.1"
os.environ["HBMON_REDIS_URL"] = "redis://localhost:6379/0" 
# Use main DB url if needed, assuming available in container env

def test_obs_53_extended():
    try:
        from hbmon.db import get_sync_session
        from hbmon.models import Observation
        from hbmon.annotation_detector import AnnotationDetector
        
        obs_id = 53
        limit_frames = 100
        
        # 1. Get Video Path
        with get_sync_session() as db:
            obs = db.get(Observation, obs_id)
            if not obs:
                logger.error(f"Observation {obs_id} not found")
                return
            
            video_rel_path = obs.video_path
            
        video_path = Path(os.environ.get("HBMON_MEDIA_DIR", "/data/media")) / video_rel_path
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return
            
        logger.info(f"Video found: {video_path}")
        
        # 2. Extract first 100 frames (Sampled)
        logger.info(f"Extracting frames from first {limit_frames} with stride 10...")
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        count = 0
        while count < limit_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 10th frame (indices 0, 10, 20...)
            if count % 10 == 0:
                frames.append(frame)
            
            count += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} sampled frames.")
        
        # 3. Run Detection
        detector = AnnotationDetector() # Will read HBMON_ANNOTATION_CONFIDENCE=0.1
        logger.info("Initializing detector...")
        detector.initialize()
        
        logger.info("Running batch detection...")
        BATCH_SIZE = 4 # Use small batch to see progress
        all_results = []
        
        t0 = time.time()
        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i+BATCH_SIZE]
            batch_results = detector.detect_batch(batch)
            all_results.extend(batch_results)
            logger.info(f"Processed {len(all_results)}/{len(frames)} frames...")
            
        total_time = time.time() - t0
        logger.info(f"Detection complete in {total_time:.2f}s")
        
        # 4. Count Detections
        frames_with_birds = 0
        total_boxes = 0
        
        for i, boxes in enumerate(all_results):
            if len(boxes) > 0:
                frames_with_birds += 1
                total_boxes += len(boxes)
                # logger.info(f"Frame {i}: {len(boxes)} boxes") # Too verbose?
                
        logger.info("-" * 40)
        logger.info(f"RESULTS for Obs {obs_id} (First 100 Frames)")
        logger.info(f"Confidence Threshold: {os.environ['HBMON_ANNOTATION_CONFIDENCE']}")
        logger.info(f"Frames with at least one bird: {frames_with_birds} / {len(frames)}")
        logger.info(f"Total bird boxes detected: {total_boxes}")
        logger.info("-" * 40)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obs_53_extended()
