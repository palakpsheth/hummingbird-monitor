import sys
import os
import logging
import cv2
import time
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("test_obs_53_nms_debug")

# Force env vars for testing
os.environ["HBMON_ANNOTATOR_DEBUG"] = "1"
os.environ["HBMON_ANNOTATION_CONFIDENCE"] = "0.1"
os.environ["HBMON_ANNOTATION_NMS_THRESHOLD"] = "0.5"
os.environ["HBMON_DEBUG_DUMP_IMAGES"] = "1" # Trigger image dump

def test_obs_53_nms_debug():
    try:
        from hbmon.db import get_sync_session
        from hbmon.models import Observation
        from hbmon.annotation_detector import AnnotationDetector
        
        # Clear debug output dir if exists
        debug_dir = Path("/app/debug_output")
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        obs_id = 53
        # Look for frames around the timestamp where overlaps likely (Frame 2 previously)
        # We know Frame 20 has a detection (from stride=10 test, index 2 was frame 20).
        # We'll extract frames 18-23 to be safe.
        target_frame = 20
        start_scan = 18
        limit_frames = 5
        
        with get_sync_session() as db:
            obs = db.get(Observation, obs_id)
            if not obs:
                logger.error(f"Observation {obs_id} not found")
                return
            video_rel_path = obs.video_path
            
        video_path = Path(os.environ.get("HBMON_MEDIA_DIR", "/data/media")) / video_rel_path
        logger.info(f"Video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        # Skip to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_scan)
        
        detect_batch_frames = []
        count = 0
        while count < limit_frames:
            ret, frame = cap.read()
            if not ret:
                break
            detect_batch_frames.append(frame)
            count += 1
        cap.release()
        logger.info(f"Extracted {len(detect_batch_frames)} frames starting from index {start_scan}.")
        
        detector = AnnotationDetector()
        logger.info("Initializing detector...")
        detector.initialize()
        
        logger.info(f"Running batch detection on {len(detect_batch_frames)} frames with NMS debug dump enabled...")
        detector.detect_batch(detect_batch_frames)
        
        # Verify output
        images = list(debug_dir.glob("*.jpg"))
        logger.info(f"Generated {len(images)} debug snapshots in {debug_dir}")
        for img in sorted(images)[:5]:
             logger.info(f" - {img.name}")
             
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obs_53_nms_debug()
