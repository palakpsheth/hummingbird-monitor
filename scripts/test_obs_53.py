import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("test_obs_53")

# Force env vars for testing
os.environ["HBMON_ANNOTATOR_DEBUG"] = "1"
# Ensure we hit the real backend
os.environ["HBMON_REDIS_URL"] = "redis://localhost:6379/0" 

def test_obs_53():
    try:
        from hbmon.annotation_jobs import preprocess_observation_job
        logger.info("Starting test for Obs 53...")
        
        # Run the job synchronously
        result = preprocess_observation_job(53, resume=False)
        
        logger.info(f"Job Result: {result}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obs_53()
