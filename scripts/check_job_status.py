import sys
import os
import logging
from hbmon.annotation_jobs import get_job_status

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("check_status")

def check_status():
    job_id = "f88bbab3-c983-495d-895d-609584fd7e1c"
    status = get_job_status(job_id)
    if status:
        logger.info(f"Job {job_id} status: {status}")
    else:
        logger.error(f"Job {job_id} not found")

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    check_status()
