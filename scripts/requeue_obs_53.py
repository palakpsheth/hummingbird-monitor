import sys
import os
import logging
import redis
from rq import Queue
from sqlalchemy import delete
from hbmon.db import get_sync_session
from hbmon.models import Observation, AnnotationFrame
from hbmon.annotation_jobs import preprocess_observation_job

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("requeue_obs_53")

def requeue_obs_53():
    obs_id = 53
    logger.info(f"Resetting annotation for Observation {obs_id}...")

    # 1. Clear existing annotations from DB
    with get_sync_session() as db:
        obs = db.get(Observation, obs_id)
        if not obs:
            logger.error(f"Observation {obs_id} not found!")
            return

        # Delete frames (cascades to boxes)
        # Note: using ORM delete for safety
        stmt = delete(AnnotationFrame).where(AnnotationFrame.observation_id == obs_id)
        result = db.execute(stmt)
        db.commit()
        logger.info(f"Deleted {result.rowcount} existing AnnotationFrame records.")

    # 2. Enqueue job
    redis_url = os.environ.get("HBMON_REDIS_URL", "redis://hbmon-redis:6379/0")
    logger.info(f"Connecting to Redis at {redis_url}...")
    
    try:
        conn = redis.from_url(redis_url)
        q = Queue("annotation", connection=conn)
        
        # Enqueue the job
        # Note: depends on exact signature of preprocess_observation_job. 
        # Typically it takes nothing or obs_id? 
        # Reading previous file shows: def preprocess_observation_job(observation_id: int):
        
        job = q.enqueue(preprocess_observation_job, obs_id, job_timeout='60m')
        logger.info(f"Enqueued job {job.id} for Observation {obs_id} position={job.get_position()}")
        
    except Exception as e:
        logger.error(f"Failed to enqueue: {e}")

if __name__ == "__main__":
    # Ensure src is in path
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    requeue_obs_53()
