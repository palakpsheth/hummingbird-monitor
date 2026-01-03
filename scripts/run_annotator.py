#!/usr/bin/env python3
"""
Entry point for annotation worker container.
- Configures logging based on HBMON_ANNOTATOR_DEBUG
- Downloads models
- Starts RQ worker via Python API (preserves logging config)
"""
import sys
import os
import logging
from redis import Redis
from rq import Worker, Connection

# Ensure src is in path
sys.path.insert(0, os.getcwd())

# Import download function
try:
    from scripts.download_annotation_models import download_models
except ImportError:
    # Fallback import if running from scripts dir
    from download_annotation_models import download_models

def main():
    debug = os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1"
    
    # Configure root logger
    # This configuration will be shared by the application code running in this process
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True # Force reconfiguration
    )
    
    logger = logging.getLogger("run_annotator")
    logger.info(f"Starting annotator worker via RQ Worker API (DEBUG={debug})")
    
    # Download models
    try:
        logger.info("Downloading/checking models...")
        download_models()
    except Exception as e:
        logger.error(f"Error downloading models: {e}")

    # Setup Redis connection
    redis_url = os.environ.get("HBMON_REDIS_URL", "redis://hbmon-redis:6379/0")
    try:
        conn = Redis.from_url(redis_url)
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Start Worker
    queues = ["annotation"]
    
    with Connection(conn):
        worker = Worker(queues)
        logger.info(f"Worker {worker.key} started, listening on {queues}")
        
        # Start work loop
        # We pass logging_level to let RQ configure its internal logger mostly,
        # but our root logger set via basicConfig should handle app logs.
        # Note: RQ might fork for jobs (default). 
        # Forked processes inherit file descriptors (logs), so it should work.
        try:
             worker.work(logging_level="DEBUG" if debug else "INFO")
        except Exception as e:
             logger.error(f"Worker crashed: {e}")
             sys.exit(1)

if __name__ == "__main__":
    main()
