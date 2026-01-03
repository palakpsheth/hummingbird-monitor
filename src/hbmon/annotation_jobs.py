# src/hbmon/annotation_jobs.py
"""
Background job processing for annotation preprocessing.

This module uses RQ (Redis Queue) for reliable background job processing.
Jobs include:
- Frame extraction from observation videos
- Auto-detection with YOLO Large + SAM
- Label propagation from observation to frames
- Nightly batch processing

Usage:
    # Enqueue a job from web routes
    from hbmon.annotation_jobs import enqueue_preprocessing
    job_id = enqueue_preprocessing(obs_id=123)
    
    # Check job status
    from hbmon.annotation_jobs import get_job_status
    status = get_job_status(job_id)

Worker command:
    rq worker annotation --url redis://localhost:6379/0
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Redis connection
REDIS_URL = os.environ.get("HBMON_REDIS_URL", "redis://localhost:6379/0")
QUEUE_ENABLED = os.environ.get("HBMON_ANNOTATION_QUEUE_ENABLED", "1") == "1"
AUTO_EXTRACT = os.environ.get("HBMON_ANNOTATION_AUTO_EXTRACT", "0") == "1"
BATCH_HOUR = int(os.environ.get("HBMON_ANNOTATION_BATCH_HOUR", "3"))

# Job timeouts (user configurable via env vars)
EXTRACTION_TIMEOUT = os.environ.get("HBMON_ANNOTATION_EXTRACTION_TIMEOUT", "30m")
DETECTION_TIMEOUT = os.environ.get("HBMON_ANNOTATION_DETECTION_TIMEOUT", "60m")
PROPAGATION_TIMEOUT = os.environ.get("HBMON_ANNOTATION_PROPAGATION_TIMEOUT", "5m")

# Job retry configuration
MAX_RETRIES = int(os.environ.get("HBMON_ANNOTATION_MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("HBMON_ANNOTATION_RETRY_DELAY", "60"))  # seconds

# Checkpointing: save progress every N frames for recovery
CHECKPOINT_INTERVAL = int(os.environ.get("HBMON_ANNOTATION_CHECKPOINT_INTERVAL", "10"))

# Lazy imports for optional dependencies
_queue = None
_redis = None


def _get_redis():
    """Get Redis connection (lazy init)."""
    global _redis
    if _redis is None:
        try:
            from redis import Redis
            _redis = Redis.from_url(REDIS_URL)
            # Test connection
            _redis.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            _redis = False
    return _redis if _redis else None


def _get_queue():
    """Get RQ queue (lazy init)."""
    global _queue
    if _queue is None:
        redis = _get_redis()
        if redis:
            try:
                from rq import Queue
                _queue = Queue("annotation", connection=redis)
            except ImportError:
                logger.warning("RQ package not installed, job queuing disabled")
                _queue = False
    return _queue if _queue else None


def is_queue_available() -> bool:
    """Check if job queue is available and enabled."""
    if not QUEUE_ENABLED:
        return False
    return _get_queue() is not None


def enqueue_preprocessing(obs_id: int, resume: bool = False) -> str | None:
    """Enqueue observation for full preprocessing pipeline.
    
    This is the main entry point for background processing.
    Includes automatic retry on failure with exponential backoff.
    
    Args:
        obs_id: Observation ID to preprocess
        resume: If True, resume from last checkpoint instead of starting fresh
        
    Returns:
        Job ID if enqueued, None if queue unavailable
    """
    queue = _get_queue()
    if not queue:
        logger.warning("Queue not available, running preprocessing inline")
        return None
    
    try:
        from rq import Retry
        
        # Configure retries with exponential backoff
        retry = Retry(max=MAX_RETRIES, interval=[RETRY_DELAY, RETRY_DELAY * 2, RETRY_DELAY * 4])
        
        job = queue.enqueue(
            preprocess_observation_job,
            obs_id,
            resume,
            job_timeout=EXTRACTION_TIMEOUT,
            result_ttl=86400,  # Keep result for 24 hours
            failure_ttl=86400,
            retry=retry,
            description=f"Preprocess observation {obs_id}",
            meta={"obs_id": obs_id, "resume": resume},
        )
        logger.info(f"Enqueued preprocessing job for obs {obs_id}: {job.id}")
        return job.id
    except ImportError:
        # Fallback without retry if Retry not available
        job = queue.enqueue(
            preprocess_observation_job,
            obs_id,
            resume,
            job_timeout=EXTRACTION_TIMEOUT,
            result_ttl=86400,
            failure_ttl=86400,
            description=f"Preprocess observation {obs_id}",
        )
        return job.id
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        return None


def enqueue_detection(obs_id: int) -> str | None:
    """Enqueue observation for auto-detection only (frames already extracted).
    
    Args:
        obs_id: Observation ID
        
    Returns:
        Job ID if enqueued, None if queue unavailable
    """
    queue = _get_queue()
    if not queue:
        return None
    
    try:
        job = queue.enqueue(
            run_detection_job,
            obs_id,
            job_timeout=DETECTION_TIMEOUT,
            result_ttl=86400,
            failure_ttl=86400,
            description=f"Detection for observation {obs_id}",
        )
        return job.id
    except Exception as e:
        logger.error(f"Failed to enqueue detection job: {e}")
        return None


def enqueue_label_propagation(obs_id: int, label: str) -> str | None:
    """Enqueue label propagation job.
    
    Args:
        obs_id: Observation ID
        label: Label to propagate (e.g., "false_positive", "verified")
        
    Returns:
        Job ID if enqueued
    """
    queue = _get_queue()
    if not queue:
        return None
    
    try:
        job = queue.enqueue(
            propagate_label_job,
            obs_id,
            label,
            job_timeout=PROPAGATION_TIMEOUT,
            result_ttl=86400,
            description=f"Propagate label '{label}' for observation {obs_id}",
        )
        return job.id
    except Exception as e:
        logger.error(f"Failed to enqueue propagation job: {e}")
        return None


def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Get status of a job.
    
    Args:
        job_id: RQ job ID
        
    Returns:
        Job status dict or None if not found
    """
    redis = _get_redis()
    if not redis:
        return None
    
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=redis)
        return {
            "id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "result": job.result if job.get_status() == "finished" else None,
            "error": str(job.exc_info) if job.exc_info else None,
        }
    except Exception as e:
        logger.debug(f"Failed to fetch job {job_id}: {e}")
        return None


def get_pending_jobs() -> list[dict[str, Any]]:
    """Get list of pending annotation jobs."""
    queue = _get_queue()
    if not queue:
        return []
    
    try:
        jobs = []
        for job in queue.jobs:
            jobs.append({
                "id": job.id,
                "description": job.description,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            })
        return jobs
    except Exception:
        return []


# ============================================================
# Job Functions (run by RQ worker)
# ============================================================

def preprocess_observation_job(obs_id: int, resume: bool = False) -> dict[str, Any]:
    """Full preprocessing pipeline for an observation.
    
    This job:
    1. Extracts frames from video (or resumes from checkpoint)
    2. Runs auto-detection on each frame
    3. Saves boxes to DB and disk
    4. Updates observation state
    
    Supports checkpointing for recovery after failures.
    
    Args:
        obs_id: Observation ID
        resume: If True, resume from last checkpoint
        
    Returns:
        Result dict with statistics
    """
    import json as json_module
    
    logger.info(f"Starting preprocessing for observation {obs_id} (resume={resume})")
    
    start_time = datetime.now(timezone.utc)
    result = {
        "obs_id": obs_id,
        "frames_extracted": 0,
        "frames_detected": 0,
        "total_boxes": 0,
        "status": "running",
        "resumed": resume,
    }
    
    # Checkpoint file for recovery
    checkpoint_dir = Path(os.environ.get("HBMON_DATA_DIR", "/data")) / "exports" / "annotations" / "checkpoints"
    checkpoint_file = checkpoint_dir / f"{obs_id}_checkpoint.json"
    
    def save_checkpoint(data: dict) -> None:
        """Save checkpoint for recovery."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text(json_module.dumps(data))
        logger.debug(f"Saved checkpoint for obs {obs_id}: {data}")
    
    def load_checkpoint() -> dict | None:
        """Load checkpoint if exists."""
        if checkpoint_file.exists():
            try:
                return json_module.loads(checkpoint_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    def clear_checkpoint() -> None:
        """Remove checkpoint file on successful completion."""
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    # Load checkpoint if resuming
    checkpoint = load_checkpoint() if resume else None
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        result["frames_extracted"] = checkpoint.get("frames_extracted", 0)
        result["frames_detected"] = checkpoint.get("frames_detected", 0)
        result["total_boxes"] = checkpoint.get("total_boxes", 0)
    
    try:
        # Import here to avoid circular imports and ensure models load in worker
        from hbmon.annotation_storage import (
            ensure_annotation_dirs,
            save_yolo_label,
            save_box_json,
            BoxData,
            sync_db_to_manifest,
            AnnotationSummary,
        )
        from hbmon.annotation_detector import AnnotationDetector
        
        # Database imports
        from hbmon.db import get_sync_session
        from hbmon.models import Observation, AnnotationFrame, AnnotationBox
        
        ensure_annotation_dirs()
        
        # Get observation
        with get_sync_session() as db:
            obs = db.get(Observation, obs_id)
            if not obs:
                result["status"] = "error"
                result["error"] = "Observation not found"
                return result
            
            video_path = Path(os.environ.get("HBMON_MEDIA_DIR", "/data/media")) / obs.video_path
            
            if not video_path.exists():
                result["status"] = "error"
                result["error"] = f"Video not found: {video_path}"
                return result
            
            # Update state to preprocessing
            obs.merge_extra({
                "annotation_summary": {
                    "state": "preprocessing",
                    "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
                }
            })
            db.commit()
        
        # Step 1: Extract frames
        logger.info(f"Extracting frames from {video_path}")
        frames_data = _extract_frames_sync(obs_id, video_path)
        result["frames_extracted"] = len(frames_data)
        
        if not frames_data:
            result["status"] = "error"
            result["error"] = "No frames extracted"
            return result
        
        # Step 2: Create AnnotationFrame records
        with get_sync_session() as db:
            for idx, frame_path in frames_data:
                db_frame = AnnotationFrame(
                    observation_id=obs_id,
                    frame_index=idx,
                    frame_path=str(frame_path),
                    bird_present=False,
                    status="queued",
                )
                db.add(db_frame)
            db.commit()
            
            # Update summary with frame counts so UI shows progress immediately
            with get_sync_session() as db:
                obs = db.get(Observation, obs_id)
                if obs:
                    total_count = len(frames_data)
                    # Keep state as preprocessing while detection runs
                    current_summary = (obs.get_extra() or {}).get("annotation_summary", {})
                    obs.merge_extra({
                        "annotation_summary": {
                            **current_summary,
                            "total_frames": total_count,
                            "pending_frames": total_count,
                            "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
                        }
                    })
                    db.commit()
        
        # Step 3: Run auto-detection
        logger.info(f"Running detection on {len(frames_data)} frames")
        detector = AnnotationDetector()
        detector.initialize()
        
        batch_size = int(os.environ.get("HBMON_ANNOTATION_BATCH_SIZE", "8"))
        
        # Get last processed frame from checkpoint for resume
        last_processed = checkpoint.get("last_frame_idx", -1) if checkpoint else -1
        
        with get_sync_session() as db:
            for i in range(0, len(frames_data), batch_size):
                batch = frames_data[i:i + batch_size]
                
                # Skip already-processed batches when resuming
                batch_max_idx = max(idx for idx, _ in batch)
                if batch_max_idx <= last_processed:
                    result["frames_detected"] += len(batch)
                    continue
                
                # Load frames
                import cv2
                images = []
                for idx, frame_path in batch:
                    # Skip individual frames already processed
                    if idx <= last_processed:
                        continue
                    img = cv2.imread(str(frame_path))
                    if img is not None:
                        images.append((idx, frame_path, img))
                
                if not images:
                    continue
                
                # Detect in batch
                batch_results = detector.detect_batch([img for _, _, img in images])
                
                for j, (idx, frame_path, _) in enumerate(images):
                    boxes = batch_results[j]
                    
                    # Get frame record
                    frame_record = db.query(AnnotationFrame).filter(
                        AnnotationFrame.observation_id == obs_id,
                        AnnotationFrame.frame_index == idx,
                    ).first()
                    
                    if not frame_record:
                        continue
                    
                    # Create box records
                    bird_present = len(boxes) > 0
                    frame_record.bird_present = bird_present
                    frame_record.status = "auto"
                    
                    box_data_list = []
                    for box in boxes:
                        db_box = AnnotationBox(
                            frame_id=frame_record.id,
                            class_id=box.class_id,
                            x=box.x,
                            y=box.y,
                            w=box.w,
                            h=box.h,
                            is_false_positive=False,
                            source=box.source,
                        )
                        db.add(db_box)
                        
                        box_data_list.append(BoxData(
                            class_id=box.class_id,
                            x=box.x,
                            y=box.y,
                            w=box.w,
                            h=box.h,
                            is_false_positive=False,
                            source=box.source,
                        ))
                    
                    result["total_boxes"] += len(boxes)
                    
                    # Save to disk
                    save_yolo_label(str(obs_id), idx, box_data_list)
                    save_box_json(str(obs_id), idx, box_data_list, bird_present)
                
                db.commit()
                result["frames_detected"] += len(images)
                
                # Log progress
                pct = (result["frames_detected"] / len(frames_data)) * 100
                logger.info(f"Obs {obs_id}: Processed {result['frames_detected']}/{len(frames_data)} frames ({pct:.1f}%)")

                # Update DB progress every 5 batches (approx 40 frames)
                if (i // batch_size) % 5 == 0:
                    obs_update = db.get(Observation, obs_id)
                    if obs_update:
                        current_summary = (obs_update.get_extra() or {}).get("annotation_summary", {})
                        obs_update.merge_extra({
                            "annotation_summary": {
                                **current_summary,
                                "frames_detected": result["frames_detected"],
                                "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
                            }
                        })
                        db.commit()
                
                # Checkpoint every CHECKPOINT_INTERVAL batches
                if (i // batch_size) % CHECKPOINT_INTERVAL == 0:
                    current_max_idx = max(idx for idx, _, _ in images)
                    save_checkpoint({
                        "frames_extracted": result["frames_extracted"],
                        "frames_detected": result["frames_detected"],
                        "total_boxes": result["total_boxes"],
                        "last_frame_idx": current_max_idx,
                    })
        
        # Step 4: Update observation state
        with get_sync_session() as db:
            obs = db.get(Observation, obs_id)
            if obs:
                total_frames = len(frames_data)
                obs.merge_extra({
                    "annotation_summary": {
                        "total_frames": total_frames,
                        "reviewed_frames": 0,
                        "pending_frames": total_frames,
                        "state": "in_review",
                        "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
                    }
                })
                db.commit()
                
                # Sync manifest
                sync_db_to_manifest(str(obs_id), AnnotationSummary(
                    total_frames=total_frames,
                    reviewed_frames=0,
                    pending_frames=total_frames,
                    state="in_review",
                ))
        
        result["status"] = "completed"
        result["duration_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Clear checkpoint on successful completion
        clear_checkpoint()
        
        logger.info(f"Preprocessing complete for obs {obs_id}: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"Preprocessing failed for obs {obs_id}")
        result["status"] = "error"
        result["error"] = str(e)
        return result


def run_detection_job(obs_id: int) -> dict[str, Any]:
    """Run detection on already-extracted frames.
    
    This is useful for re-running detection with different models.
    """
    logger.info(f"Running detection for observation {obs_id}")
    
    # Similar to preprocessing but skip frame extraction
    # Implementation would query existing frames and run detection
    
    return {"obs_id": obs_id, "status": "not_implemented"}


def propagate_label_job(obs_id: int, label: str) -> dict[str, Any]:
    """Propagate observation-level label to all frames.
    
    Args:
        obs_id: Observation ID
        label: Label to apply ("false_positive", "verified", etc.)
        
    Returns:
        Result dict
    """
    logger.info(f"Propagating label '{label}' for observation {obs_id}")
    
    try:
        from hbmon.db import get_sync_session
        from hbmon.models import AnnotationFrame
        from hbmon.annotation_storage import save_yolo_label, save_box_json, BoxData
        
        with get_sync_session() as db:
            frames = db.query(AnnotationFrame).filter(
                AnnotationFrame.observation_id == obs_id
            ).all()
            
            frames_updated = 0
            
            for frame in frames:
                if label == "false_positive":
                    # Mark as no bird, clear boxes or mark all as FP
                    frame.bird_present = False
                    frame.status = "complete"
                    
                    # Mark all boxes as false positive
                    for box in frame.boxes:
                        box.is_false_positive = True
                    
                    # Regenerate disk files
                    box_data = [
                        BoxData(
                            class_id=b.class_id,
                            x=b.x,
                            y=b.y,
                            w=b.w,
                            h=b.h,
                            is_false_positive=True,
                            source=b.source,
                        )
                        for b in frame.boxes
                    ]
                    save_yolo_label(str(obs_id), frame.frame_index, box_data)
                    save_box_json(str(obs_id), frame.frame_index, box_data, False)
                    
                elif label == "verified":
                    # Keep existing annotations, mark as complete
                    frame.status = "complete"
                    frame.reviewed_at = datetime.now(timezone.utc)
                
                frames_updated += 1
            
            db.commit()
        
        return {
            "obs_id": obs_id,
            "label": label,
            "frames_updated": frames_updated,
            "status": "completed",
        }
        
    except Exception as e:
        logger.exception(f"Label propagation failed for obs {obs_id}")
        return {"obs_id": obs_id, "status": "error", "error": str(e)}


def nightly_batch_extract_job() -> dict[str, Any]:
    """Nightly batch extraction of frames for new observations.
    
    Finds observations without annotation frames and enqueues them.
    """
    logger.info("Running nightly batch extraction")
    
    try:
        from hbmon.db import get_sync_session
        from hbmon.models import Observation, AnnotationFrame
        
        with get_sync_session() as db:
            # Find observations without annotation frames
            obs_with_frames = db.query(AnnotationFrame.observation_id).distinct()
            
            unprocessed = db.query(Observation).filter(
                ~Observation.id.in_(obs_with_frames)
            ).all()
            
            enqueued = 0
            for obs in unprocessed:
                job_id = enqueue_preprocessing(obs.id)
                if job_id:
                    enqueued += 1
        
        return {
            "found": len(unprocessed),
            "enqueued": enqueued,
            "status": "completed",
        }
        
    except Exception as e:
        logger.exception("Nightly batch extraction failed")
        return {"status": "error", "error": str(e)}


def _extract_frames_sync(obs_id: int, video_path: Path) -> list[tuple[int, Path]]:
    """Synchronously extract frames from video.
    
    Args:
        obs_id: Observation ID
        video_path: Path to video file
        
    Returns:
        List of (frame_index, frame_path) tuples
    """
    import cv2
    from hbmon.annotation_storage import get_frame_path
    
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = get_frame_path(str(obs_id), idx)
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame)
            frames.append((idx, frame_path))
            idx += 1
    finally:
        cap.release()
    
    return frames
