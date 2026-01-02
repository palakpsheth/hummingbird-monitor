"""
Utilities for reading and processing existing observation data.

This module provides support functions for:
- Extracting video metadata from existing clips
- Updating observation metadata in the database
- Batch processing of observations
- Cache management for compressed videos
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select
    _SQLA_AVAILABLE = True
except ImportError:
    _SQLA_AVAILABLE = False

logger = logging.getLogger(__name__)


def extract_video_metadata(video_path: Path) -> dict[str, Any]:
    """
    Extract metadata from a video file using OpenCV.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing:
        - fps: Frames per second
        - width: Video width in pixels
        - height: Video height in pixels  
        - frame_count: Total number of frames
        - duration: Duration in seconds
        - fourcc: Codec fourcc code
        - file_size_bytes: File size in bytes
        - file_size_mb: File size in MB
        
    Raises:
        RuntimeError: If OpenCV is not available
        FileNotFoundError: If video file doesn't exist
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for video metadata extraction")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    metadata: dict[str, Any] = {}
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        try:
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                metadata["fps"] = round(fps, 2)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                metadata["width"] = width
                metadata["height"] = height
                metadata["resolution"] = f"{width}×{height}"
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                metadata["frame_count"] = frame_count
                if fps > 0:
                    metadata["duration"] = round(frame_count / fps, 2)
            
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc != 0:
                codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                metadata["fourcc"] = codec_str
        finally:
            cap.release()
        
        # Get file size
        stat = video_path.stat()
        metadata["file_size_bytes"] = stat.st_size
        metadata["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting video metadata from {video_path}: {e}")
        raise


async def update_observation_video_metadata(
    db: AsyncSession,
    observation_id: int,
    video_metadata: dict[str, Any],
) -> bool:
    """
    Update an observation's extra_json with video metadata.
    
    Args:
        db: Database session
        observation_id: Observation ID to update
        video_metadata: Video metadata dict (from extract_video_metadata)
        
    Returns:
        True if updated successfully, False otherwise
    """
    if not _SQLA_AVAILABLE:
        raise RuntimeError("SQLAlchemy is required for database operations")
    
    from hbmon.models import Observation
    
    try:
        # Fetch observation
        obs = await db.get(Observation, observation_id)
        if obs is None:
            logger.warning(f"Observation {observation_id} not found")
            return False
        
        # Get existing extra data
        extra = obs.get_extra() or {}
        if not isinstance(extra, dict):
            extra = {}
        
        # Add or update video metadata
        if "media" not in extra:
            extra["media"] = {}
        
        extra["media"]["video"] = video_metadata
        
        # Save back to observation
        obs.set_extra(extra)
        await db.commit()
        
        logger.info(f"Updated video metadata for observation {observation_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating observation {observation_id}: {e}")
        await db.rollback()
        return False


async def process_observations_batch(
    db: AsyncSession,
    media_dir: Path,
    batch_size: int = 100,
    update_metadata: bool = True,
) -> dict[str, int]:
    """
    Process all observations and extract/update video metadata.
    
    Args:
        db: Database session
        media_dir: Path to media directory containing videos
        batch_size: Number of observations to process per batch
        update_metadata: Whether to update database with extracted metadata
        
    Returns:
        Statistics dict with counts of processed, updated, failed
    """
    if not _SQLA_AVAILABLE:
        raise RuntimeError("SQLAlchemy is required for database operations")
    
    from hbmon.models import Observation
    
    stats = {
        "total": 0,
        "processed": 0,
        "updated": 0,
        "failed": 0,
        "no_video": 0,
    }
    
    try:
        # Count total observations with videos
        total_count = (
            await db.execute(
                select(Observation)
                .where(Observation.video_path.isnot(None))
                .where(Observation.video_path != "")
            )
        ).scalars().all()
        
        stats["total"] = len(total_count)
        
        for obs in total_count:
            if not obs.video_path:
                stats["no_video"] += 1
                continue
            
            video_file = media_dir / obs.video_path
            
            if not video_file.exists():
                logger.warning(f"Video file not found for observation {obs.id}: {obs.video_path}")
                stats["failed"] += 1
                continue
            
            try:
                # Extract metadata
                metadata = extract_video_metadata(video_file)
                stats["processed"] += 1
                
                # Update database if requested
                if update_metadata:
                    success = await update_observation_video_metadata(db, obs.id, metadata)
                    if success:
                        stats["updated"] += 1
                    else:
                        stats["failed"] += 1
                        
            except Exception as e:
                logger.error(f"Failed to process observation {obs.id}: {e}")
                stats["failed"] += 1
        
        return stats
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise


async def clean_compressed_cache(
    media_dir: Path,
    max_age_days: int = 7,
    max_cache_size_gb: float = 10.0,
) -> dict[str, Any]:
    """
    Clean up old compressed video cache files.
    
    Args:
        media_dir: Path to media directory
        max_age_days: Remove cached files older than this many days
        max_cache_size_gb: If cache exceeds this size, remove oldest files
        
    Returns:
        Statistics dict with files_removed, space_freed_mb
    """
    cache_dir = media_dir / ".cache" / "compressed"
    
    if not cache_dir.exists():
        return {"files_removed": 0, "space_freed_mb": 0.0}
    
    stats = {
        "files_removed": 0,
        "space_freed_mb": 0.0,
    }
    
    try:
        import time
        
        now = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        # Get all cached files with their stats
        cached_files: list[tuple[Path, float, float]] = []  # (path, size, mtime)
        total_size = 0.0
        
        for file_path in cache_dir.glob("*.mp4"):
            try:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                cached_files.append((file_path, size_mb, stat.st_mtime))
                total_size += size_mb
            except OSError:
                continue
        
        # Sort by modification time (oldest first)
        cached_files.sort(key=lambda x: x[2])
        
        # Remove old files
        for file_path, size_mb, mtime in cached_files:
            should_remove = False
            
            # Remove if too old
            if (now - mtime) > max_age_seconds:
                should_remove = True
                logger.info(f"Removing old cached file: {file_path.name} (age: {(now - mtime) / 86400:.1f} days)")
            
            # Remove if cache is too large (remove oldest first)
            elif total_size > (max_cache_size_gb * 1024):
                should_remove = True
                logger.info(f"Removing cached file to reduce cache size: {file_path.name}")
                total_size -= size_mb
            
            if should_remove:
                try:
                    file_path.unlink()
                    stats["files_removed"] += 1
                    stats["space_freed_mb"] += size_mb
                except OSError as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
        
        stats["space_freed_mb"] = round(stats["space_freed_mb"], 2)
        
        return stats
        
    except Exception as e:
        logger.error(f"Cache cleanup error: {e}")
        raise


def get_observation_video_path(obs_data: dict[str, Any] | Any) -> Path | None:
    """
    Extract video path from observation data.
    
    Args:
        obs_data: Observation object or dict with video_path attribute/key
        
    Returns:
        Path object or None if no video path
    """
    video_path = None
    
    if isinstance(obs_data, dict):
        video_path = obs_data.get("video_path")
    elif hasattr(obs_data, "video_path"):
        video_path = obs_data.video_path
    
    if video_path and isinstance(video_path, str):
        return Path(video_path)
    
    return None


def validate_video_file(video_path: Path) -> tuple[bool, str]:
    """
    Validate that a video file exists and can be opened.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not _CV2_AVAILABLE:
        return False, "OpenCV not available"
    
    if not video_path.exists():
        return False, f"File not found: {video_path}"
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Failed to open video file"
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "Failed to read video frames"
        
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)
