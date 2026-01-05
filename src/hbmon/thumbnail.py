"""Thumbnail generation and caching utilities.

This module provides efficient thumbnail generation for images displayed
in the web UI. Thumbnails are cached on disk to avoid repeated processing.

Cache structure:
    /media/.cache/thumbnails/{width}/{relative_path}

Example:
    Original: /media/observations/2026-01-04/obs_123_snapshot.jpg
    Thumbnail: /media/.cache/thumbnails/300/observations/2026-01-04/obs_123_snapshot.jpg
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default thumbnail width in pixels
DEFAULT_THUMB_WIDTH = 300

# Maximum allowed thumbnail width
MAX_THUMB_WIDTH = 600

# JPEG quality for thumbnails (0-100, higher = better quality but larger file)
THUMB_QUALITY = 85


def get_media_dir() -> Path:
    """Get the media directory from environment or default."""
    return Path(os.environ.get("HBMON_MEDIA_DIR", "/media"))


def get_thumbnail_cache_dir() -> Path:
    """Get the thumbnail cache directory."""
    return get_media_dir() / ".cache" / "thumbnails"


def get_thumbnail_path(relative_path: str, width: int = DEFAULT_THUMB_WIDTH) -> Path:
    """Get the path where a thumbnail should be cached.
    
    Args:
        relative_path: Path relative to media directory (e.g., "observations/2026-01/snap.jpg")
        width: Thumbnail width in pixels
        
    Returns:
        Absolute path to the thumbnail file
    """
    # Normalize width to valid range
    width = min(max(width, 50), MAX_THUMB_WIDTH)
    
    cache_dir = get_thumbnail_cache_dir() / str(width)
    return cache_dir / relative_path


def get_source_path(relative_path: str) -> Path:
    """Get the full path to the source image.
    
    Args:
        relative_path: Path relative to media directory
        
    Returns:
        Absolute path to the source file
    """
    return get_media_dir() / relative_path


def is_thumbnail_stale(thumb_path: Path, source_path: Path) -> bool:
    """Check if a thumbnail needs to be regenerated.
    
    A thumbnail is stale if:
    - It doesn't exist
    - The source file is newer than the thumbnail
    
    Args:
        thumb_path: Path to the thumbnail
        source_path: Path to the source image
        
    Returns:
        True if thumbnail needs to be regenerated
    """
    if not thumb_path.exists():
        return True
    
    if not source_path.exists():
        return True
    
    # Compare modification times
    thumb_mtime = thumb_path.stat().st_mtime
    source_mtime = source_path.stat().st_mtime
    
    return source_mtime > thumb_mtime


def generate_thumbnail(
    source_path: Path,
    thumb_path: Path,
    width: int = DEFAULT_THUMB_WIDTH,
) -> bool:
    """Generate a thumbnail from a source image.
    
    Uses Pillow if available, falls back to OpenCV.
    
    Args:
        source_path: Path to the source image
        thumb_path: Path where thumbnail should be saved
        width: Target width in pixels (height calculated to maintain aspect ratio)
        
    Returns:
        True if thumbnail was generated successfully
    """
    if not source_path.exists():
        logger.warning(f"Source image not found: {source_path}")
        return False
    
    try:
        # Try Pillow first (preferred for quality)
        try:
            from PIL import Image
            
            with Image.open(source_path) as img:
                # Calculate height maintaining aspect ratio
                aspect = img.height / img.width
                new_height = int(width * aspect)
                
                # Use high-quality resampling
                resized = img.resize((width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary (for RGBA/palette images)
                if resized.mode in ("RGBA", "P"):
                    resized = resized.convert("RGB")
                
                # Ensure parent directory exists
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save as JPEG with quality setting
                resized.save(thumb_path, "JPEG", quality=THUMB_QUALITY, optimize=True)
                
            logger.debug(f"Generated thumbnail: {thumb_path} ({width}px)")
            return True
            
        except ImportError:
            pass
        
        # Fallback to OpenCV
        try:
            import cv2
            
            img = cv2.imread(str(source_path))
            if img is None:
                logger.warning(f"OpenCV could not read: {source_path}")
                return False
            
            # Calculate dimensions
            h, w = img.shape[:2]
            aspect = h / w
            new_height = int(width * aspect)
            
            # Resize with inter-area for downscaling
            resized = cv2.resize(img, (width, new_height), interpolation=cv2.INTER_AREA)
            
            # Ensure parent directory exists
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JPEG
            cv2.imwrite(str(thumb_path), resized, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
            
            logger.debug(f"Generated thumbnail (cv2): {thumb_path} ({width}px)")
            return True
            
        except ImportError:
            logger.error("Neither Pillow nor OpenCV available for thumbnail generation")
            return False
            
    except Exception as e:
        logger.error(f"Failed to generate thumbnail for {source_path}: {e}")
        return False


def get_or_create_thumbnail(
    relative_path: str,
    width: int = DEFAULT_THUMB_WIDTH,
) -> Path | None:
    """Get a thumbnail, generating it if necessary.
    
    This is the main entry point for thumbnail generation.
    
    Args:
        relative_path: Path relative to media directory
        width: Target width in pixels
        
    Returns:
        Path to the thumbnail, or None if generation failed
    """
    # Normalize width
    width = min(max(width, 50), MAX_THUMB_WIDTH)
    
    source_path = get_source_path(relative_path)
    thumb_path = get_thumbnail_path(relative_path, width)
    
    # Check if we need to generate
    if is_thumbnail_stale(thumb_path, source_path):
        if not generate_thumbnail(source_path, thumb_path, width):
            return None
    
    return thumb_path


def clean_old_thumbnails(max_age_days: int = 30) -> int:
    """Remove thumbnails older than the specified age.
    
    Args:
        max_age_days: Maximum age in days before thumbnails are deleted
        
    Returns:
        Number of thumbnails deleted
    """
    cache_dir = get_thumbnail_cache_dir()
    if not cache_dir.exists():
        return 0
    
    max_age_seconds = max_age_days * 24 * 60 * 60
    now = time.time()
    deleted = 0
    
    try:
        for thumb_file in cache_dir.rglob("*"):
            if not thumb_file.is_file():
                continue
            
            try:
                file_age = now - thumb_file.stat().st_mtime
                if file_age > max_age_seconds:
                    thumb_file.unlink()
                    deleted += 1
            except OSError:
                continue
        
        # Clean up empty directories
        for dir_path in sorted(cache_dir.rglob("*"), reverse=True):
            if dir_path.is_dir():
                try:
                    dir_path.rmdir()  # Only removes if empty
                except OSError:
                    continue
                    
    except Exception as e:
        logger.error(f"Error cleaning thumbnails: {e}")
    
    if deleted > 0:
        logger.info(f"Cleaned {deleted} old thumbnails")
    
    return deleted


def get_thumbnail_stats() -> dict:
    """Get statistics about the thumbnail cache.
    
    Returns:
        Dictionary with cache statistics
    """
    cache_dir = get_thumbnail_cache_dir()
    if not cache_dir.exists():
        return {"exists": False, "count": 0, "size_bytes": 0}
    
    count = 0
    total_size = 0
    
    try:
        for thumb_file in cache_dir.rglob("*"):
            if thumb_file.is_file():
                count += 1
                total_size += thumb_file.stat().st_size
    except Exception:
        pass
    
    return {
        "exists": True,
        "count": count,
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
    }
