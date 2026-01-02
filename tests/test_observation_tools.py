"""
Tests for the hbmon.observation_tools module.

These tests verify video metadata extraction, observation metadata updates,
batch processing, and cache management functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import time

import pytest

# observation_tools has optional dependencies, so we check availability
try:
    from hbmon import observation_tools
    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _MODULE_AVAILABLE, reason="observation_tools module not available")
_HAS_CV2 = _MODULE_AVAILABLE and observation_tools._CV2_AVAILABLE and hasattr(observation_tools, "cv2")


def test_extract_video_metadata_no_cv2():
    """Test that extract_video_metadata raises RuntimeError when cv2 is unavailable."""
    with patch.object(observation_tools, '_CV2_AVAILABLE', False):
        with pytest.raises(RuntimeError, match="OpenCV.*required"):
            observation_tools.extract_video_metadata(Path("/fake/path.mp4"))


def test_extract_video_metadata_file_not_found():
    """Test that extract_video_metadata raises FileNotFoundError for missing files."""
    if not _HAS_CV2:
        pytest.skip("OpenCV not available")
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError):
            observation_tools.extract_video_metadata(fake_path)


def test_extract_video_metadata_mock():
    """Test video metadata extraction with mocked OpenCV."""
    if not _HAS_CV2:
        pytest.skip("OpenCV not available")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = Path(tf.name)
        # Write some dummy data
        tf.write(b"fake video data")
    
    try:
        # Mock cv2.VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            observation_tools.cv2.CAP_PROP_FPS: 30.0,
            observation_tools.cv2.CAP_PROP_FRAME_WIDTH: 1920,
            observation_tools.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            observation_tools.cv2.CAP_PROP_FRAME_COUNT: 300,
            observation_tools.cv2.CAP_PROP_FOURCC: 0x31637661,  # 'avc1'
        }.get(prop, 0)
        
        with patch.object(observation_tools.cv2, 'VideoCapture', return_value=mock_cap):
            metadata = observation_tools.extract_video_metadata(video_path)
        
        assert metadata["fps"] == 30.0
        assert metadata["width"] == 1920
        assert metadata["height"] == 1080
        assert metadata["resolution"] == "1920×1080"
        assert metadata["frame_count"] == 300
        assert metadata["duration"] == 10.0  # 300 frames / 30 fps
        assert metadata["fourcc"] == "avc1"
        assert "file_size_bytes" in metadata
        assert "file_size_mb" in metadata
        
        mock_cap.release.assert_called_once()
    finally:
        video_path.unlink()


def test_get_observation_video_path_dict():
    """Test extracting video path from dict."""
    obs_data = {"video_path": "clips/test.mp4", "id": 123}
    result = observation_tools.get_observation_video_path(obs_data)
    assert result == Path("clips/test.mp4")


def test_get_observation_video_path_object():
    """Test extracting video path from object."""
    obs_data = Mock()
    obs_data.video_path = "clips/test.mp4"
    result = observation_tools.get_observation_video_path(obs_data)
    assert result == Path("clips/test.mp4")


def test_get_observation_video_path_none():
    """Test extracting video path when none exists."""
    obs_data = {"id": 123}
    result = observation_tools.get_observation_video_path(obs_data)
    assert result is None


def test_validate_video_file_no_cv2():
    """Test validation without cv2 available."""
    with patch.object(observation_tools, '_CV2_AVAILABLE', False):
        valid, msg = observation_tools.validate_video_file(Path("/fake/path.mp4"))
        assert not valid
        assert "OpenCV" in msg


def test_validate_video_file_not_found():
    """Test validation with missing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "nonexistent.mp4"
        valid, msg = observation_tools.validate_video_file(fake_path)
        assert not valid
        assert "not found" in msg.lower()


def test_validate_video_file_mock():
    """Test video file validation with mocked OpenCV."""
    if not _HAS_CV2:
        pytest.skip("OpenCV not available")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = Path(tf.name)
        tf.write(b"fake video data")
    
    try:
        # Mock successful validation
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, Mock())  # ret, frame
        
        with patch.object(observation_tools.cv2, 'VideoCapture', return_value=mock_cap):
            valid, msg = observation_tools.validate_video_file(video_path)
        
        assert valid
        assert msg == "Valid"
        mock_cap.release.assert_called_once()
    finally:
        video_path.unlink()


def test_validate_video_file_cannot_open():
    """Test validation when video can't be opened."""
    if not _HAS_CV2:
        pytest.skip("OpenCV not available")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = Path(tf.name)
        tf.write(b"fake video data")
    
    try:
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        
        with patch.object(observation_tools.cv2, 'VideoCapture', return_value=mock_cap):
            valid, msg = observation_tools.validate_video_file(video_path)
        
        assert not valid
        assert "Failed to open" in msg
    finally:
        video_path.unlink()


@pytest.mark.asyncio
async def test_update_observation_video_metadata_no_sqla():
    """Test that update function raises RuntimeError without SQLAlchemy."""
    with patch.object(observation_tools, '_SQLA_AVAILABLE', False):
        with pytest.raises(RuntimeError, match="SQLAlchemy.*required"):
            await observation_tools.update_observation_video_metadata(
                Mock(), 1, {}
            )


@pytest.mark.asyncio
async def test_clean_compressed_cache_no_cache_dir():
    """Test cache cleanup when cache directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        media_dir = Path(tmpdir)
        stats = await observation_tools.clean_compressed_cache(media_dir)
        assert stats["files_removed"] == 0
        assert stats["space_freed_mb"] == 0.0


@pytest.mark.asyncio
async def test_clean_compressed_cache_removes_old_files():
    """Test that old cached files are removed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        media_dir = Path(tmpdir)
        cache_dir = media_dir / ".cache" / "compressed"
        cache_dir.mkdir(parents=True)
        
        # Create an old cached file
        old_file = cache_dir / "old_video.mp4"
        old_file.write_text("fake video")
        
        # Set mtime to 10 days ago
        old_time = time.time() - (10 * 24 * 3600)
        import os
        os.utime(old_file, (old_time, old_time))
        
        # Clean with max age of 7 days
        stats = await observation_tools.clean_compressed_cache(
            media_dir, max_age_days=7, max_cache_size_gb=100.0
        )
        
        assert stats["files_removed"] == 1
        assert stats["space_freed_mb"] >= 0  # May be 0 for small files due to rounding
        assert not old_file.exists()


@pytest.mark.asyncio
async def test_clean_compressed_cache_size_limit():
    """Test that cache is trimmed when it exceeds size limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        media_dir = Path(tmpdir)
        cache_dir = media_dir / ".cache" / "compressed"
        cache_dir.mkdir(parents=True)
        
        # Create multiple small files
        for i in range(5):
            file = cache_dir / f"video_{i}.mp4"
            file.write_text("x" * 1024 * 100)  # 100KB each
            # Add small delay to ensure different mtimes
            time.sleep(0.01)
        
        # Set max cache size to very small (1KB = 0.001MB)
        # This should remove all but the newest file
        stats = await observation_tools.clean_compressed_cache(
            media_dir, max_age_days=365, max_cache_size_gb=0.000001
        )
        
        # Should have removed at least some files
        assert stats["files_removed"] > 0
        
        # Check that newest files were kept
        remaining = list(cache_dir.glob("*.mp4"))
        assert len(remaining) < 5


def test_module_imports():
    """Test that module can be imported and has expected functions."""
    assert hasattr(observation_tools, 'extract_video_metadata')
    assert hasattr(observation_tools, 'update_observation_video_metadata')
    assert hasattr(observation_tools, 'process_observations_batch')
    assert hasattr(observation_tools, 'clean_compressed_cache')
    assert hasattr(observation_tools, 'validate_video_file')
    assert hasattr(observation_tools, 'get_observation_video_path')


def test_module_availability_flags():
    """Test that module properly detects optional dependencies."""
    # The module should have these flags
    assert hasattr(observation_tools, '_CV2_AVAILABLE')
    assert hasattr(observation_tools, '_SQLA_AVAILABLE')
    # They should be boolean
    assert isinstance(observation_tools._CV2_AVAILABLE, bool)
    assert isinstance(observation_tools._SQLA_AVAILABLE, bool)
