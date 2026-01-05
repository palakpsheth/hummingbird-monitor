"""Unit tests for thumbnail generation module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# Import the thumbnail module
from hbmon import thumbnail


class TestThumbnailPaths:
    """Test path generation functions."""

    def test_get_thumbnail_path_default_width(self, tmp_path: Path) -> None:
        """Test thumbnail path with default width."""
        with patch.object(thumbnail, "get_media_dir", return_value=tmp_path):
            result = thumbnail.get_thumbnail_path("observations/snap.jpg")
            assert result == tmp_path / ".cache" / "thumbnails" / "300" / "observations" / "snap.jpg"

    def test_get_thumbnail_path_custom_width(self, tmp_path: Path) -> None:
        """Test thumbnail path with custom width."""
        with patch.object(thumbnail, "get_media_dir", return_value=tmp_path):
            result = thumbnail.get_thumbnail_path("observations/snap.jpg", width=400)
            assert result == tmp_path / ".cache" / "thumbnails" / "400" / "observations" / "snap.jpg"

    def test_get_thumbnail_path_width_clamping(self, tmp_path: Path) -> None:
        """Test that width is clamped to valid range."""
        with patch.object(thumbnail, "get_media_dir", return_value=tmp_path):
            # Test minimum clamping
            result_min = thumbnail.get_thumbnail_path("snap.jpg", width=10)
            assert "50" in str(result_min)
            
            # Test maximum clamping
            result_max = thumbnail.get_thumbnail_path("snap.jpg", width=1000)
            assert "600" in str(result_max)

    def test_get_source_path(self, tmp_path: Path) -> None:
        """Test source path resolution."""
        with patch.object(thumbnail, "get_media_dir", return_value=tmp_path):
            result = thumbnail.get_source_path("observations/snap.jpg")
            assert result == tmp_path / "observations" / "snap.jpg"


class TestThumbnailStaleness:
    """Test thumbnail cache validation."""

    def test_is_thumbnail_stale_missing_thumb(self, tmp_path: Path) -> None:
        """Stale if thumbnail doesn't exist."""
        source = tmp_path / "source.jpg"
        source.write_bytes(b"fake image")
        thumb = tmp_path / "thumb.jpg"
        
        assert thumbnail.is_thumbnail_stale(thumb, source) is True

    def test_is_thumbnail_stale_missing_source(self, tmp_path: Path) -> None:
        """Stale if source doesn't exist."""
        source = tmp_path / "source.jpg"
        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"fake thumb")
        
        assert thumbnail.is_thumbnail_stale(thumb, source) is True

    def test_is_thumbnail_stale_thumb_older(self, tmp_path: Path) -> None:
        """Stale if source is newer than thumbnail."""
        source = tmp_path / "source.jpg"
        thumb = tmp_path / "thumb.jpg"
        
        # Create thumb first
        thumb.write_bytes(b"fake thumb")
        import time
        time.sleep(0.1)
        # Then create source (newer)
        source.write_bytes(b"fake image")
        
        assert thumbnail.is_thumbnail_stale(thumb, source) is True

    def test_is_thumbnail_stale_thumb_newer(self, tmp_path: Path) -> None:
        """Not stale if thumbnail is newer than source."""
        source = tmp_path / "source.jpg"
        thumb = tmp_path / "thumb.jpg"
        
        # Create source first
        source.write_bytes(b"fake image")
        import time
        time.sleep(0.1)
        # Then create thumb (newer)
        thumb.write_bytes(b"fake thumb")
        
        assert thumbnail.is_thumbnail_stale(thumb, source) is False


class TestThumbnailGeneration:
    """Test actual thumbnail generation."""

    def test_generate_thumbnail_missing_source(self, tmp_path: Path) -> None:
        """Returns False if source doesn't exist."""
        source = tmp_path / "missing.jpg"
        thumb = tmp_path / "thumb.jpg"
        
        result = thumbnail.generate_thumbnail(source, thumb)
        assert result is False
        assert not thumb.exists()

    def test_generate_thumbnail_creates_directories(self, tmp_path: Path) -> None:
        """Creates parent directories for thumbnail."""
        # Create a simple test image using PIL if available
        try:
            from PIL import Image
            
            source = tmp_path / "source.jpg"
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            img.save(source, "JPEG")
            
            thumb = tmp_path / "deep" / "nested" / "thumb.jpg"
            result = thumbnail.generate_thumbnail(source, thumb, width=50)
            
            assert result is True
            assert thumb.exists()
            assert thumb.parent.exists()
            
        except ImportError:
            pytest.skip("Pillow not available")

    def test_generate_thumbnail_resizes_correctly(self, tmp_path: Path) -> None:
        """Thumbnail is resized to specified width."""
        try:
            from PIL import Image
            
            source = tmp_path / "source.jpg"
            img = Image.new("RGB", (1920, 1080), (255, 0, 0))
            img.save(source, "JPEG")
            
            thumb = tmp_path / "thumb.jpg"
            result = thumbnail.generate_thumbnail(source, thumb, width=300)
            
            assert result is True
            
            # Check dimensions
            with Image.open(thumb) as thumb_img:
                assert thumb_img.width == 300
                # Height should maintain aspect ratio (16:9)
                expected_height = int(300 * (1080 / 1920))
                assert thumb_img.height == expected_height
                
        except ImportError:
            pytest.skip("Pillow not available")


class TestGetOrCreateThumbnail:
    """Test the main thumbnail retrieval function."""

    def test_get_or_create_thumbnail_generates_new(self, tmp_path: Path) -> None:
        """Generates thumbnail if it doesn't exist."""
        try:
            from PIL import Image
            
            with patch.object(thumbnail, "get_media_dir", return_value=tmp_path):
                # Create source image
                source = tmp_path / "test.jpg"
                img = Image.new("RGB", (800, 600), (0, 255, 0))
                img.save(source, "JPEG")
                
                result = thumbnail.get_or_create_thumbnail("test.jpg", width=200)
                
                assert result is not None
                assert result.exists()
                
                with Image.open(result) as thumb_img:
                    assert thumb_img.width == 200
                    
        except ImportError:
            pytest.skip("Pillow not available")

    def test_get_or_create_thumbnail_uses_cache(self, tmp_path: Path) -> None:
        """Returns cached thumbnail if valid."""
        try:
            from PIL import Image
            
            with patch.object(thumbnail, "get_media_dir", return_value=tmp_path):
                # Create source image
                source = tmp_path / "test.jpg"
                img = Image.new("RGB", (800, 600), (0, 255, 0))
                img.save(source, "JPEG")
                
                # Generate first time
                result1 = thumbnail.get_or_create_thumbnail("test.jpg", width=200)
                mtime1 = result1.stat().st_mtime if result1 else 0
                
                import time
                time.sleep(0.1)
                
                # Request again - should use cache
                result2 = thumbnail.get_or_create_thumbnail("test.jpg", width=200)
                mtime2 = result2.stat().st_mtime if result2 else 0
                
                # Same file, not regenerated
                assert mtime1 == mtime2
                
        except ImportError:
            pytest.skip("Pillow not available")


class TestThumbnailStats:
    """Test cache statistics."""

    def test_get_thumbnail_stats_empty(self, tmp_path: Path) -> None:
        """Returns empty stats for non-existent cache."""
        with patch.object(thumbnail, "get_thumbnail_cache_dir", return_value=tmp_path / "nonexistent"):
            stats = thumbnail.get_thumbnail_stats()
            assert stats["exists"] is False
            assert stats["count"] == 0

    def test_get_thumbnail_stats_with_files(self, tmp_path: Path) -> None:
        """Returns correct stats for existing cache."""
        cache_dir = tmp_path / ".cache" / "thumbnails"
        cache_dir.mkdir(parents=True)
        
        # Create some test files
        for i in range(3):
            (cache_dir / f"thumb{i}.jpg").write_bytes(b"x" * 100)
        
        with patch.object(thumbnail, "get_thumbnail_cache_dir", return_value=cache_dir):
            stats = thumbnail.get_thumbnail_stats()
            assert stats["exists"] is True
            assert stats["count"] == 3
            assert stats["size_bytes"] == 300
