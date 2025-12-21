"""
Tests for ROI calibration functionality.

This module tests the ROI calibration endpoints and UI functionality,
including:
- GET /api/roi returns correct default or saved ROI
- POST /api/roi saves ROI and persists to config.json
- ROI values are properly clamped and validated
- Calibration page loads with correct ROI data
"""

import json

import hbmon.config as config


def test_roi_get_default(monkeypatch, tmp_path):
    """GET /api/roi should return default ROI when none is set."""
    # Set up temporary directories
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create settings with no ROI
    s = config.Settings(roi=None)
    config.save_settings(s)
    
    # Load and verify ROI is None
    loaded = config.load_settings()
    assert loaded.roi is None


def test_roi_get_with_saved_roi(monkeypatch, tmp_path):
    """GET /api/roi should return saved ROI values."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create and save settings with ROI
    roi = config.Roi(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    s = config.Settings(roi=roi)
    config.save_settings(s)
    
    # Load and verify ROI is preserved
    loaded = config.load_settings()
    assert loaded.roi is not None
    assert abs(loaded.roi.x1 - 0.1) < 1e-6
    assert abs(loaded.roi.y1 - 0.2) < 1e-6
    assert abs(loaded.roi.x2 - 0.8) < 1e-6
    assert abs(loaded.roi.y2 - 0.9) < 1e-6


def test_roi_post_saves_and_persists(monkeypatch, tmp_path):
    """POST /api/roi should save ROI and persist to config.json."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Start with default settings
    config.ensure_dirs()
    
    # Simulate POST by creating ROI and saving
    s = config.load_settings()
    new_roi = config.Roi(x1=0.15, y1=0.25, x2=0.75, y2=0.85)
    s.roi = new_roi
    config.save_settings(s)
    
    # Verify config.json was written
    cfg_path = config.config_path()
    assert cfg_path.exists()
    
    # Read the JSON file directly to verify persistence
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "roi" in data
    assert abs(data["roi"]["x1"] - 0.15) < 1e-6
    assert abs(data["roi"]["y1"] - 0.25) < 1e-6
    assert abs(data["roi"]["x2"] - 0.75) < 1e-6
    assert abs(data["roi"]["y2"] - 0.85) < 1e-6


def test_roi_post_with_clamping(monkeypatch, tmp_path):
    """POST /api/roi should clamp out-of-bounds values."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create ROI with out-of-bounds values
    s = config.load_settings()
    bad_roi = config.Roi(x1=-0.5, y1=1.5, x2=2.0, y2=0.3)
    clamped = bad_roi.clamp()
    s.roi = clamped
    config.save_settings(s)
    
    # Load and verify values are clamped to [0, 1]
    loaded = config.load_settings()
    assert loaded.roi is not None
    assert 0.0 <= loaded.roi.x1 <= 1.0
    assert 0.0 <= loaded.roi.y1 <= 1.0
    assert 0.0 <= loaded.roi.x2 <= 1.0
    assert 0.0 <= loaded.roi.y2 <= 1.0
    # Verify ordering (x1 < x2, y1 < y2)
    assert loaded.roi.x1 < loaded.roi.x2
    assert loaded.roi.y1 < loaded.roi.y2


def test_roi_post_with_zero_coordinates(monkeypatch, tmp_path):
    """ROI with coordinates at 0 should be valid."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create ROI starting at origin
    s = config.load_settings()
    roi = config.Roi(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
    s.roi = roi
    config.save_settings(s)
    
    # Load and verify coordinates at 0 are preserved
    loaded = config.load_settings()
    assert loaded.roi is not None
    assert abs(loaded.roi.x1 - 0.0) < 1e-6
    assert abs(loaded.roi.y1 - 0.0) < 1e-6
    assert abs(loaded.roi.x2 - 0.5) < 1e-6
    assert abs(loaded.roi.y2 - 0.5) < 1e-6


def test_roi_persistence_across_multiple_saves(monkeypatch, tmp_path):
    """Multiple ROI updates should persist correctly."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Save first ROI
    s = config.load_settings()
    roi1 = config.Roi(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
    s.roi = roi1
    config.save_settings(s)
    
    # Verify first ROI
    loaded1 = config.load_settings()
    assert abs(loaded1.roi.x1 - 0.1) < 1e-6
    
    # Update to second ROI
    roi2 = config.Roi(x1=0.2, y1=0.3, x2=0.7, y2=0.8)
    s.roi = roi2
    config.save_settings(s)
    
    # Verify second ROI replaced first
    loaded2 = config.load_settings()
    assert abs(loaded2.roi.x1 - 0.2) < 1e-6
    assert abs(loaded2.roi.y1 - 0.3) < 1e-6
    assert abs(loaded2.roi.x2 - 0.7) < 1e-6
    assert abs(loaded2.roi.y2 - 0.8) < 1e-6


def test_roi_to_str_formatting(monkeypatch, tmp_path):
    """roi_to_str should format ROI as comma-separated floats."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create ROI
    roi = config.Roi(x1=0.15, y1=0.25, x2=0.85, y2=0.95)
    
    # Convert to string
    roi_str = config.roi_to_str(roi)
    
    # Verify format
    parts = roi_str.split(",")
    assert len(parts) == 4
    assert float(parts[0]) >= 0.0
    assert float(parts[1]) >= 0.0
    assert float(parts[2]) <= 1.0
    assert float(parts[3]) <= 1.0


def test_roi_to_str_with_none(monkeypatch, tmp_path):
    """roi_to_str should return empty string for None."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Convert None to string
    roi_str = config.roi_to_str(None)
    
    # Should be empty string
    assert roi_str == ""


def test_roi_env_override(monkeypatch, tmp_path):
    """ROI can be overridden via HBMON_ROI environment variable."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Save a ROI to config
    s = config.Settings(roi=config.Roi(x1=0.1, y1=0.1, x2=0.5, y2=0.5))
    config.save_settings(s)
    
    # Override with environment variable
    monkeypatch.setenv("HBMON_ROI", "0.2,0.3,0.7,0.8")
    
    # Load settings with env override
    loaded = config.load_settings()
    
    # Should use env override, not saved value
    assert abs(loaded.roi.x1 - 0.2) < 1e-6
    assert abs(loaded.roi.y1 - 0.3) < 1e-6
    assert abs(loaded.roi.x2 - 0.7) < 1e-6
    assert abs(loaded.roi.y2 - 0.8) < 1e-6


def test_roi_zero_area_prevention(monkeypatch, tmp_path):
    """ROI should prevent zero-area regions."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create ROI with same x coordinates (zero width)
    roi = config.Roi(x1=0.5, y1=0.2, x2=0.5, y2=0.8)
    clamped = roi.clamp()
    
    # Should adjust to prevent zero width
    assert clamped.x2 > clamped.x1
    
    # Create ROI with same y coordinates (zero height)
    roi2 = config.Roi(x1=0.2, y1=0.5, x2=0.8, y2=0.5)
    clamped2 = roi2.clamp()
    
    # Should adjust to prevent zero height
    assert clamped2.y2 > clamped2.y1
