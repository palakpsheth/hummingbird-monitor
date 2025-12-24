"""
Tests for the background image feature in hbmon.config.

These tests verify that background_image_path, background_dir, and
the Settings.background_image field work correctly.
"""

import hbmon.config as config


def test_background_dir_path(monkeypatch, tmp_path):
    """background_dir returns a path under data_dir."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    config.ensure_dirs()
    
    bg_dir = config.background_dir()
    assert bg_dir.parent == config.data_dir()
    assert bg_dir.name == "background"


def test_background_image_path(monkeypatch, tmp_path):
    """background_image_path returns the correct path for the background image."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    config.ensure_dirs()
    
    bg_path = config.background_image_path()
    assert bg_path.parent == config.background_dir()
    assert bg_path.name == "background.jpg"


def test_ensure_dirs_creates_background_dir(monkeypatch, tmp_path):
    """ensure_dirs should create the background directory."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Should not exist yet
    bg_dir = tmp_path / "data" / "background"
    assert not bg_dir.exists()
    
    config.ensure_dirs()
    
    # Should exist now
    assert bg_dir.exists()
    assert bg_dir.is_dir()


def test_settings_background_image_default():
    """Settings.background_image should default to empty string."""
    s = config.Settings()
    assert s.background_image == ""


def test_settings_background_image_roundtrip(monkeypatch, tmp_path):
    """background_image should be preserved when saving and loading settings."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create settings with background_image
    s = config.Settings(background_image="background/background.jpg")
    config.save_settings(s)
    
    # Load and verify
    s2 = config.load_settings()
    assert s2.background_image == "background/background.jpg"


def test_settings_background_image_empty_roundtrip(monkeypatch, tmp_path):
    """Empty background_image should be preserved when saving and loading."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    
    # Create settings without background_image
    s = config.Settings()
    config.save_settings(s)
    
    # Load and verify
    s2 = config.load_settings()
    assert s2.background_image == ""
