"""
Tests that exercise saving and loading settings in ``hbmon.config`` as well
as database URL logic in ``hbmon.db``.  These tests operate on a
temporary directory to avoid polluting the real file system and ensure
permissions are handled correctly.
"""



import hbmon.config as config
import hbmon.db as db


def test_save_and_load_settings_roundtrip(monkeypatch, tmp_path):
    """Saving and loading settings should preserve values and respect env overrides."""
    # Point data/media directories to temporary paths
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    # Create a settings object with custom values
    s = config.Settings(rtsp_url="rtsptest", fps_limit=12.34, timezone="America/New_York")
    # Save to config.json
    config.save_settings(s)
    # After saving, config.json should exist
    cfg = config.config_path()
    assert cfg.exists()
    # Load settings back
    s2 = config.load_settings()
    assert s2.rtsp_url == "rtsptest"
    assert abs(s2.fps_limit - 12.34) < 1e-6
    assert s2.timezone == "America/New_York"
    # Override via environment
    monkeypatch.setenv("HBMON_RTSP_URL", "override")
    monkeypatch.setenv("HBMON_TIMEZONE", "UTC")
    s3 = config.load_settings()
    assert s3.rtsp_url == "override"
    assert s3.timezone == "UTC"
    # Explicitly opt out of env overrides to pick up persisted config values
    s4 = config.load_settings(apply_env_overrides=False)
    assert s4.rtsp_url == "rtsptest"
    assert s4.timezone == "America/New_York"


def test_load_settings_bootstrap_uses_env(monkeypatch, tmp_path):
    """When no config file exists, load_settings seeds values from env even without overrides."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("HBMON_RTSP_URL", "env_rtsp_url")
    cfg = config.config_path()
    if cfg.exists():
        cfg.unlink()
    s = config.load_settings(apply_env_overrides=False)
    assert s.rtsp_url == "env_rtsp_url"


def test_load_settings_corrupted_file(monkeypatch, tmp_path):
    """If config.json is corrupted, load_settings should fall back to defaults."""
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    # Ensure directories exist
    config.ensure_dirs()
    # Write an invalid JSON file
    cfg = config.config_path()
    cfg.write_text("not a json", encoding="utf-8")
    s = config.load_settings()
    # Since file is invalid, default rtsp_url should be empty string
    assert s.rtsp_url == ""
    # fps_limit should be default value (8.0)
    assert abs(s.fps_limit - 8.0) < 1e-6


def test_load_settings_corrupted_file_no_overrides(monkeypatch, tmp_path):
    """
    When config.json is corrupted and env overrides are disabled, fallback should still seed from env.
    This matches worker behavior (apply_env_overrides=False) while retaining bootstrap from env.
    """
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("HBMON_RTSP_URL", "env_rtsp_url")
    config.ensure_dirs()
    cfg = config.config_path()
    cfg.write_text("not a json", encoding="utf-8")
    s = config.load_settings(apply_env_overrides=False)
    assert s.rtsp_url == "env_rtsp_url"


def test_get_db_url(monkeypatch, tmp_path):
    """get_db_url returns the configured URL or constructs one based on data_dir."""
    # Explicit DB URL via env should be returned as-is
    monkeypatch.setenv("HBMON_DB_URL", "sqlite:///custom.db")
    assert db.get_db_url() == "sqlite:///custom.db"
    # Remove override and check default path; data_dir influences result
    monkeypatch.setenv("HBMON_DB_URL", "")
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    # Ensure directories exist so db_path returns within tmp_path
    config.ensure_dirs()
    url = db.get_db_url()
    # Should point to hbmon.sqlite under data dir with four slashes
    expected_path = config.db_path().as_posix().lstrip("/")
    assert url == f"sqlite:////{expected_path}"
