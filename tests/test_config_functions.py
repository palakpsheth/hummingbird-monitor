"""
Additional tests for config module functions.

These tests cover the remaining config module utilities.
"""

from __future__ import annotations



import hbmon.config as config


class TestEnvBool:
    """Tests for the env_bool function."""

    def test_env_bool_true_values(self, monkeypatch):
        """Test env_bool recognizes true values."""
        for value in ["1", "true", "yes", "y", "on", "TRUE", "Yes", "Y", "ON"]:
            monkeypatch.setenv("TEST_BOOL", value)
            assert config.env_bool("TEST_BOOL", False) is True

    def test_env_bool_false_values(self, monkeypatch):
        """Test env_bool recognizes false values."""
        for value in ["0", "false", "no", "n", "off", "FALSE", "No", "N", "OFF"]:
            monkeypatch.setenv("TEST_BOOL", value)
            assert config.env_bool("TEST_BOOL", True) is False

    def test_env_bool_default_on_missing(self, monkeypatch):
        """Test env_bool returns default when missing."""
        monkeypatch.delenv("TEST_BOOL_MISSING", raising=False)
        assert config.env_bool("TEST_BOOL_MISSING", True) is True
        assert config.env_bool("TEST_BOOL_MISSING", False) is False

    def test_env_bool_default_on_unrecognized(self, monkeypatch):
        """Test env_bool returns default on unrecognized value."""
        monkeypatch.setenv("TEST_BOOL", "maybe")
        assert config.env_bool("TEST_BOOL", True) is True
        assert config.env_bool("TEST_BOOL", False) is False


class TestEnvInt:
    """Tests for the env_int function."""

    def test_env_int_valid(self, monkeypatch):
        """Test env_int parses valid integer."""
        monkeypatch.setenv("TEST_INT", "42")
        assert config.env_int("TEST_INT", 0) == 42

    def test_env_int_invalid(self, monkeypatch):
        """Test env_int returns default on invalid value."""
        monkeypatch.setenv("TEST_INT", "not_an_int")
        assert config.env_int("TEST_INT", 10) == 10


class TestEnvFloat:
    """Tests for the env_float function."""

    def test_env_float_valid(self, monkeypatch):
        """Test env_float parses valid float."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert config.env_float("TEST_FLOAT", 0.0) == 3.14

    def test_env_float_invalid(self, monkeypatch):
        """Test env_float returns default on invalid value."""
        monkeypatch.setenv("TEST_FLOAT", "not_a_float")
        assert config.env_float("TEST_FLOAT", 1.5) == 1.5


class TestSettingsWithEnvOverrides:
    """Tests for Settings.with_env_overrides method."""

    def test_with_env_overrides_applies_all(self, monkeypatch):
        """Test that with_env_overrides applies all env vars."""
        monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://test")
        monkeypatch.setenv("HBMON_CAMERA_NAME", "testcam")
        monkeypatch.setenv("HBMON_FPS_LIMIT", "15")
        monkeypatch.setenv("HBMON_DETECT_CONF", "0.40")
        monkeypatch.setenv("HBMON_DETECT_IOU", "0.50")
        monkeypatch.setenv("HBMON_MIN_BOX_AREA", "800")
        monkeypatch.setenv("HBMON_COOLDOWN_SECONDS", "4.0")
        monkeypatch.setenv("HBMON_MIN_SPECIES_PROB", "0.45")
        monkeypatch.setenv("HBMON_MATCH_THRESHOLD", "0.30")
        monkeypatch.setenv("HBMON_EMA_ALPHA", "0.15")
        monkeypatch.setenv("HBMON_TIMEZONE", "America/Chicago")

        s = config.Settings().with_env_overrides()

        assert s.rtsp_url == "rtsp://test"
        assert s.camera_name == "testcam"
        assert s.fps_limit == 15.0
        assert s.detect_conf == 0.40
        assert s.detect_iou == 0.50
        assert s.min_box_area == 800
        assert s.cooldown_seconds == 4.0
        assert s.min_species_prob == 0.45
        assert s.match_threshold == 0.30
        assert s.ema_alpha == 0.15
        assert s.timezone == "America/Chicago"

    def test_with_env_overrides_roi(self, monkeypatch):
        """Test that with_env_overrides applies ROI from env."""
        monkeypatch.setenv("HBMON_ROI", "0.1,0.2,0.8,0.9")

        s = config.Settings().with_env_overrides()

        assert s.roi is not None
        assert s.roi.x1 == 0.1
        assert s.roi.y1 == 0.2
        assert s.roi.x2 == 0.8
        assert s.roi.y2 == 0.9

    def test_with_env_overrides_invalid_roi(self, monkeypatch):
        """Test that invalid ROI env is ignored."""
        monkeypatch.setenv("HBMON_ROI", "invalid")

        s = config.Settings().with_env_overrides()
        # ROI should remain None
        assert s.roi is None


class TestRoi:
    """Tests for the Roi dataclass."""

    def test_roi_clamp_zero_area(self):
        """Test that clamp prevents zero-area ROI."""
        roi = config.Roi(x1=0.5, y1=0.5, x2=0.5, y2=0.5)
        clamped = roi.clamp()

        # Should have minimum size
        assert clamped.x2 > clamped.x1
        assert clamped.y2 > clamped.y1


class TestSettingsFromDict:
    """Tests for _settings_from_dict function."""

    def test_settings_from_dict_with_roi(self):
        """Test loading settings from dict with ROI."""
        data = {
            "rtsp_url": "rtsp://test",
            "camera_name": "cam1",
            "roi": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
        }

        s = config._settings_from_dict(data)

        assert s.rtsp_url == "rtsp://test"
        assert s.roi is not None
        assert s.roi.x1 == 0.1

    def test_settings_from_dict_without_roi(self):
        """Test loading settings from dict without ROI."""
        data = {
            "rtsp_url": "rtsp://test",
            "camera_name": "cam1",
        }

        s = config._settings_from_dict(data)

        assert s.roi is None

    def test_settings_from_dict_invalid_roi(self):
        """Test loading settings from dict with invalid ROI."""
        data = {
            "rtsp_url": "rtsp://test",
            "roi": {"x1": "invalid"},
        }

        s = config._settings_from_dict(data)

        assert s.roi is None


class TestLoadSaveSettings:
    """Tests for load_settings and save_settings functions."""

    def test_load_settings_creates_file(self, monkeypatch, tmp_path):
        """Test that load_settings creates config.json if missing."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

        config.load_settings()

        assert config.config_path().exists()

    def test_save_and_load_settings(self, monkeypatch, tmp_path):
        """Test saving and loading settings round-trip."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

        original = config.Settings(
            rtsp_url="rtsp://mystream",
            camera_name="mycam",
            detect_conf=0.42,
            roi=config.Roi(x1=0.1, y1=0.2, x2=0.8, y2=0.9),
        )
        config.save_settings(original)

        loaded = config.load_settings(apply_env_overrides=False)

        assert loaded.rtsp_url == "rtsp://mystream"
        assert loaded.camera_name == "mycam"
        assert loaded.detect_conf == 0.42
        assert loaded.roi is not None
        assert loaded.roi.x1 == 0.1

    def test_load_settings_handles_corrupt_file(self, monkeypatch, tmp_path):
        """Test that load_settings handles corrupt config.json."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

        # Create corrupt config file
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json")

        # Should not raise, should create new settings
        s = config.load_settings()
        assert isinstance(s, config.Settings)

    def test_load_settings_handles_non_object_root(self, monkeypatch, tmp_path):
        """Test that load_settings handles non-object JSON root."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

        # Create config with array root
        config_file = tmp_path / "config.json"
        config_file.write_text("[1, 2, 3]")

        s = config.load_settings()
        assert isinstance(s, config.Settings)


class TestRoiToStr:
    """Tests for roi_to_str function."""

    def test_roi_to_str_none(self):
        """Test roi_to_str with None."""
        assert config.roi_to_str(None) == ""

    def test_roi_to_str_valid(self):
        """Test roi_to_str with valid ROI."""
        roi = config.Roi(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
        result = config.roi_to_str(roi)

        assert "0.1" in result
        assert "0.2" in result
        assert "0.8" in result
        assert "0.9" in result


class TestPathFunctions:
    """Tests for path helper functions."""

    def test_data_dir(self, monkeypatch, tmp_path):
        """Test data_dir function."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        assert config.data_dir() == tmp_path

    def test_media_dir(self, monkeypatch, tmp_path):
        """Test media_dir function."""
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path))
        assert config.media_dir() == tmp_path

    def test_config_path(self, monkeypatch, tmp_path):
        """Test config_path function."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        assert config.config_path() == tmp_path / "config.json"

    def test_db_path(self, monkeypatch, tmp_path):
        """Test db_path function."""
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
        assert config.db_path() == tmp_path / "hbmon.sqlite"

    def test_snapshots_dir(self, monkeypatch, tmp_path):
        """Test snapshots_dir function."""
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path))
        assert config.snapshots_dir() == tmp_path / "snapshots"

    def test_clips_dir(self, monkeypatch, tmp_path):
        """Test clips_dir function."""
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path))
        assert config.clips_dir() == tmp_path / "clips"
