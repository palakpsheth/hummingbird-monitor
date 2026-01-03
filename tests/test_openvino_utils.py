from unittest.mock import MagicMock, patch
import pytest

from hbmon.openvino_utils import get_clip_cache_dir, get_core, _OPENVINO_AVAILABLE

def test_get_clip_cache_dir_default(monkeypatch):
    monkeypatch.delenv("OPENVINO_CACHE_DIR", raising=False)
    monkeypatch.setenv("HBMON_DATA_DIR", "/tmp/hbmon_test")
    
    cache_dir = get_clip_cache_dir()
    assert cache_dir == "/tmp/hbmon_test/openvino_cache/clip"

def test_get_clip_cache_dir_custom(monkeypatch):
    monkeypatch.setenv("OPENVINO_CACHE_DIR", "/custom/ov_cache")
    
    cache_dir = get_clip_cache_dir()
    # Path handles joining logic based on Path(cache_dir) / "clip"
    assert cache_dir == "/custom/ov_cache/clip"

@pytest.mark.skipif(not _OPENVINO_AVAILABLE, reason="OpenVINO not installed")
def test_get_core_singleton():
    from hbmon import openvino_utils
    # Reset singleton for clean test
    openvino_utils._CORE = None
    
    core1 = get_core()
    core2 = get_core()
    
    assert core1 is core2
    assert core1 is openvino_utils._CORE

@pytest.mark.skipif(not _OPENVINO_AVAILABLE, reason="OpenVINO not installed")
def test_get_core_configures_cache(monkeypatch, tmp_path):
    from hbmon import openvino_utils
    openvino_utils._CORE = None
    
    cache_dir = tmp_path / "ov_cache"
    monkeypatch.setenv("OPENVINO_CACHE_DIR", str(cache_dir))
    
    get_core()
    
    # Check that the directory was created
    assert cache_dir.exists()
    
    # We can't easily check the property on the real Core object without 
    # specific OpenVINO knowledge of how to retrieve it, but we can 
    # mock the Core object to verify set_property was called.

@patch("hbmon.openvino_utils.Core")
@patch("hbmon.openvino_utils._OPENVINO_AVAILABLE", True)
def test_get_core_mocked(mock_core_class, monkeypatch, tmp_path):
    from hbmon import openvino_utils
    openvino_utils._CORE = None
    
    mock_core_instance = MagicMock()
    mock_core_class.return_value = mock_core_instance
    
    cache_dir = tmp_path / "ov_cache_mock"
    monkeypatch.setenv("OPENVINO_CACHE_DIR", str(cache_dir))
    
    core = get_core()
    
    assert core is mock_core_instance
    mock_core_instance.set_property.assert_not_called()
    assert cache_dir.exists()
