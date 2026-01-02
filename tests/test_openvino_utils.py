import sys
import types
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
    mock_core_instance.set_property.assert_called_once_with({"CACHE_DIR": str(cache_dir)})
    assert cache_dir.exists()


def test_get_core_uses_openvino_cache_dir(monkeypatch, tmp_path):
    from hbmon import openvino_utils

    class FakeCore:
        def __init__(self):
            self.set_property_calls = []

        def set_property(self, payload):
            self.set_property_calls.append(payload)

    openvino_utils._CORE = None
    monkeypatch.setattr(openvino_utils, "_OPENVINO_AVAILABLE", True)
    monkeypatch.setattr(openvino_utils, "Core", FakeCore)

    cache_dir = tmp_path / "ov_cache"
    monkeypatch.setenv("OPENVINO_CACHE_DIR", str(cache_dir))

    core = openvino_utils.get_core()

    assert isinstance(core, FakeCore)
    assert cache_dir.exists()
    assert core.set_property_calls == [{"CACHE_DIR": str(cache_dir)}]


def test_get_core_uses_data_dir_and_warns_on_cache_failure(monkeypatch, tmp_path, caplog):
    from hbmon import openvino_utils

    class FakeCore:
        def set_property(self, payload):
            raise RuntimeError("boom")

    openvino_utils._CORE = None
    monkeypatch.setattr(openvino_utils, "_OPENVINO_AVAILABLE", True)
    monkeypatch.setattr(openvino_utils, "Core", FakeCore)

    data_dir = tmp_path / "data_root"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENVINO_CACHE_DIR", raising=False)

    with caplog.at_level("WARNING"):
        openvino_utils.get_core()

    expected_cache_dir = data_dir / "openvino_cache"
    assert expected_cache_dir.exists()
    assert "Failed to set OpenVINO CACHE_DIR" in caplog.text


def test_get_available_openvino_devices_success(monkeypatch):
    from hbmon import openvino_utils

    class FakeCore:
        available_devices = ["CPU", "GPU"]

    monkeypatch.setattr(openvino_utils, "_OPENVINO_AVAILABLE", True)
    monkeypatch.setattr(openvino_utils, "get_core", lambda: FakeCore())

    assert openvino_utils.get_available_openvino_devices() == ["CPU", "GPU"]


def test_get_available_openvino_devices_exception(monkeypatch):
    from hbmon import openvino_utils

    monkeypatch.setattr(openvino_utils, "_OPENVINO_AVAILABLE", True)

    def raise_core():
        raise RuntimeError("nope")

    monkeypatch.setattr(openvino_utils, "get_core", raise_core)

    assert openvino_utils.get_available_openvino_devices() == []


@pytest.mark.parametrize("devices", [["GPU"], ["GPU.0"]])
def test_validate_openvino_gpu_devices(monkeypatch, devices):
    from hbmon import openvino_utils

    monkeypatch.setattr(openvino_utils, "get_available_openvino_devices", lambda: devices)

    assert openvino_utils.validate_openvino_gpu() is True


def test_force_openvino_gpu_override_patches_compile(monkeypatch):
    from hbmon import openvino_utils

    class FakeCore:
        def __init__(self):
            self.calls = []

        def compile_model(self, model, device_name=None, config=None):
            self.calls.append((model, device_name, config))
            return device_name

    openvino_module = types.ModuleType("openvino")
    openvino_runtime = types.ModuleType("openvino.runtime")
    openvino_runtime.Core = FakeCore
    openvino_module.runtime = openvino_runtime
    monkeypatch.setitem(sys.modules, "openvino", openvino_module)
    monkeypatch.setitem(sys.modules, "openvino.runtime", openvino_runtime)

    monkeypatch.setattr(openvino_utils, "_OPENVINO_AVAILABLE", True)

    openvino_utils.force_openvino_gpu_override()

    core = FakeCore()
    assert core.compile_model("model", None, None) == "GPU"
    assert core.compile_model("model", "AUTO", None) == "GPU"
    assert core.compile_model("model", "CPU", None) == "GPU"

    patched_compile = FakeCore.compile_model
    openvino_utils.force_openvino_gpu_override()
    assert FakeCore.compile_model is patched_compile
    assert getattr(FakeCore, "_is_hbmon_patched", False) is True


def test_select_clip_device_gpu_available(monkeypatch):
    from hbmon import openvino_utils

    monkeypatch.setattr(openvino_utils, "validate_openvino_gpu", lambda: True)

    assert openvino_utils.select_clip_device("openvino-gpu") == "GPU"


def test_select_clip_device_gpu_unavailable_warns(monkeypatch, caplog):
    from hbmon import openvino_utils

    monkeypatch.setattr(openvino_utils, "validate_openvino_gpu", lambda: False)

    with caplog.at_level("WARNING"):
        device = openvino_utils.select_clip_device("openvino-gpu")

    assert device == "CPU"
    assert "falling back to CPU" in caplog.text
