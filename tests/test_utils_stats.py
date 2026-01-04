
"""
Tests for system stats collection utilities.
"""
import pytest
from unittest.mock import MagicMock
import hbmon.utils as utils
import subprocess

def test_get_intel_gpu_load_success(monkeypatch):
    # Mock shutil.which to verify we check for the binary
    monkeypatch.setattr(utils.shutil, "which", lambda x: "/usr/bin/intel_gpu_top")
    
    # Mock subprocess.run to return sample JSON output
    # Simulating 2 samples. First is partial/init (often ignored or 0), second has load.
    # The real parser handles multiple JSON objects concatenated
    
    sample_json = """
    {
        "engines": { "Render/3D/0": { "busy": 10.0 } },
        "frequency": { "actual": 300 }
    }
    {
        "engines": { "Render/3D/0": { "busy": 55.5 }, "Video/0": { "busy": 20.0 } },
        "frequency": { "actual": 800 }
    }
    """
    
    mock_run = MagicMock()
    mock_run.return_value.stdout = sample_json
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    load = utils.get_intel_gpu_load()
    
    # Should check for intel_gpu_top with correct args
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "timeout"
    assert args[2] == "intel_gpu_top"
    assert "100" in args # Sampling period
    
    # Should pick the MAX load (55.5%)
    assert load == 55.5 # Floating point approximation check not needed if exact match, but float equality...
    # Wait, 55.5 is returned as float.
    assert abs(load - 55.5) < 0.1

def test_get_intel_gpu_load_parsing_robustness(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda x: "/usr/bin/intel_gpu_top")
    
    # Case: Empty output (timeout kill before output?)
    mock_run = MagicMock()
    mock_run.return_value.stdout = ""
    monkeypatch.setattr(subprocess, "run", mock_run)
    assert utils.get_intel_gpu_load() == 0.0 # Default to 0.0 if installed but no output
    
    # Case: Malformed JSON or partial last line
    mock_run.return_value.stdout = '{"engines": {"Render/3D/0": {"busy": 42.0}}}\n{"engines": {'
    assert utils.get_intel_gpu_load() == 42.0

def test_get_intel_gpu_load_not_installed(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda x: None)
    assert utils.get_intel_gpu_load() is None
