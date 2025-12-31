from unittest.mock import MagicMock, patch

from hbmon.worker import _load_yolo_model

@patch("hbmon.worker.YOLO")
@patch("hbmon.worker.is_openvino_available")
@patch("hbmon.worker.validate_openvino_gpu")
def test_load_yolo_model_path_resolution(mock_gpu, mock_ov_av, mock_yolo, monkeypatch, tmp_path):
    # Setup mocks
    mock_ov_av.return_value = True
    mock_gpu.return_value = True
    
    # 1. Test with OPENVINO_CACHE_DIR set
    cache_dir = tmp_path / "ov_cache"
    monkeypatch.setenv("HBMON_YOLO_BACKEND", "openvino-gpu")
    monkeypatch.setenv("OPENVINO_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HBMON_YOLO_MODEL", "yolo11n.pt")
    
    expected_path = cache_dir / "yolo" / "yolo11n_openvino_model"
    
    def exists_side_effect(self_path):
        return str(self_path) == str(expected_path)

    # Mock Path.exists to return True for the model
    with patch("hbmon.worker.Path.exists", autospec=True, side_effect=exists_side_effect):
        _load_yolo_model()
        # Check that YOLO was called with the correct path
        mock_yolo.assert_any_call(str(expected_path))
        
    # 2. Test without OPENVINO_CACHE_DIR but with YOLO_CONFIG_DIR
    mock_yolo.reset_mock()
    monkeypatch.delenv("OPENVINO_CACHE_DIR")
    yolo_config = tmp_path / "yolo_config"
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(yolo_config))
    
    expected_path_2 = yolo_config / "yolo11n_openvino_model"
    
    def exists_side_effect_2(self_path):
        return str(self_path) == str(expected_path_2)

    with patch("hbmon.worker.Path.exists", autospec=True, side_effect=exists_side_effect_2):
        _load_yolo_model()
        mock_yolo.assert_any_call(str(expected_path_2))

@patch("hbmon.worker.YOLO")
@patch("hbmon.worker.is_openvino_available")
@patch("shutil.move")
@patch("shutil.rmtree")
def test_load_yolo_model_export_path(mock_rmtree, mock_move, mock_ov_av, mock_yolo, monkeypatch, tmp_path):
    mock_ov_av.return_value = True
    
    cache_dir = tmp_path / "ov_cache_export"
    monkeypatch.setenv("HBMON_YOLO_BACKEND", "openvino-cpu")
    monkeypatch.setenv("OPENVINO_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HBMON_YOLO_MODEL", "yolo11n.pt")
    
    expected_path = cache_dir / "yolo" / "yolo11n_openvino_model"
    expected_parent = cache_dir / "yolo"
    
    def exists_side_effect_3(self_path):
        # Always return False for the model path to trigger export
        if str(self_path) == str(expected_path):
            return False
        return True

    # Mock Path.exists
    with patch("hbmon.worker.Path.exists", autospec=True, side_effect=exists_side_effect_3):
        # Mock the YOLO instance and its export method
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = "/tmp/fake_export_path"
        mock_yolo.return_value = mock_yolo_instance
        
        _load_yolo_model()
        
        # Verify it tried to create the directory
        assert expected_parent.exists()
        
        # Verify export was called
        mock_yolo_instance.export.assert_called_once_with(format="openvino", half=False)
        
        # Verify shutil.move was called with the correct arguments
        mock_move.assert_called_once_with("/tmp/fake_export_path", str(expected_path))
