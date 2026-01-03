from unittest.mock import patch


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
        print(f"\nDEBUG mock_yolo calls: {mock_yolo.call_args_list}")
        # Check that YOLO was called with the correct path
        mock_yolo.assert_any_call(str(expected_path), task="detect")
        
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
        mock_yolo.assert_any_call(str(expected_path_2), task="detect")


