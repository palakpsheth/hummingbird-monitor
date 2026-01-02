import pytest

from hbmon.yolo_utils import resolve_predict_imgsz


@pytest.mark.parametrize(
    ("imgsz_env", "frame_shape", "expected"),
    [
        ("auto", None, [1088, 1920]),
        ("auto", (1080, 1920), [1088, 1920]),
        ("auto", (721, 1281), [736, 1312]),
    ],
)
def test_resolve_predict_imgsz_auto(imgsz_env, frame_shape, expected):
    assert resolve_predict_imgsz(imgsz_env, frame_shape) == expected


@pytest.mark.parametrize(
    ("imgsz_env", "expected"),
    [
        ("640, 480", [640, 480]),
        (" 640 , 480 ", [640, 480]),
        ("640", [640]),
        ("640,", [640]),
        ("", [1088, 1920]),
        ("   ", [1088, 1920]),
        ("notanint", [1088, 1920]),
        ("640,notanint", [1088, 1920]),
    ],
)
def test_resolve_predict_imgsz_parsing(imgsz_env, expected):
    assert resolve_predict_imgsz(imgsz_env) == expected
