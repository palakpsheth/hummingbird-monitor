from pathlib import Path
import types

import numpy as np
import pytest

import hbmon.worker as worker


def _setup_writer(monkeypatch, open_map: dict[str, bool], writes: list[tuple[Path, str]]):
    """
    Configure dummy cv2 VideoWriter.

    monkeypatch: pytest monkeypatch fixture used to patch cv2 references.
    open_map: map of fourcc string -> isOpened return value.
    writes: list collecting (Path, fourcc) tuples for each frame write.
    """
    class DummyWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.fourcc = fourcc
            self._opened = open_map.get(fourcc, True)

        def isOpened(self) -> bool:  # noqa: N802 (OpenCV style)
            return self._opened

        def write(self, frame) -> None:
            writes.append((self.path, self.fourcc))

        def release(self) -> None:
            self._opened = False

    fake_cv2 = types.SimpleNamespace(
        VideoWriter=DummyWriter,
        VideoWriter_fourcc=lambda *chars: "".join(chars),
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)


def _dummy_cap(frames: list[np.ndarray]):
    """
    Return a dummy VideoCapture that yields provided frames then stops.

    frames: list of numpy frames to emit.
    """
    class DummyCap:
        def __init__(self) -> None:
            self.idx = 0

        def read(self):
            if self.idx < len(frames):
                f = frames[self.idx]
                self.idx += 1
                return True, f
            return False, None

    return DummyCap()


def test_record_clip_prefers_avc1(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": True}, writes)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert writes and writes[0][0].suffix == ".mp4"
    assert writes[0][1] == "avc1"


def test_record_clip_falls_back_to_h264(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "H264": True}, writes)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert writes and writes[0][0].suffix == ".mp4"
    assert writes[0][1] == "H264"


def test_record_clip_falls_back_to_mp4v(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "H264": False, "mp4v": True}, writes)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert writes and writes[0][0].suffix == ".mp4"
    assert writes[0][1] == "mp4v"


def test_record_clip_falls_back_to_avi(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "H264": False, "mp4v": False, "XVID": True}, writes)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".avi"
    assert writes and writes[0][0].suffix == ".avi"
    assert writes[0][1] == "XVID"


def test_record_clip_converts_mp4v(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    class DummyWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.fourcc = fourcc
            self._opened = self.fourcc == "mp4v"

        def isOpened(self) -> bool:  # noqa: N802
            return self._opened

        def write(self, frame) -> None:
            self.path.write_bytes(b"frame")

        def release(self) -> None:
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoWriter=DummyWriter,
        VideoWriter_fourcc=lambda *chars: "".join(chars),
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)

    def fake_convert(input_path: Path, output_path: Path, *, timeout: float = 60.0) -> bool:
        output_path.write_bytes(b"converted")
        return True

    monkeypatch.setattr(worker, "_convert_to_h264", fake_convert)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert result_path.exists()


def test_record_clip_conversion_failure_cleans_tmp(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    class DummyWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.fourcc = fourcc
            self._opened = self.fourcc == "mp4v"

        def isOpened(self) -> bool:  # noqa: N802
            return self._opened

        def write(self, frame) -> None:
            self.path.write_bytes(b"frame")

        def release(self) -> None:
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoWriter=DummyWriter,
        VideoWriter_fourcc=lambda *chars: "".join(chars),
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)

    def fake_convert(input_path: Path, output_path: Path, *, timeout: float = 60.0) -> bool:
        output_path.write_bytes(b"converted")
        return False

    monkeypatch.setattr(worker, "_convert_to_h264", fake_convert)

    out_path = tmp_path / "clip.mp4"
    original_unlink = Path.unlink

    def fake_unlink(self):
        if self.suffix == ".mp4" and self.name.endswith("_h264.mp4"):
            raise OSError("boom")
        return original_unlink(self)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert result_path.exists()


def test_record_clip_conversion_rename_failure(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    class DummyWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.fourcc = fourcc
            self._opened = self.fourcc == "mp4v"

        def isOpened(self) -> bool:  # noqa: N802
            return self._opened

        def write(self, frame) -> None:
            self.path.write_bytes(b"frame")

        def release(self) -> None:
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoWriter=DummyWriter,
        VideoWriter_fourcc=lambda *chars: "".join(chars),
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)

    def fake_convert(input_path: Path, output_path: Path, *, timeout: float = 60.0) -> bool:
        output_path.write_bytes(b"converted")
        return True

    monkeypatch.setattr(worker, "_convert_to_h264", fake_convert)

    original_rename = Path.rename

    def fake_rename(self, target):
        if self.suffix == ".mp4" and self.name.endswith("_h264.mp4"):
            raise OSError("boom")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fake_rename)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert result_path.exists()


def test_record_clip_raises_when_all_codecs_fail(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "H264": False, "mp4v": False, "XVID": False}, writes)

    out_path = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError):
        worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)


def test_ffmpeg_available_returns_bool(monkeypatch):
    """Test that _ffmpeg_available returns a boolean."""
    # Test when ffmpeg is not found
    monkeypatch.setattr(worker.shutil, "which", lambda _: None)
    assert worker._ffmpeg_available() is False

    # Test when ffmpeg is found
    monkeypatch.setattr(worker.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    assert worker._ffmpeg_available() is True


def test_convert_to_h264_returns_false_when_ffmpeg_unavailable(monkeypatch, tmp_path):
    """Test that _convert_to_h264 returns False when FFmpeg is not available."""
    monkeypatch.setattr(worker.shutil, "which", lambda _: None)

    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"fake video")

    result = worker._convert_to_h264(input_path, output_path)
    assert result is False


def test_convert_to_h264_returns_false_on_subprocess_error(monkeypatch, tmp_path):
    """Test that _convert_to_h264 returns False when subprocess fails."""
    monkeypatch.setattr(worker.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    # Mock subprocess.run to simulate failure
    def mock_run(*args, **kwargs):
        result = types.SimpleNamespace()
        result.returncode = 1
        result.stderr = b"Error: mock failure"
        return result

    monkeypatch.setattr(worker.subprocess, "run", mock_run)

    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"fake video")

    result = worker._convert_to_h264(input_path, output_path)
    assert result is False


def test_draw_bbox_draws_rectangle(monkeypatch):
    """Test that _draw_bbox draws a rectangle on the frame without confidence label."""
    rectangles_drawn: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int], int]] = []

    fake_cv2 = types.SimpleNamespace(
        rectangle=lambda img, pt1, pt2, color, thickness: rectangles_drawn.append((pt1, pt2, color, thickness)),
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = worker.Det(x1=10, y1=20, x2=50, y2=60, conf=0.9)

    result = worker._draw_bbox(frame, det, show_confidence=False)

    # Should have drawn one rectangle
    assert len(rectangles_drawn) == 1
    pt1, pt2, color, thickness = rectangles_drawn[0]
    assert pt1 == (10, 20)
    assert pt2 == (50, 60)
    assert color == (0, 255, 0)  # green
    assert thickness == 2

    # Result should be a copy (different object)
    assert result is not frame


def test_draw_bbox_custom_color_and_thickness(monkeypatch):
    """Test that _draw_bbox uses custom color and thickness when provided."""
    rectangles_drawn: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int], int]] = []

    fake_cv2 = types.SimpleNamespace(
        rectangle=lambda img, pt1, pt2, color, thickness: rectangles_drawn.append((pt1, pt2, color, thickness)),
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = worker.Det(x1=5, y1=10, x2=30, y2=40, conf=0.8)

    worker._draw_bbox(frame, det, color=(255, 0, 0), thickness=5, show_confidence=False)

    assert len(rectangles_drawn) == 1
    _, _, color, thickness = rectangles_drawn[0]
    assert color == (255, 0, 0)  # custom color (blue in BGR)
    assert thickness == 5


def test_draw_bbox_with_confidence_label(monkeypatch):
    """Test that _draw_bbox draws confidence label when show_confidence=True."""
    rectangles_drawn: list[tuple] = []
    texts_drawn: list[tuple] = []

    fake_cv2 = types.SimpleNamespace(
        rectangle=lambda img, pt1, pt2, color, thickness: rectangles_drawn.append((pt1, pt2, color, thickness)),
        putText=lambda img, text, org, font, scale, color, thickness: texts_drawn.append((text, org, color)),
        getTextSize=lambda text, font, scale, thickness: ((30, 12), 2),
        FONT_HERSHEY_SIMPLEX=0,
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = worker.Det(x1=10, y1=30, x2=50, y2=60, conf=0.85)

    worker._draw_bbox(frame, det, show_confidence=True)

    # Should have drawn bbox rectangle + background rectangle for text
    assert len(rectangles_drawn) == 2
    # First is the bounding box
    assert rectangles_drawn[0][0] == (10, 30)
    assert rectangles_drawn[0][1] == (50, 60)

    # Should have drawn text with confidence
    assert len(texts_drawn) == 1
    assert texts_drawn[0][0] == worker._format_bbox_label(det)


def test_draw_bbox_raises_when_cv2_unavailable(monkeypatch):
    """Test that _draw_bbox raises RuntimeError when OpenCV is not available."""
    monkeypatch.setattr(worker, "_CV2_AVAILABLE", False)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = worker.Det(x1=0, y1=0, x2=10, y2=10, conf=0.5)

    with pytest.raises(RuntimeError, match="OpenCV"):
        worker._draw_bbox(frame, det)


def test_record_clip_from_rtsp_invokes_capture(monkeypatch, tmp_path):
    calls: dict[str, int] = {"set": 0, "release": 0}
    out_path = tmp_path / "clip.mp4"

    class DummyCap:
        def __init__(self, url: str) -> None:
            self.url = url

        def isOpened(self) -> bool:  # noqa: N802 (OpenCV style)
            return True

        def set(self, *_args, **_kwargs) -> None:
            calls["set"] += 1

        def release(self) -> None:
            calls["release"] += 1

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=DummyCap,
        CAP_PROP_BUFFERSIZE=2,
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)
    monkeypatch.setattr(worker, "_record_clip_opencv", lambda cap, out, seconds, max_fps=20.0: out)

    result = worker._record_clip_from_rtsp("rtsp://example", out_path, seconds=1.0)
    assert result == out_path
    assert calls["set"] == 1
    assert calls["release"] == 1


def test_record_clip_from_rtsp_raises_when_capture_fails(monkeypatch, tmp_path):
    out_path = tmp_path / "clip.mp4"

    class DummyCap:
        def __init__(self, url: str) -> None:
            self.url = url

        def isOpened(self) -> bool:  # noqa: N802 (OpenCV style)
            return False

        def release(self) -> None:
            return None

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=DummyCap,
        CAP_PROP_BUFFERSIZE=2,
    )

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", fake_cv2)

    with pytest.raises(RuntimeError, match="RTSP stream"):
        worker._record_clip_from_rtsp("rtsp://example", out_path, seconds=1.0)
