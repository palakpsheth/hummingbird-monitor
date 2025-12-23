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
