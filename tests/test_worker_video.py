from pathlib import Path
import types

import numpy as np
import pytest

import hbmon.worker as worker


def _setup_writer(monkeypatch, open_map: dict[str, bool], writes: list[tuple[Path, str]]):
    """Configure dummy cv2 VideoWriter; open_map controls isOpened per fourcc and records writes."""
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
    """Return a dummy VideoCapture that yields provided frames then stops."""
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


def test_record_clip_falls_back_to_mp4v(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "mp4v": True}, writes)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    assert writes and writes[0][0].suffix == ".mp4"
    assert writes[0][1] == "mp4v"


def test_record_clip_falls_back_to_avi(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "mp4v": False, "XVID": True}, writes)

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".avi"
    assert writes and writes[0][0].suffix == ".avi"
    assert writes[0][1] == "XVID"


def test_record_clip_raises_when_all_codecs_fail(monkeypatch, tmp_path):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    writes: list[tuple[Path, str]] = []
    _setup_writer(monkeypatch, {"avc1": False, "mp4v": False, "XVID": False}, writes)

    out_path = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError):
        worker._record_clip_opencv(_dummy_cap(frames), out_path, seconds=0.01, max_fps=5.0)
