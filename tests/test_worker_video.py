from pathlib import Path
import types

import numpy as np

import hbmon.worker as worker


def test_record_clip_prefers_avc1(monkeypatch, tmp_path):
    """
    Ensure clip recording prefers an AVC1/MP4 writer when available so
    browser playback works without downloading the entire file first.
    """

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    class DummyCap:
        def __init__(self) -> None:
            self.idx = 0

        def read(self):
            if self.idx < len(frames):
                f = frames[self.idx]
                self.idx += 1
                return True, f
            return False, None

    writes: list[tuple[Path, str]] = []

    class DummyWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.fourcc = fourcc
            self._opened = True

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

    out_path = tmp_path / "clip.mp4"
    result_path = worker._record_clip_opencv(DummyCap(), out_path, seconds=0.01, max_fps=5.0)

    assert result_path.suffix == ".mp4"
    # First writer should use avc1 for streaming-friendly MP4 files
    assert writes and writes[0][0].suffix == ".mp4"
    assert writes[0][1] == "avc1"
