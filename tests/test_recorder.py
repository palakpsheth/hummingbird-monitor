from __future__ import annotations

import importlib
import sys
import types

import pytest


def _make_cv2_stub(
    *,
    open_results: dict[str, bool] | None = None,
    raise_on_init: set[str] | None = None,
    raise_on_write: bool = False,
):
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.open_results = open_results or {}
    cv2_stub.raise_on_init = raise_on_init or set()
    cv2_stub.raise_on_write = raise_on_write
    cv2_stub.instances = []

    def video_writer_fourcc(*codec: str) -> str:
        return "".join(codec)

    class FakeVideoWriter:
        def __init__(self, path: str, fourcc: str, fps: float, size: tuple[int, int]):
            if fourcc in cv2_stub.raise_on_init:
                raise RuntimeError(f"init failed for {fourcc}")
            self.path = path
            self.fourcc = fourcc
            self.fps = fps
            self.size = size
            self._opened = cv2_stub.open_results.get(fourcc, False)
            self.released = False
            self.writes: list[object] = []
            cv2_stub.instances.append(self)

        def isOpened(self) -> bool:  # noqa: N802 - match OpenCV API
            return self._opened

        def write(self, frame: object) -> None:
            if cv2_stub.raise_on_write:
                raise RuntimeError("write failed")
            self.writes.append(frame)

        def release(self) -> None:
            self.released = True

    cv2_stub.VideoWriter_fourcc = video_writer_fourcc
    cv2_stub.VideoWriter = FakeVideoWriter
    return cv2_stub


def _load_recorder(monkeypatch: pytest.MonkeyPatch, cv2_stub: types.ModuleType):
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    import hbmon.recorder as recorder

    importlib.reload(recorder)
    return recorder.BackgroundRecorder


def test_recorder_writes_frames_with_first_codec(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cv2_stub = _make_cv2_stub(open_results={"avc1": True})
    recorder_cls = _load_recorder(monkeypatch, cv2_stub)

    recorder = recorder_cls(tmp_path / "out.mp4", fps=30.0, width=640, height=480)
    recorder.start()
    recorder.feed({"frame": 1})
    recorder.feed({"frame": 2})
    recorder.stop()

    assert recorder.error is None
    assert not recorder.thread.is_alive()
    assert len(cv2_stub.instances) == 1
    writer = cv2_stub.instances[0]
    assert writer.writes == [{"frame": 1}, {"frame": 2}]
    assert writer.released is True


def test_recorder_fallbacks_across_codecs(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cv2_stub = _make_cv2_stub(open_results={"avc1": False, "H264": True})
    recorder_cls = _load_recorder(monkeypatch, cv2_stub)

    recorder = recorder_cls(tmp_path / "out.mp4", fps=24.0, width=320, height=240)
    recorder.start()
    recorder.feed("frame")
    recorder.stop()

    assert recorder.error is None
    assert not recorder.thread.is_alive()
    assert len(cv2_stub.instances) == 2
    assert cv2_stub.instances[0].released is True
    assert cv2_stub.instances[1].writes == ["frame"]


def test_recorder_failure_sets_error_and_drains_queue(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cv2_stub = _make_cv2_stub(open_results={"avc1": False, "H264": False, "mp4v": False, "XVID": False})
    recorder_cls = _load_recorder(monkeypatch, cv2_stub)

    recorder = recorder_cls(tmp_path / "out.mp4", fps=15.0, width=100, height=100)
    recorder.feed("frame")
    recorder.feed("frame2")
    recorder.start()
    recorder.thread.join(timeout=1)

    assert recorder.error == "Failed to initialize any compatible VideoWriter"
    assert not recorder.thread.is_alive()
    assert recorder.queue.empty()


def test_recorder_sets_error_on_write_exception(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cv2_stub = _make_cv2_stub(open_results={"avc1": True}, raise_on_write=True)
    recorder_cls = _load_recorder(monkeypatch, cv2_stub)

    recorder = recorder_cls(tmp_path / "out.mp4", fps=30.0, width=640, height=480)
    recorder.start()
    recorder.feed("frame")
    recorder.stop()

    assert recorder.error == "write failed"
    assert not recorder.thread.is_alive()
