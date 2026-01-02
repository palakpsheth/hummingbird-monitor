"""
Tests for BackgroundRecorder in hbmon.recorder.

These tests verify the video writer initialization, codec fallback logic, and background
thread lifecycle (start, feed, stop) of the BackgroundRecorder class. The tests use a
stubbed cv2 module to simulate OpenCV's VideoWriter API without requiring actual video
encoding or filesystem operations.

Key scenarios tested:
- Successful writer initialization on the first codec
- Codec fallback when isOpened() returns False
- Complete initialization failure draining the queue and setting error
- Write exception handling that sets error without deadlocking
- Queue draining behavior and background thread termination
"""

from __future__ import annotations

import types
from unittest.mock import patch


def _make_cv2_stub(
    *,
    open_results: dict[str, bool] | None = None,
    raise_on_init: set[str] | None = None,
    raise_on_write: bool = False,
):
    """Create a lightweight ``cv2`` stub for testing ``hbmon.recorder``.

    The returned object is a :class:`types.ModuleType` instance that mimics the small
    subset of the OpenCV API used by the recorder code:

    * ``VideoWriter_fourcc(*codec)`` – returns the codec string by concatenating the
      individual characters.
    * ``VideoWriter(...)`` – returns a ``FakeVideoWriter`` instance that records
      initialization parameters, write calls, and release status.

    The behavior of the stubbed ``VideoWriter`` can be controlled via the parameters
    to simulate success and failure scenarios without requiring the real OpenCV
    dependency.

    Parameters
    ----------
    open_results:
        Mapping from codec string (e.g. ``"avc1"``, ``"H264"``) to a boolean
        indicating whether the corresponding ``FakeVideoWriter`` instance should
        report success from :meth:`isOpened`. Codecs not present in this mapping
        default to ``False`` (i.e. ``isOpened()`` will return ``False``).
    raise_on_init:
        Set of codec strings for which ``FakeVideoWriter.__init__`` should raise
        :class:`RuntimeError`. This is used to test how the recorder handles
        initialization failures for specific codecs.
    raise_on_write:
        When ``True``, :meth:`FakeVideoWriter.write` raises
        :class:`RuntimeError("write failed")` on every call instead of recording
        frames. This is used to verify that the recorder surfaces write errors and
        drains its internal queue.

    Returns
    -------
    types.ModuleType
        A module-like object that can be patched into :mod:`sys.modules` under the
        name ``"cv2"``.

    Examples
    --------
    Basic usage in tests mirrors the patterns in this module::

        cv2_stub = _make_cv2_stub(open_results={"avc1": True}, raise_on_write=False)
        with patch("hbmon.recorder.cv2", cv2_stub):
            from hbmon.recorder import BackgroundRecorder

            recorder = BackgroundRecorder(tmp_path / "out.mp4", fps=30.0, width=640, height=480)
            recorder.start()
            recorder.feed({"frame": 1})
            recorder.stop()

            assert recorder.error is None
            assert cv2_stub.instances[0].writes == [{"frame": 1}]

    """
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.open_results = open_results or {}
    cv2_stub.raise_on_init = raise_on_init or set()
    cv2_stub.raise_on_write = raise_on_write
    cv2_stub.instances = []

    def video_writer_fourcc(*codec: str) -> str:
        return "".join(codec)

    class FakeVideoWriter:
        """Test stub that simulates :class:`cv2.VideoWriter`.

        This fake writer mimics the minimal behavior used by the recorder tests:

        * `isOpened()` reports whether the writer successfully opened, based on `open_results`.
        * `write()` appends frames to ``self.writes`` and can raise on write when configured.
        * `release()` marks the writer as released via ``self.released``.
        """

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


def test_recorder_writes_frames_with_first_codec(tmp_path):
    """Verify BackgroundRecorder writes frames when the first codec (avc1) initializes successfully."""
    cv2_stub = _make_cv2_stub(open_results={"avc1": True})

    with patch("hbmon.recorder.cv2", cv2_stub):
        from hbmon.recorder import BackgroundRecorder

        recorder = BackgroundRecorder(tmp_path / "out.mp4", fps=30.0, width=640, height=480)
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


def test_recorder_falls_back_across_codecs(tmp_path):
    """Verify BackgroundRecorder falls back to H264 when avc1 fails to open."""
    cv2_stub = _make_cv2_stub(open_results={"avc1": False, "H264": True})

    with patch("hbmon.recorder.cv2", cv2_stub):
        from hbmon.recorder import BackgroundRecorder

        recorder = BackgroundRecorder(tmp_path / "out.mp4", fps=24.0, width=320, height=240)
        recorder.start()
        recorder.feed("frame")
        recorder.stop()

        assert recorder.error is None
        assert not recorder.thread.is_alive()
        assert len(cv2_stub.instances) == 2
        assert cv2_stub.instances[0].released is True
        assert cv2_stub.instances[1].writes == ["frame"]


def test_recorder_failure_sets_error_and_drains_queue(tmp_path):
    """Verify BackgroundRecorder sets an error and drains the queue when all codecs fail to initialize."""
    cv2_stub = _make_cv2_stub(open_results={"avc1": False, "H264": False, "mp4v": False, "XVID": False})

    with patch("hbmon.recorder.cv2", cv2_stub):
        from hbmon.recorder import BackgroundRecorder

        recorder = BackgroundRecorder(tmp_path / "out.mp4", fps=15.0, width=100, height=100)
        recorder.feed("frame")
        recorder.feed("frame2")
        recorder.start()
        recorder.thread.join(timeout=1)

        assert recorder.error == "Failed to initialize any compatible VideoWriter"
        assert not recorder.thread.is_alive()
        assert recorder.queue.empty()


def test_recorder_sets_error_on_write_exception(tmp_path):
    """Verify BackgroundRecorder sets an error when a write operation raises an exception."""
    cv2_stub = _make_cv2_stub(open_results={"avc1": True}, raise_on_write=True)

    with patch("hbmon.recorder.cv2", cv2_stub):
        from hbmon.recorder import BackgroundRecorder

        recorder = BackgroundRecorder(tmp_path / "out.mp4", fps=30.0, width=640, height=480)
        recorder.start()
        recorder.feed("frame")
        recorder.stop()

        assert recorder.error == "write failed"
        assert not recorder.thread.is_alive()
