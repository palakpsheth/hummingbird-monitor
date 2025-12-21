# tests/test_ffmpeg_recorder.py
from __future__ import annotations

from pathlib import Path

import pytest


def test_build_ffmpeg_command_for_clip(tmp_path: Path):
    """
    Spec: recorder builds a stable ffmpeg command using:
    - input: RTSP URL or pipe
    - output path (.mp4)
    - fps, codec, etc.
    """
    from hbmon.video.ffmpeg import build_record_command  # type: ignore

    out = tmp_path / "clip.mp4"
    cmd = build_record_command(
        input_url="rtsp://example/stream",
        output_path=out,
        duration_seconds=6.0,
        fps=15,
    )
    # Basic sanity
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    assert str(out) in cmd
    assert any(x in cmd for x in ["-t", "-to"]), "Must include duration control"


def test_recorder_invokes_subprocess(monkeypatch, tmp_path: Path):
    """
    Spec: record_clip() should call subprocess with expected args
    and raise a useful error on non-zero exit.
    """
    calls = []

    class DummyCompleted:
        def __init__(self, returncode: int):
            self.returncode = returncode
            self.stdout = b""
            self.stderr = b"oops"

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return DummyCompleted(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    from hbmon.video.ffmpeg import record_clip  # type: ignore

    out = tmp_path / "clip.mp4"
    record_clip(
        input_url="rtsp://example/stream",
        output_path=out,
        duration_seconds=2.0,
        fps=15,
    )

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0] == "ffmpeg"
    assert kwargs.get("check") is False  # we handle returncode ourselves
    assert kwargs.get("capture_output") is True


def test_recorder_raises_on_failure(monkeypatch, tmp_path: Path):
    class DummyCompleted:
        def __init__(self, returncode: int):
            self.returncode = returncode
            self.stdout = b""
            self.stderr = b"bad"

    def fake_run(cmd, **kwargs):
        return DummyCompleted(returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)

    from hbmon.video.ffmpeg import record_clip, FFMpegError  # type: ignore

    with pytest.raises(FFMpegError):
        record_clip(
            input_url="rtsp://example/stream",
            output_path=tmp_path / "clip.mp4",
            duration_seconds=1.0,
            fps=15,
        )
