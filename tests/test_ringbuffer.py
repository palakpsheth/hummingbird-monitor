# tests/test_ringbuffer.py
from __future__ import annotations


import numpy as np


def test_ring_buffer_keeps_last_n_seconds():
    """
    Spec for your pre-trigger buffer.
    Expected behavior:
    - Maintain frames with timestamps.
    - When asked for "last X seconds", return only frames within that window.
    """
    from hbmon.video.ringbuffer import FrameRingBuffer  # type: ignore

    buf = FrameRingBuffer(max_seconds=2.0)

    t0 = 1000.0
    # Insert 5 frames 0.5s apart (covers 2.0s span)
    for i in range(5):
        ts = t0 + i * 0.5
        frame = np.full((2, 2, 3), i, dtype=np.uint8)
        buf.push(frame=frame, timestamp=ts)

    last_1s = buf.get_last(seconds=1.0, now=t0 + 2.0)
    # At t0+2.0, last 1.0s includes frames at 1.5 and 2.0
    assert len(last_1s) == 2
    assert int(last_1s[0]["frame"][0, 0, 0]) == 3
    assert int(last_1s[1]["frame"][0, 0, 0]) == 4

    last_3s = buf.get_last(seconds=3.0, now=t0 + 2.0)
    # Buffer only stores 2s window by design; should return all it has.
    assert len(last_3s) == 5


def test_ring_buffer_drops_old_frames():
    from hbmon.video.ringbuffer import FrameRingBuffer  # type: ignore

    buf = FrameRingBuffer(max_seconds=1.0)
    t0 = 2000.0

    for i in range(10):
        buf.push(frame=np.zeros((1, 1, 3), dtype=np.uint8), timestamp=t0 + i * 0.25)

    # At t0+2.25, only last ~1s should remain => roughly 4-5 frames
    frames = buf.get_last(seconds=10.0, now=t0 + 2.25)
    assert 3 <= len(frames) <= 6
    assert frames[-1]["timestamp"] == t0 + 9 * 0.25
