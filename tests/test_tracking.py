"""
Unit tests for tracking-based visit detection.

Tests the TrackedDet, TrackState dataclasses and _collect_tracked_detections function.
"""

from unittest.mock import MagicMock
import numpy as np
from pathlib import Path

from hbmon.worker import (
    Det,
    TrackedDet,
    TrackState,
    VisitState,
    CandidateItem,
    _collect_tracked_detections,
)


class TestTrackedDet:
    """Tests for TrackedDet dataclass."""

    def test_tracked_det_creation(self):
        """TrackedDet can be created with track_id."""
        det = TrackedDet(x1=10, y1=20, x2=110, y2=120, conf=0.5, track_id=42)
        assert det.x1 == 10
        assert det.y1 == 20
        assert det.x2 == 110
        assert det.y2 == 120
        assert det.conf == 0.5
        assert det.track_id == 42

    def test_tracked_det_inherits_area(self):
        """TrackedDet inherits area property from Det."""
        det = TrackedDet(x1=0, y1=0, x2=100, y2=50, conf=0.8, track_id=1)
        assert det.area == 5000  # 100 * 50

    def test_tracked_det_default_track_id(self):
        """TrackedDet has default track_id of -1."""
        det = TrackedDet(x1=0, y1=0, x2=10, y2=10, conf=0.5)
        assert det.track_id == -1

    def test_tracked_det_is_det_subclass(self):
        """TrackedDet is a subclass of Det."""
        det = TrackedDet(x1=0, y1=0, x2=10, y2=10, conf=0.5, track_id=1)
        assert isinstance(det, Det)


class TestTrackState:
    """Tests for TrackState dataclass."""

    def test_track_state_creation(self):
        """TrackState can be created with required fields."""
        state = TrackState(
            track_id=1,
            state=VisitState.IDLE,
            start_ts=1000.0,
            last_seen_ts=1000.0
        )
        assert state.track_id == 1
        assert state.state == VisitState.IDLE
        assert state.start_ts == 1000.0
        assert state.last_seen_ts == 1000.0
        assert state.best_candidate is None
        assert state.best_score == (-1.0, 0)
        assert state.recorder is None
        assert state.video_path is None
        assert state.frames_tracked == 0

    def test_track_state_lifecycle_idle_to_recording(self):
        """TrackState transitions from IDLE to RECORDING."""
        state = TrackState(
            track_id=1,
            state=VisitState.IDLE,
            start_ts=1000.0,
            last_seen_ts=1000.0
        )
        # Simulate detection → start recording
        state.state = VisitState.RECORDING
        state.frames_tracked = 1
        
        assert state.state == VisitState.RECORDING
        assert state.frames_tracked == 1

    def test_track_state_lifecycle_recording_to_finalizing(self):
        """TrackState transitions from RECORDING to FINALIZING on track loss."""
        state = TrackState(
            track_id=1,
            state=VisitState.RECORDING,
            start_ts=1000.0,
            last_seen_ts=1005.0,
            frames_tracked=100
        )
        # Simulate track lost → finalize
        state.state = VisitState.FINALIZING
        
        assert state.state == VisitState.FINALIZING

    def test_track_state_with_video_path(self):
        """TrackState can hold video path."""
        video_path = Path("/data/clips/20260105/abc123.mp4")
        state = TrackState(
            track_id=1,
            state=VisitState.RECORDING,
            start_ts=1000.0,
            last_seen_ts=1000.0,
            video_path=video_path
        )
        assert state.video_path == video_path

    def test_track_state_best_score_update(self):
        """TrackState best_score can be compared and updated."""
        state = TrackState(
            track_id=1,
            state=VisitState.RECORDING,
            start_ts=1000.0,
            last_seen_ts=1000.0
        )
        
        # First detection
        score1 = (0.3, 1000)
        if score1 > state.best_score:
            state.best_score = score1
        assert state.best_score == (0.3, 1000)
        
        # Better detection (higher confidence)
        score2 = (0.5, 800)
        if score2 > state.best_score:
            state.best_score = score2
        assert state.best_score == (0.5, 800)
        
        # Worse detection (lower confidence)
        score3 = (0.4, 1200)
        if score3 > state.best_score:
            state.best_score = score3
        assert state.best_score == (0.5, 800)  # Unchanged


class TestCollectTrackedDetections:
    """Tests for _collect_tracked_detections function."""

    def _mock_box(self, x1, y1, x2, y2, conf, cls, track_id=None):
        """Create a mock box object."""
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].detach.return_value.cpu.return_value.numpy.return_value = np.array([x1, y1, x2, y2])
        box.conf = MagicMock()
        box.conf.item.return_value = conf
        box.cls = MagicMock()
        box.cls.item.return_value = cls
        if track_id is not None:
            box.id = MagicMock()
            box.id.item.return_value = track_id
        else:
            box.id = None
        return box

    def _mock_results(self, boxes, names=None):
        """Create mock YOLO results."""
        r0 = MagicMock()
        r0.boxes = boxes
        r0.names = names or {14: 'bird'}
        return [r0]

    def test_empty_results(self):
        """Returns empty list for empty results."""
        dets = _collect_tracked_detections([], 100, 14)
        assert dets == []

    def test_none_boxes(self):
        """Returns empty list when boxes is None."""
        r0 = MagicMock()
        r0.boxes = None
        dets = _collect_tracked_detections([r0], 100, 14)
        assert dets == []

    def test_untracked_gets_synthetic_id(self):
        """Untracked detections get synthetic negative track IDs."""
        box_tracked = self._mock_box(0, 0, 100, 100, 0.8, 14, track_id=1)
        box_untracked = self._mock_box(0, 0, 100, 100, 0.9, 14, track_id=None)
        
        results = self._mock_results([box_tracked, box_untracked])
        dets = _collect_tracked_detections(results, 100, 14)
        
        # Both should be returned - untracked gets synthetic negative ID
        assert len(dets) == 2
        assert dets[0].track_id == 1  # Real track ID
        assert dets[1].track_id < 0   # Synthetic negative track ID

    def test_filters_non_bird(self):
        """Filters out non-bird detections."""
        box_bird = self._mock_box(0, 0, 100, 100, 0.8, 14, track_id=1)
        box_person = self._mock_box(0, 0, 100, 100, 0.9, 0, track_id=2)  # cls=0 is person
        
        results = self._mock_results([box_bird, box_person])
        dets = _collect_tracked_detections(results, 100, 14)
        
        assert len(dets) == 1
        assert dets[0].track_id == 1

    def test_filters_small_boxes(self):
        """Filters out boxes below min_box_area."""
        box_large = self._mock_box(0, 0, 100, 100, 0.8, 14, track_id=1)  # area=10000
        box_small = self._mock_box(0, 0, 10, 10, 0.9, 14, track_id=2)    # area=100
        
        results = self._mock_results([box_large, box_small])
        dets = _collect_tracked_detections(results, 500, 14)  # min_area=500
        
        assert len(dets) == 1
        assert dets[0].track_id == 1

    def test_extracts_track_id(self):
        """Correctly extracts track_id from box.id."""
        box = self._mock_box(10, 20, 110, 120, 0.75, 14, track_id=42)
        
        results = self._mock_results([box])
        dets = _collect_tracked_detections(results, 100, 14)
        
        assert len(dets) == 1
        assert dets[0].track_id == 42
        assert dets[0].conf == 0.75
        assert dets[0].x1 == 10
        assert dets[0].y1 == 20


class TestCandidateItemTracking:
    """Tests for tracking metadata in CandidateItem."""

    def test_candidate_item_tracking_fields(self):
        """CandidateItem can hold tracking metadata."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=0, y1=0, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(
            frame=frame,
            det_full=det,
            track_id=42,
            tracking_stats={
                "track_id": 42,
                "frames_tracked": 100,
                "track_high_thresh": 0.1,
                "track_low_thresh": 0.01,
            }
        )
        
        assert item.track_id == 42
        assert item.tracking_stats is not None
        assert item.tracking_stats["frames_tracked"] == 100

    def test_candidate_item_tracking_optional(self):
        """Tracking fields are optional in CandidateItem."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=0, y1=0, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(frame=frame, det_full=det)
        
        assert item.track_id is None
        assert item.tracking_stats is None
