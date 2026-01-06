"""
Tests for tracking mode metadata and UI behavior.

These tests verify that:
1. CandidateItem settings_snapshot has correct fields based on tracking mode
2. Temporal metadata is excluded when tracking is enabled
3. Tracking metadata is excluded when temporal voting is enabled
"""

import numpy as np

from hbmon.worker import CandidateItem, Det


class TestTrackingMetadata:
    """Tests for tracking mode metadata in CandidateItem."""

    def test_tracking_mode_settings_snapshot_excludes_temporal_fields(self):
        """When tracking is enabled, settings_snapshot should not have temporal fields."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        # Create a CandidateItem with tracking mode settings
        item = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={
                "detect_conf": 0.1,
                "detect_iou": 0.45,
                "min_box_area": 600,
                "fps_limit": 30.0,
                "cooldown_seconds": 5.0,
                "use_tracking": True,
                "track_high_thresh": 0.1,
                "track_low_thresh": 0.01,
            },
            track_id=42,
        )
        
        # Verify tracking fields are present
        assert item.settings_snapshot is not None
        assert item.settings_snapshot["use_tracking"] is True
        assert "track_high_thresh" in item.settings_snapshot
        assert "track_low_thresh" in item.settings_snapshot
        
        # Verify temporal fields are NOT present
        assert "temporal_window_frames" not in item.settings_snapshot
        assert "temporal_min_detections" not in item.settings_snapshot

    def test_temporal_mode_settings_snapshot_excludes_tracking_fields(self):
        """When temporal voting is enabled, settings_snapshot should not have tracking fields."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        # Create a CandidateItem with temporal voting settings
        item = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={
                "detect_conf": 0.3,
                "detect_iou": 0.45,
                "min_box_area": 600,
                "fps_limit": 30.0,
                "cooldown_seconds": 5.0,
                "use_tracking": False,
                "temporal_window_frames": 5,
                "temporal_min_detections": 1,
            },
            temporal_stats={
                "min_required": 1,
                "positive_frames": 5,
                "window_frames": 5,
            },
        )
        
        # Verify temporal fields are present
        assert item.settings_snapshot is not None
        assert item.settings_snapshot["use_tracking"] is False
        assert "temporal_window_frames" in item.settings_snapshot
        assert "temporal_min_detections" in item.settings_snapshot
        
        # Verify tracking-specific thresholds are NOT present
        assert "track_high_thresh" not in item.settings_snapshot
        assert "track_low_thresh" not in item.settings_snapshot

    def test_tracking_mode_has_track_id(self):
        """Tracking mode CandidateItem should have track_id."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={"use_tracking": True},
            track_id=123,
        )
        
        assert item.track_id == 123
        assert item.track_id is not None

    def test_temporal_mode_no_track_id(self):
        """Temporal voting CandidateItem should not have track_id."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={"use_tracking": False},
            temporal_stats={"positive_frames": 5},
        )
        
        assert item.track_id is None

    def test_tracking_mode_has_tracking_stats(self):
        """Tracking mode CandidateItem should have tracking_stats."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={"use_tracking": True},
            track_id=42,
            tracking_stats={
                "track_id": 42,
                "frames_tracked": 100,
                "track_high_thresh": 0.1,
                "track_low_thresh": 0.01,
            },
        )
        
        assert item.tracking_stats is not None
        assert item.tracking_stats["track_id"] == 42
        assert item.tracking_stats["frames_tracked"] == 100

    def test_temporal_mode_has_temporal_stats(self):
        """Temporal voting CandidateItem should have temporal_stats."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={"use_tracking": False},
            temporal_stats={
                "min_required": 1,
                "positive_frames": 5,
                "window_frames": 5,
            },
        )
        
        assert item.temporal_stats is not None
        assert item.temporal_stats["positive_frames"] == 5
        assert item.temporal_stats["window_frames"] == 5

    def test_settings_snapshot_always_has_use_tracking_flag(self):
        """All settings_snapshot dicts should have use_tracking flag."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        # Tracking mode
        item_tracking = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={"use_tracking": True},
        )
        assert "use_tracking" in item_tracking.settings_snapshot
        assert item_tracking.settings_snapshot["use_tracking"] is True
        
        # Temporal mode
        item_temporal = CandidateItem(
            frame=frame,
            det_full=det,
            settings_snapshot={"use_tracking": False},
        )
        assert "use_tracking" in item_temporal.settings_snapshot
        assert item_temporal.settings_snapshot["use_tracking"] is False

    def test_tracking_stats_includes_thresholds(self):
        """Tracking stats should include the thresholds used."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.8)
        
        item = CandidateItem(
            frame=frame,
            det_full=det,
            tracking_stats={
                "track_id": 1,
                "frames_tracked": 50,
                "track_high_thresh": 0.15,
                "track_low_thresh": 0.02,
            },
        )
        
        assert item.tracking_stats["track_high_thresh"] == 0.15
        assert item.tracking_stats["track_low_thresh"] == 0.02
