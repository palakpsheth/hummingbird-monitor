"""
Integration tests for hbmon detection and classification pipeline.

These tests require ML dependencies (PyTorch, YOLO, CLIP) and real test data.
They are marked with @pytest.mark.integration and are skipped by default.

To run these tests:
    uv run pytest -m integration

Test data should be placed in tests/integration/test_data/ with the structure
documented in tests/integration/test_data/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def load_test_cases(test_data_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """
    Load all test cases from the test data directory.

    Returns a list of (test_case_dir, metadata) tuples.
    """
    test_cases = []

    if not test_data_dir.exists():
        return test_cases

    for item in test_data_dir.iterdir():
        if item.is_dir():
            metadata_file = item / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    # Only include cases that have actual test data (snapshot.jpg)
                    snapshot = item / "snapshot.jpg"
                    if snapshot.exists():
                        test_cases.append((item, metadata))
                except (json.JSONDecodeError, OSError):
                    pass

            # Check for nested directories (e.g., edge_cases/)
            for subitem in item.iterdir():
                if subitem.is_dir():
                    sub_metadata_file = subitem / "metadata.json"
                    if sub_metadata_file.exists():
                        try:
                            with open(sub_metadata_file) as f:
                                sub_metadata = json.load(f)
                            snapshot = subitem / "snapshot.jpg"
                            if snapshot.exists():
                                test_cases.append((subitem, sub_metadata))
                        except (json.JSONDecodeError, OSError):
                            pass

    return test_cases


@pytest.fixture
def integration_test_data_dir() -> Path:
    """Return the path to the integration test data directory."""
    return Path(__file__).parent / "test_data"


# Module-level cached YOLO model to avoid reloading for each test
_yolo_model = None


def get_yolo_model():
    """Get or load the YOLO model (cached at module level)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolo11n.pt")
    return _yolo_model


class TestDetectionPipeline:
    """Integration tests for the YOLO detection pipeline."""

    def test_detection_with_test_data(self, integration_test_data_dir: Path):
        """
        Test detection on all available test cases.

        This test is skipped if no test data with snapshots is available.
        """
        test_cases = load_test_cases(integration_test_data_dir)

        if not test_cases:
            pytest.skip("No test data with snapshots available. Add test data to run integration tests.")

        # Import dependencies only when running integration tests
        try:
            import cv2
        except ImportError as e:
            pytest.skip(f"ML dependencies not available: {e}")

        try:
            yolo = get_yolo_model()
        except ImportError as e:
            pytest.skip(f"YOLO not available: {e}")

        for test_dir, metadata in test_cases:
            snapshot_path = test_dir / "snapshot.jpg"
            expected = metadata.get("expected", {})

            # Read the image
            img = cv2.imread(str(snapshot_path))
            if img is None:
                pytest.fail(f"Failed to read snapshot: {snapshot_path}")

            # Run detection with default sensitivity
            sensitivity_tests = metadata.get("sensitivity_tests", [])
            default_test = next((t for t in sensitivity_tests if t.get("name") == "default"), None)

            if default_test:
                params = default_test.get("params", {})
                detect_conf = params.get("detect_conf", 0.35)
                detect_iou = params.get("detect_iou", 0.45)
            else:
                detect_conf = 0.35
                detect_iou = 0.45

            # Run YOLO detection
            results = yolo.predict(
                img,
                conf=detect_conf,
                iou=detect_iou,
                classes=[14],  # bird class
                verbose=False,
            )

            # Check detection result
            r0 = results[0]
            detected = r0.boxes is not None and len(r0.boxes) > 0

            # Only validate if human_verified is True and expected detection is set
            if expected.get("human_verified") and expected.get("detection") is not None:
                expected_detection = expected.get("detection")
                assert detected == expected_detection, (
                    f"Detection mismatch for {test_dir.name}: "
                    f"expected {expected_detection}, got {detected}"
                )


class TestClassificationPipeline:
    """Integration tests for the CLIP classification pipeline."""

    def test_classification_with_test_data(self, integration_test_data_dir: Path):
        """
        Test species classification on all available test cases.

        This test validates that human-verified labels match the classification output.
        """
        test_cases = load_test_cases(integration_test_data_dir)

        if not test_cases:
            pytest.skip("No test data with snapshots available.")

        # Import dependencies
        try:
            import cv2
            from hbmon.clip_model import ClipModel
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

        # Load CLIP model
        try:
            clip_model = ClipModel(device="cpu")
        except RuntimeError as e:
            pytest.skip(f"Failed to load CLIP model: {e}")

        for test_dir, metadata in test_cases:
            snapshot_path = test_dir / "snapshot.jpg"
            expected = metadata.get("expected", {})

            # Only test if human_verified and species_label is set
            if not expected.get("human_verified") or not expected.get("species_label"):
                continue

            # Read the image
            img = cv2.imread(str(snapshot_path))
            if img is None:
                pytest.fail(f"Failed to read snapshot: {snapshot_path}")

            # Run classification
            label, prob = clip_model.predict_species_label_prob(img)

            expected_label = expected.get("species_label")
            assert label == expected_label, (
                f"Classification mismatch for {test_dir.name}: "
                f"expected {expected_label}, got {label} (prob={prob:.2f})"
            )


class TestSensitivityParameters:
    """Integration tests for detection sensitivity parameter variations."""

    def test_sensitivity_variations(self, integration_test_data_dir: Path):
        """
        Test that sensitivity parameters affect detection as expected.

        For each test case, runs detection with different parameter sets
        and validates the expected outcomes.
        """
        test_cases = load_test_cases(integration_test_data_dir)

        if not test_cases:
            pytest.skip("No test data available.")

        try:
            import cv2
        except ImportError as e:
            pytest.skip(f"ML dependencies not available: {e}")

        try:
            yolo = get_yolo_model()
        except ImportError as e:
            pytest.skip(f"YOLO not available: {e}")

        for test_dir, metadata in test_cases:
            snapshot_path = test_dir / "snapshot.jpg"
            sensitivity_tests = metadata.get("sensitivity_tests", [])

            if not sensitivity_tests:
                continue

            img = cv2.imread(str(snapshot_path))
            if img is None:
                continue

            for sens_test in sensitivity_tests:
                params = sens_test.get("params", {})
                expected_detection = sens_test.get("expected_detection")

                # Skip tests where expected outcome is not defined
                if expected_detection is None:
                    continue

                results = yolo.predict(
                    img,
                    conf=params.get("detect_conf", 0.35),
                    iou=params.get("detect_iou", 0.45),
                    classes=[14],
                    verbose=False,
                )

                r0 = results[0]
                detected = r0.boxes is not None and len(r0.boxes) > 0

                assert detected == expected_detection, (
                    f"Sensitivity test '{sens_test.get('name')}' failed for {test_dir.name}: "
                    f"expected detection={expected_detection}, got {detected}"
                )
