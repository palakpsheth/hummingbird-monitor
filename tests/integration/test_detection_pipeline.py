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
import os
from pathlib import Path
from typing import Any

import pytest
import numpy as np

from hbmon.config import Settings
from hbmon.worker import (
    Det,
    DEFAULT_BIRD_CLASS_ID,
    _apply_roi,
    _compute_motion_mask,
    _detection_overlaps_motion,
    _pick_best_bird_det,
    _sanitize_bg_params,
)
from hbmon.yolo_utils import resolve_predict_imgsz

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

        model_path = Path("yolo11n.pt")
        try:
            _yolo_model = YOLO(str(model_path))
        except RuntimeError as e:
            message = str(e).lower()
            if "pytorchstreamreader" in message or "corrupt" in message or "invalid header" in message:
                if model_path.exists():
                    model_path.unlink()
                _yolo_model = YOLO(str(model_path))
            else:
                raise
    return _yolo_model


def _resolve_bird_class_id(yolo) -> int:
    bird_class_id = None
    try:
        names = getattr(yolo, "names", None)
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).strip().lower() == "bird":
                    bird_class_id = int(k)
                    break
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).strip().lower() == "bird":
                    bird_class_id = int(i)
                    break
    except Exception:
        bird_class_id = None

    if bird_class_id is None:
        bird_class_id = int(os.getenv("HBMON_BIRD_CLASS_ID", str(DEFAULT_BIRD_CLASS_ID)))
    return bird_class_id


def _load_background_image(test_dir: Path) -> "np.ndarray | None":
    for name in ("background.jpg", "background.jpeg", "background.png"):
        candidate = test_dir / name
        if candidate.exists():
            try:
                import cv2
            except ImportError:
                return None
            return cv2.imread(str(candidate))
    return None


def _apply_sensitivity_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    for key in (
        "detect_conf",
        "detect_iou",
        "min_box_area",
        "cooldown_seconds",
        "min_species_prob",
        "match_threshold",
        "ema_alpha",
        "crop_padding",
        "bg_subtraction_enabled",
        "bg_motion_threshold",
        "bg_motion_blur",
        "bg_min_overlap",
    ):
        if key in overrides:
            setattr(settings, key, overrides[key])
    return settings


def _build_sensitivity_settings(metadata: dict[str, Any], params: dict[str, Any]) -> Settings:
    settings = Settings()
    original_observation = metadata.get("original_observation") or {}
    extra = original_observation.get("extra") or {}
    baseline = extra.get("sensitivity") or {}
    settings = _apply_sensitivity_overrides(settings, baseline)
    settings = _apply_sensitivity_overrides(settings, params)
    return settings


def _count_filtered_boxes(
    results: Any,
    *,
    min_box_area: int,
    bird_class_id: int,
    motion_mask: "np.ndarray | None",
    min_motion_overlap: float,
) -> int:
    if not results:
        return 0
    r0 = results[0]
    if r0.boxes is None:
        return 0
    boxes = r0.boxes
    count = 0
    for b in boxes:
        try:
            cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            if cls != bird_class_id:
                continue
            xyxy = b.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
            det = Det(x1=x1, y1=y1, x2=x2, y2=y2, conf=0.0)
            if det.area < min_box_area:
                continue
            if motion_mask is not None:
                if not _detection_overlaps_motion(det, motion_mask, min_overlap_ratio=min_motion_overlap):
                    continue
            count += 1
        except Exception:
            continue
    return count


def _run_detection_pipeline(
    *,
    img,
    settings: Settings,
    yolo,
    bird_class_id: int,
    background_img,
) -> dict[str, Any]:
    roi_frame, _ = _apply_roi(img, settings)
    bg_enabled, bg_motion_threshold, bg_motion_blur, bg_min_overlap = _sanitize_bg_params(
        enabled=bool(settings.bg_subtraction_enabled),
        threshold=int(settings.bg_motion_threshold),
        blur=int(settings.bg_motion_blur),
        min_overlap=float(settings.bg_min_overlap),
    )
    bg_active = bool(bg_enabled and background_img is not None)
    motion_mask = None
    if bg_active:
        try:
            bg_roi, _ = _apply_roi(background_img, settings)
            motion_mask = _compute_motion_mask(
                roi_frame,
                bg_roi,
                threshold=bg_motion_threshold,
                blur_size=bg_motion_blur,
            )
        except Exception:
            motion_mask = None

    imgsz_env = os.getenv("HBMON_YOLO_IMGSZ", "1088,1920").strip()
    imgsz = resolve_predict_imgsz(imgsz_env, roi_frame.shape)
    
    results = yolo.predict(
        roi_frame,
        conf=float(settings.detect_conf),
        iou=float(settings.detect_iou),
        classes=[bird_class_id],
        imgsz=imgsz,
        verbose=False,
    )

    det = _pick_best_bird_det(
        results,
        int(settings.min_box_area),
        bird_class_id,
        motion_mask=motion_mask,
        min_motion_overlap=bg_min_overlap,
    )

    raw_box_count = 0
    max_conf = None
    if results:
        r0 = results[0]
        if r0.boxes is not None:
            raw_box_count = len(r0.boxes)
            try:
                max_conf = float(r0.boxes.conf.max().item())
            except Exception:
                max_conf = None

    filtered_box_count = _count_filtered_boxes(
        results,
        min_box_area=int(settings.min_box_area),
        bird_class_id=bird_class_id,
        motion_mask=motion_mask,
        min_motion_overlap=bg_min_overlap,
    )

    return {
        "detected": det is not None,
        "raw_box_count": raw_box_count,
        "filtered_box_count": filtered_box_count,
        "max_conf": max_conf,
        "bg_active": bg_active,
        "bg_motion_threshold": bg_motion_threshold,
        "bg_motion_blur": bg_motion_blur,
        "bg_min_overlap": bg_min_overlap,
        "imgsz": imgsz,
    }


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

        bird_class_id = _resolve_bird_class_id(yolo)

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
            params = default_test.get("params", {}) if default_test else {}
            settings = _build_sensitivity_settings(metadata, params)
            background_img = _load_background_image(test_dir)
            details = _run_detection_pipeline(
                img=img,
                settings=settings,
                yolo=yolo,
                bird_class_id=bird_class_id,
                background_img=background_img,
            )
            detected = details["detected"]

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

            # Skip cases where the label was not accepted during review.
            if expected.get("species_accepted") is False:
                continue

            # Read the image
            img = cv2.imread(str(snapshot_path))
            if img is None:
                pytest.fail(f"Failed to read snapshot: {snapshot_path}")

            # Run classification
            label, prob = clip_model.predict_species_label_prob(img)

            expected_label = expected.get("species_label_final") or expected.get("species_label")
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

        bird_class_id = _resolve_bird_class_id(yolo)

        for test_dir, metadata in test_cases:
            snapshot_path = test_dir / "snapshot.jpg"
            sensitivity_tests = metadata.get("sensitivity_tests", [])

            if not sensitivity_tests:
                continue

            img = cv2.imread(str(snapshot_path))
            if img is None:
                continue

            background_img = _load_background_image(test_dir)

            for sens_test in sensitivity_tests:
                params = sens_test.get("params", {})
                expected_detection = sens_test.get("expected_detection")
                settings = _build_sensitivity_settings(metadata, params)

                # Skip tests where expected outcome is not defined
                if expected_detection is None:
                    continue

                details = _run_detection_pipeline(
                    img=img,
                    settings=settings,
                    yolo=yolo,
                    bird_class_id=bird_class_id,
                    background_img=background_img,
                )
                detected = details["detected"]

                sensitivity_summary = {
                    "detect_conf": settings.detect_conf,
                    "detect_iou": settings.detect_iou,
                    "min_box_area": settings.min_box_area,
                    "cooldown_seconds": settings.cooldown_seconds,
                    "min_species_prob": settings.min_species_prob,
                    "match_threshold": settings.match_threshold,
                    "ema_alpha": settings.ema_alpha,
                    "crop_padding": settings.crop_padding,
                    "bg_subtraction_enabled": settings.bg_subtraction_enabled,
                    "bg_motion_threshold": settings.bg_motion_threshold,
                    "bg_motion_blur": settings.bg_motion_blur,
                    "bg_min_overlap": settings.bg_min_overlap,
                }

                assert detected == expected_detection, (
                    f"Sensitivity test '{sens_test.get('name')}' failed for {test_dir.name}: "
                    f"expected detection={expected_detection}, got {detected}. "
                    f"sensitivity={sensitivity_summary}, observed_boxes={details['raw_box_count']}, "
                    f"filtered_boxes={details['filtered_box_count']}, max_conf={details['max_conf']}, "
                    f"bg_active={details['bg_active']}, imgsz={details['imgsz']}"
                )


class TestMetadataSchema:
    """Integration tests for expected metadata schemas in test cases."""

    def test_identification_metadata_schema(self, integration_test_data_dir: Path):
        """
        Validate identification metadata shape when present in metadata.json.
        """
        test_cases = load_test_cases(integration_test_data_dir)

        if not test_cases:
            pytest.skip("No test data with snapshots available.")

        required_keys = {
            "individual_id",
            "match_score",
            "species_label",
            "species_prob",
            "species_label_final",
            "species_accepted",
        }

        for test_dir, metadata in test_cases:
            # Prefer the newer nested structure used by the coverage branch…
            original_observation = metadata.get("original_observation") or {}
            extra = original_observation.get("extra") or {}
            identification = extra.get("identification")

            # …but fall back to the older top-level key if present.
            if identification is None:
                identification = metadata.get("identification")

            if identification is None:
                continue

            assert isinstance(identification, dict), (
                f"Identification metadata must be a dict in {test_dir.name}, "
                f"got {type(identification).__name__}"
            )

            missing = required_keys - set(identification.keys())
            assert not missing, (
                f"Identification metadata missing keys {sorted(missing)} in {test_dir.name}"
            )

    def test_background_subtraction_metadata_schema(self, integration_test_data_dir: Path):
        """
        Validate background subtraction metadata shape when present in metadata.json.
        """
        test_cases = load_test_cases(integration_test_data_dir)

        if not test_cases:
            pytest.skip("No test data with snapshots available.")

        required_keys = {
            "bg_motion_threshold",
            "bg_motion_blur",
            "bg_min_overlap",
            "bg_subtraction_enabled",
        }

        for test_dir, metadata in test_cases:
            original_observation = metadata.get("original_observation") or {}
            extra = original_observation.get("extra") or {}
            sensitivity = extra.get("sensitivity")

            if sensitivity is None:
                continue

            assert isinstance(sensitivity, dict), (
                f"Sensitivity metadata must be a dict in {test_dir.name}, "
                f"got {type(sensitivity).__name__}"
            )

            missing = required_keys - set(sensitivity.keys())
            assert not missing, (
                f"Sensitivity metadata missing keys {sorted(missing)} in {test_dir.name}"
            )

            assert isinstance(sensitivity["bg_subtraction_enabled"], bool), (
                f"bg_subtraction_enabled must be boolean in {test_dir.name}"
            )
            assert isinstance(sensitivity["bg_motion_threshold"], int), (
                f"bg_motion_threshold must be int in {test_dir.name}"
            )
            assert isinstance(sensitivity["bg_motion_blur"], int), (
                f"bg_motion_blur must be int in {test_dir.name}"
            )
            assert isinstance(sensitivity["bg_min_overlap"], (float, int)), (
                f"bg_min_overlap must be float in {test_dir.name}"
            )
