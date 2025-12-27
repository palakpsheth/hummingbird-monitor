# Integration Test Data

This directory contains test cases for integration tests. Each test case folder
represents a specific scenario with real or simulated observation data.

## Directory Structure

```
test_data/
├── README.md                    # This file
├── flying_0/                    # Test case: flying hummingbird #0
│   ├── snapshot.jpg             # Snapshot image as saved by the worker
│   ├── clip.mp4                 # Video clip as saved by the worker
│   └── metadata.json            # Observation metadata
├── flying_1/                    # Test case: flying hummingbird #1
│   └── ...
├── perched_0/                   # Test case: perched hummingbird #0
│   └── ...
├── perched_1/                   # Test case: perched hummingbird #1
│   └── ...
├── feeding_0/                   # Test case: feeding at feeder #0
│   └── ...
├── multiple_birds_0/            # Test case: multiple birds in frame #0
│   └── ...
├── false_positive_0/            # Test case: false positive detection #0
│   └── ...
└── edge_cases/                  # Edge case scenarios
    ├── low_light_0/             # Low light conditions
    ├── motion_blur_0/           # Motion blur
    └── partial_occlusion_0/     # Partially occluded bird
```

## Test Case Metadata Format

Each test case folder contains a `metadata.json` file with the following structure:

```json
{
    "description": "Short description of the test case",
    "expected": {
        "detection": true,
        "species_label": "Anna's Hummingbird",
        "species_label_final": "Anna's Hummingbird",
        "species_accepted": true,
        "behavior": "flying",
        "human_verified": true
    },
    "source": {
        "camera": "hummingbirdcam",
        "timestamp_utc": "2024-01-15T14:30:00Z",
        "location": "Southern California"
    },
    "sensitivity_tests": [
        {
            "name": "default",
            "params": {
                "detect_conf": 0.35,
                "detect_iou": 0.45,
                "min_box_area": 600
            },
            "expected_detection": true
        },
        {
            "name": "high_confidence",
            "params": {
                "detect_conf": 0.75,
                "detect_iou": 0.45,
                "min_box_area": 600
            },
            "expected_detection": true
        },
        {
            "name": "strict",
            "params": {
                "detect_conf": 0.90,
                "detect_iou": 0.45,
                "min_box_area": 1000
            },
            "expected_detection": false
        }
    ],
    "original_observation": {
        "species_label": "Anna's Hummingbird",
        "species_prob": 0.87,
        "bbox_xyxy": [120, 80, 320, 280],
        "match_score": 0.92,
        "extra": {
            "sensitivity": {
                "detect_conf": 0.35,
                "detect_iou": 0.45,
                "min_box_area": 600,
                "bg_motion_threshold": 30,
                "bg_motion_blur": 5,
                "bg_min_overlap": 0.15,
                "bg_subtraction_enabled": true
            },
            "detection": {
                "box_confidence": 0.78,
                "bbox_xyxy": [120, 80, 320, 280],
                "bbox_area": 40000,
                "bbox_area_ratio_frame": 0.0833,
                "bbox_area_ratio_roi": 0.0912,
                "nms_iou_threshold": 0.45,
                "background_subtraction_enabled": true
            },
            "identification": {
                "individual_id": 1,
                "match_score": 0.92,
                "species_label": "Anna's Hummingbird",
                "species_prob": 0.87,
                "species_label_final": "Anna's Hummingbird",
                "species_accepted": true
            },
            "review": {
                "label": "true_positive"
            }
        }
    }
}
```

## Field Descriptions

### `expected`
- `detection`: Whether a bird should be detected in this test case
- `species_label`: Expected species classification (if human verified)
- `species_label_final` (optional): Final species label after review
- `species_accepted` (optional): Whether the reviewed species label was accepted
- `behavior`: Category of behavior (flying, perched, feeding, etc.)
- `human_verified`: Whether a human has verified the ground truth

### `sensitivity_tests`
Array of detection sensitivity parameter combinations to test. Each entry includes:
- `name`: Human-readable test name
- `params`: Detection parameters to use
- `expected_detection`: Whether detection should succeed with these params

### `original_observation`
The raw observation data as captured by the worker, including:
- Species classification results
- Bounding box coordinates
- Match score for individual re-identification
- Extra metadata including sensitivity, detection, and identification settings at capture time
  - `sensitivity`: detection thresholds (confidence, IoU, min box area) plus background subtraction tuning
  - `detection`: detector confidence, bbox geometry, bbox area + frame/ROI ratios, and IoU threshold used for NMS
  - `identification`: individual match score, species match probability, and species labels used for review

## Adding New Test Cases

1. Create a new folder with a descriptive name (e.g., `flying_2`)
2. Add the `snapshot.jpg` image captured by the worker
3. Add the `clip.mp4` video if available
4. Create `metadata.json` following the schema above
5. Ensure `human_verified` is set to `true` if you've manually confirmed the labels

### Using the export bundle

The observation detail page can export a tar.gz bundle that already contains the
expected test layout:

- `metadata.json` with `expected`, `source`, `sensitivity_tests`, and `original_observation`
- `snapshot.jpg` (raw snapshot)
- `clip.mp4` (video clip)
- Optional `background.jpg` (reference background frame for motion filtering; included when
  a background image is configured on the server)
- Optional `snapshot_annotated.jpg` and `snapshot_clip.jpg` when available

Extract the bundle under `tests/integration/test_data/` to add a new case.

## Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and are skipped by default.

To run integration tests:
```bash
# Run only integration tests
uv run pytest -m integration

# Run all tests including integration
uv run pytest -m ""
```

## Notes

- Test cases with `human_verified: true` in the metadata will be used to validate
  that classification results match the expected labels
- Sensitivity tests allow testing the same image with different detection parameters
- Edge case folders help ensure the system handles difficult scenarios gracefully
