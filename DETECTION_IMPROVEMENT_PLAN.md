# Detection Improvement Plan

> **Goal**: Make hummingbird detection more robust to reduce missed bird visits.
>
> **Generated**: 2025-12-30
>
> **Hardware**: Intel Core i7-13700H (14 cores, 20 threads), 32GB RAM, Intel Iris Xe GPU

---

## Hardware Profile

| Component | Specification | Implications |
|-----------|---------------|--------------|
| **CPU** | Intel Core i7-13700H (14 cores, 20 threads, 5.0 GHz) | Excellent multi-threaded performance; can handle higher FPS |
| **RAM** | 32GB (19GB available) | Plenty of headroom for buffers and caching |
| **GPU** | Intel Iris Xe (integrated) | Supports OpenVINO acceleration - **untapped potential** |
| **Current Load** | ~60% CPU, load avg 6.49 | Room for optimization; some headroom available |
| **Disk** | 98GB available | Sufficient for clips and debug frames |
| **Inference** | CPU-only (PyTorch) | Can be accelerated with OpenVINO on Intel GPU |

---

## Root Causes for Missed Detections

Bird visits can be missed at these filtering stages:

1. **YOLO Detection Stage** - Bird not detected or confidence too low
2. **Minimum Box Area Filter** - Detection filtered as too small (`min_box_area=600`)
3. **Motion Mask Overlap Filter** - Detection rejected due to insufficient motion overlap (`bg_min_overlap=0.15`)
4. **Cooldown Filter** - Detection during cooldown period (`cooldown_seconds=2-4`)
5. **Frame Sampling** - Bird present between sampled frames (`fps_limit`)

---

## Phase 1: Immediate Tuning (No Code Changes)

These adjustments leverage your hardware capabilities:

### 1.1 Increase Frame Rate (Your CPU Can Handle It)
```bash
HBMON_FPS_LIMIT=25  # Increase from 8-20
```
- **Why**: i7-13700H has plenty of headroom at 60% load
- **Impact**: More frames = higher chance of catching fast-moving hummingbirds
- **Your hardware**: 14 cores can easily handle 25+ FPS YOLO inference

### 1.2 Lower Detection Confidence Threshold
```bash
HBMON_DETECT_CONF=0.15  # Currently 0.25-0.30
```
- **Why**: Catch more detections; motion filtering will remove false positives
- **Trade-off**: Minimal with motion filtering enabled

### 1.3 Reduce Minimum Box Area
```bash
HBMON_MIN_BOX_AREA=300  # Currently 600
```
- **Why**: Hummingbirds at distance or in flight appear smaller
- **Trade-off**: Motion filtering handles false positives

### 1.4 Lower Motion Overlap Requirement
```bash
HBMON_BG_MIN_OVERLAP=0.08  # Currently 0.15
```
- **Why**: Fast-moving birds may have less overlap with motion mask

### 1.5 Reduce Motion Detection Threshold
```bash
HBMON_BG_MOTION_THRESHOLD=20  # Currently 30
```
- **Why**: More sensitive to subtle motion from small birds

### 1.6 Reduce Cooldown Period
```bash
HBMON_COOLDOWN_SECONDS=1.0  # Currently 2-4
```
- **Why**: Your system can handle more frequent observations
- **Impact**: Captures rapid successive visits

### 1.7 Enable Debug Logging
```bash
HBMON_DEBUG_YOLO=1
HBMON_BG_LOG_REJECTED=1
HBMON_BG_REJECTED_SAVE_CLIP=1
HBMON_DEBUG_SAVE_FRAMES=1
```

---

## Phase 2: Hardware Acceleration (High Impact, Medium Effort)

### 2.1 Enable OpenVINO for Intel GPU Acceleration ⭐ **RECOMMENDED**

Your Intel Iris Xe GPU is currently **unused**. OpenVINO can accelerate YOLO inference 2-4x.

**Implementation Steps**:

1. **Export YOLO to OpenVINO format**:
```bash
# In the container or locally
pip install openvino ultralytics
yolo export model=yolo11n.pt format=openvino
```

2. **Update environment**:
```bash
HBMON_YOLO_MODEL=yolo11n_openvino_model/
HBMON_DEVICE=GPU  # Use Intel GPU via OpenVINO
```

3. **Update Dockerfile** to include OpenVINO:
```dockerfile
RUN pip install openvino openvino-dev
```

**Expected Impact**:
- 2-4x faster inference on Intel Iris Xe
- Can increase `HBMON_FPS_LIMIT` to 30-40 FPS
- Lower CPU usage (offload to GPU)

### 2.2 Use Larger YOLO Model (Your Hardware Can Handle It)

With OpenVINO acceleration, you can use a more accurate model:

```bash
HBMON_YOLO_MODEL=yolo11s.pt  # Small instead of Nano
# Or with OpenVINO:
HBMON_YOLO_MODEL=yolo11s_openvino_model/
```

**Impact**: Better detection accuracy, especially for small/distant birds

---

## Phase 3: Code Improvements (Medium Effort)

### 3.1 Temporal Smoothing / Detection Persistence
**File**: `src/hbmon/worker.py`

```python
# Track recent detections and trigger if N detections in M frames
recent_detections = deque(maxlen=5)  # Last 5 frames
recent_detections.append(len(detections) > 0)
if sum(recent_detections) >= 2:  # 2+ detections in 5 frames
    # Trigger observation
```

**Why**: At 25 FPS, a bird is visible for many frames; requiring 2/5 frames reduces false positives while catching real birds.

### 3.2 Confidence-Weighted Motion Overlap
**File**: `src/hbmon/worker.py` (lines 1240-1245)

```python
# High-confidence detections need less motion overlap
effective_overlap = bg_min_overlap * (1.0 - det.conf * 0.5)
if float(stats["bbox_overlap_ratio"]) >= effective_overlap:
    kept_entries.append((candidate_det, stats))
```

### 3.3 Adaptive Motion Mask Morphology
**File**: `src/hbmon/worker.py` (lines 315-318)

```python
# Smaller kernel to preserve small motion areas
kernel_size = env_int("HBMON_BG_MORPH_KERNEL", 3)  # Default 3 instead of 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
```

### 3.4 Pre-Trigger Ring Buffer (Leverage Your RAM)
**File**: `src/hbmon/worker.py`

With 19GB available RAM, you can maintain a generous frame buffer:

```python
# 3 seconds of pre-trigger buffer at 25 FPS = 75 frames
# At ~2MB per 1080p frame = ~150MB buffer (trivial for your system)
PRE_TRIGGER_SECONDS = 3.0
frame_buffer = deque(maxlen=int(fps_limit * PRE_TRIGGER_SECONDS))
```

**Impact**: Captures bird arrival, not just post-detection

---

## Phase 4: Advanced Improvements (Higher Effort)

### 4.1 Fine-Tuned YOLO Model
**Priority**: Highest impact for detection accuracy

1. Export labeled observations from your system
2. Fine-tune YOLOv11 on hummingbird-specific data
3. Deploy: `HBMON_YOLO_MODEL=hummingbird_yolo.pt`

### 4.2 Multi-Worker Detection (Leverage All Cores)

Your i7-13700H has 14 cores but the worker is single-threaded for detection:

```python
# Use ThreadPoolExecutor for parallel frame processing
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)
# Process multiple frames in parallel
```

### 4.3 Object Tracking (ByteTrack)

With higher FPS, tracking becomes more effective:
- Track birds across frames
- End visit only after N seconds of no tracking
- Reduces duplicate observations

---

## Recommended Implementation Order (Hardware-Optimized)

| Priority | Task | Effort | Impact | Hardware Benefit |
|----------|------|--------|--------|------------------|
| **1** | Phase 1 tuning (increase FPS to 25) | Low | High | Uses available CPU headroom |
| **2** | Enable OpenVINO on Intel GPU | Medium | Very High | 2-4x faster inference |
| **3** | Use YOLOv11s model | Low | Medium | Better accuracy with OpenVINO |
| **4** | Temporal smoothing | Medium | High | Works better at higher FPS |
| **5** | Pre-trigger buffer (3 sec) | Medium | High | Uses available RAM |
| **6** | Fine-tuned YOLO model | High | Very High | Best detection accuracy |

---

## Quick Start: Recommended `.env` Changes for Your Hardware

```bash
# === DETECTION (optimized for i7-13700H) ===
# Higher FPS - your CPU can handle it
HBMON_FPS_LIMIT=25

# More sensitive detection (motion filtering handles false positives)
HBMON_DETECT_CONF=0.15
HBMON_MIN_BOX_AREA=300

# === MOTION FILTERING (more sensitive) ===
HBMON_BG_MOTION_THRESHOLD=20
HBMON_BG_MIN_OVERLAP=0.08

# === EVENT FREQUENCY (faster system can handle more) ===
HBMON_COOLDOWN_SECONDS=1.0

# === DEBUGGING (understand what's being missed) ===
HBMON_DEBUG_YOLO=1
HBMON_BG_LOG_REJECTED=1
HBMON_BG_REJECTED_SAVE_CLIP=1
HBMON_DEBUG_SAVE_FRAMES=1
HBMON_DEBUG_EVERY_SECONDS=30

# === FUTURE: OpenVINO acceleration ===
# After setting up OpenVINO:
# HBMON_YOLO_MODEL=yolo11n_openvino_model/
# HBMON_DEVICE=GPU
```

---

## Monitoring After Changes

1. **Check CPU usage**: Should stay under 80% with new settings
   ```bash
   docker stats hbmon-worker
   ```

2. **Review candidates**: Visit `/candidates` to see rejected detections
   
3. **Check detection rate**: Compare observations before/after tuning

4. **Watch for false positives**: If too many, slightly increase `HBMON_DETECT_CONF` or `HBMON_BG_MIN_OVERLAP`

---

## OpenVINO Setup Guide (For Phase 2)

```bash
# 1. Install OpenVINO in your environment
pip install openvino ultralytics

# 2. Export YOLO model to OpenVINO format
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='openvino')"

# 3. Copy the exported model to your data directory
cp -r yolo11n_openvino_model/ /media/palak/hbmon/hummingbird-monitor/data/

# 4. Update .env
HBMON_YOLO_MODEL=/data/yolo11n_openvino_model/

# 5. Restart worker
docker compose restart hbmon-worker
```

This will offload YOLO inference to your Intel Iris Xe GPU, freeing CPU for higher frame rates and other processing.
