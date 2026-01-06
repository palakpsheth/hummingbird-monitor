# src/hbmon/annotation_detector.py
"""
High-accuracy detection for annotation preprocessing.

This module provides auto-detection capabilities using larger, more accurate models
than the real-time detection pipeline. It supports:
- YOLO Large/XLarge models for better detection accuracy
- SAM (Segment Anything Model) for precise bounding box refinement
- Batched frame processing for efficiency

The goal is to minimize manual annotation burden by providing high-quality
initial detections that only need minor corrections.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import numpy as np
import math

from hbmon.yolo_utils import resolve_predict_imgsz

try:
    from hbmon.openvino_utils import force_openvino_gpu_override
except ImportError:
    def force_openvino_gpu_override() -> None:
        pass


logger = logging.getLogger(__name__)

# Configuration from environment
ANNOTATION_YOLO_MODEL = os.environ.get("HBMON_ANNOTATION_YOLO_MODEL", "yolo11l.pt")
ANNOTATION_USE_SAM = os.environ.get("HBMON_ANNOTATION_USE_SAM", "1") == "1"
ANNOTATION_SAM_MODEL = os.environ.get("HBMON_ANNOTATION_SAM_MODEL", "sam_b")
ANNOTATION_CONFIDENCE = float(os.environ.get("HBMON_ANNOTATION_CONFIDENCE", "0.15"))
ANNOTATION_BATCH_SIZE = int(os.environ.get("HBMON_ANNOTATION_BATCH_SIZE", "8"))
ANNOTATION_NMS_THRESHOLD = float(os.environ.get("HBMON_ANNOTATION_NMS_THRESHOLD", "0.5"))

# SAHI Configuration
ANNOTATION_USE_SAHI = os.environ.get("HBMON_ANNOTATION_USE_SAHI", "1") == "1"
SAHI_SLICE_HEIGHT = int(os.environ.get("HBMON_SAHI_SLICE_HEIGHT", "640"))
SAHI_SLICE_WIDTH = int(os.environ.get("HBMON_SAHI_SLICE_WIDTH", "640"))
SAHI_OVERLAP_RATIO = float(os.environ.get("HBMON_SAHI_OVERLAP_RATIO", "0.2"))


@dataclass
class DetectedBox:
    """A detected bounding box with confidence."""
    class_id: int
    x: float  # center x (0-1 normalized)
    y: float  # center y (0-1 normalized)
    w: float  # width (0-1 normalized)
    h: float  # height (0-1 normalized)
    confidence: float
    source: str = "auto"  # auto, sam-refined

    def to_yolo_format(self) -> str:
        """Convert to YOLO label format."""
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


class AnnotationDetector:
    """High-accuracy detector for annotation preprocessing.
    
    Uses larger YOLO models and optional SAM refinement for better
    detection accuracy than real-time detection.
    """
    
    
    def __init__(
        self,
        yolo_model: str | None = None,
        use_sam: bool | None = None,
        sam_model: str | None = None,
        confidence: float | None = None,
        use_sahi: bool | None = None,
    ):
        """Initialize the annotation detector.
        
        Args:
            yolo_model: YOLO model path/name (default from env)
            use_sam: Whether to use SAM refinement (default from env)
            sam_model: SAM model variant (default from env)
            confidence: Detection confidence threshold (default from env)
            use_sahi: Whether to use SAHI sliced inference (default from env)
        """
        self.yolo_model_name = yolo_model or ANNOTATION_YOLO_MODEL
        self.use_sam = use_sam if use_sam is not None else ANNOTATION_USE_SAM
        self.sam_model_name = sam_model or ANNOTATION_SAM_MODEL
        self.confidence = confidence if confidence is not None else ANNOTATION_CONFIDENCE
        self.use_sahi = use_sahi if use_sahi is not None else ANNOTATION_USE_SAHI
        
        self._yolo = None
        self._sahi_model = None
        self._sam = None
        self._initialized = False
        self.bird_class_id: int | None = None

    def _resolve_bird_class_id(self) -> None:
        """Resolve the class ID for 'bird' (or 'hummingbird') from the model."""
        if self._yolo is None:
            return
            
        # Default COCO bird class is usually 14, but we check names dynamically
        names = getattr(self._yolo, 'names', {})
        target_id = None
        
        # Handle dict or list names
        if isinstance(names, dict):
            for k, v in names.items():
                name_str = str(v).strip().lower()
                if 'bird' in name_str or 'hummingbird' in name_str:
                    target_id = int(k)
                    break
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                name_str = str(v).strip().lower()
                if 'bird' in name_str or 'hummingbird' in name_str:
                    target_id = int(i)
                    break
                    
        if target_id is not None:
            self.bird_class_id = target_id
            logger.info(f"Resolved annotator bird_class_id={target_id} from model")
        else:
            logger.warning("Could not resolve 'bird' class ID from model names. Falling back to COCO default (14).")
            self.bird_class_id = 14
        
        # Override to 14 for COCO models to be safe
        # Override to 14 for COCO models to be safe, unless we specifically found 'hummingbird'
        if self.bird_class_id != 14:
            logger.info(f"Using resolved bird_class_id={self.bird_class_id}")
            # removed the forced override here to allow 'hummingbird' class if found

    def _load_yolo(self) -> Any:
        """Load YOLO model for annotation detection."""
        if self._yolo is not None:
            return self._yolo
            
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("ultralytics package not installed")
        
        logger.info(f"Loading annotation YOLO model: {self.yolo_model_name}")
        
        # Check for OpenVINO backend preference
        backend = os.environ.get("HBMON_INFERENCE_BACKEND", "cpu")
        
        if "openvino" in backend.lower():
            # Use OpenVINO for acceleration
            # Apply GPU override logic if applicable (same as valid worker shim)
            force_openvino_gpu_override()
            
            model_path = self._get_openvino_model_path()
            if model_path and model_path.exists():
                logger.info(f"Using OpenVINO model: {model_path}")
                self._yolo = YOLO(str(model_path), task="detect")
            else:
                logger.info(f"Loading PyTorch model and exporting to OpenVINO: {self.yolo_model_name}")
                # Try to load from YOLO_CONFIG_DIR first
                config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/data/yolo"))
                pt_path = config_dir / self.yolo_model_name
                
                load_path = str(pt_path) if pt_path.exists() else self.yolo_model_name
                self._yolo = YOLO(load_path, task="detect")
                
                # Export to OpenVINO format for future use
                try:
                    self._export_to_openvino()
                except Exception as e:
                    logger.warning(f"OpenVINO export failed, using PyTorch: {e}")
        else:
            # Try to load from YOLO_CONFIG_DIR first
            config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/data/yolo"))
            pt_path = config_dir / self.yolo_model_name
            load_path = str(pt_path) if pt_path.exists() else self.yolo_model_name
            
            self._yolo = YOLO(load_path, task="detect")
        
        return self._yolo

    def _get_openvino_model_path(self) -> Path | None:
        """Get path to OpenVINO-exported model if it exists."""
        # Use YOLO_CONFIG_DIR for persistent storage instead of cache dir which might be ephemeral
        cache_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/data/yolo"))
        model_name = Path(self.yolo_model_name).stem
        ov_path = cache_dir / "models" / f"{model_name}_openvino_model" / f"{model_name}.xml"
        return ov_path if ov_path.exists() else None

    def _export_to_openvino(self) -> Path | None:
        """Export YOLO model to OpenVINO format."""
        if self._yolo is None:
            return None
            
        # Use YOLO_CONFIG_DIR for persistent storage
        cache_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/data/yolo"))
        export_dir = cache_dir / "models"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Exporting annotation YOLO model to OpenVINO format...")
            self._yolo.export(format="openvino", half=False)
            logger.info("OpenVINO export complete")
        except Exception as e:
            logger.warning(f"OpenVINO export failed: {e}")
        
        return self._get_openvino_model_path()

    def _load_sahi_model(self) -> Any:
        """Load AutoDetectionModel for SAHI."""
        if self._sahi_model is not None:
            return self._sahi_model

        if not self.use_sahi:
            return None
            
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            logger.warning("sahi package not installed, skipping SAHI")
            self.use_sahi = False
            return None

        # Ensure we have the base YOLO model loaded (to ensure download/export logic runs)
        self._load_yolo()
        
        logger.info(f"Loading SAHI detection model wrapping: {self.yolo_model_name}")
        
        # Determine model path similar to _load_yolo
        model_path = self.yolo_model_name
        backend = os.environ.get("HBMON_INFERENCE_BACKEND", "cpu")
        
        # SAHI supports 'yolov8' model type which uses ultralytics under the hood
        # If we have an OpenVINO model path, SAHI's AutoDetectionModel might not support it directly
        # without custom configuration. For now, we will stick to the PyTorch model path
        # or the OpenVINO model path if SAHI/Ultralytics supports it.
        # Ultralytics YOLO() can load OpenVINO paths, and SAHI uses YOLO().
        
        if "openvino" in backend.lower():
             ov_path = self._get_openvino_model_path()
             if ov_path and ov_path.exists():
                 model_path = str(ov_path)
        
        try:
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=self.confidence,
                device="cpu", # Let Ultralytics handle device (or set cuda/mps if needed)
            )
            # Ensure classes are populated
            if self.bird_class_id is not None:
                 # SAHI might need specific class mapping but it usually respects the model's
                 pass
                 
        except Exception as e:
            logger.error(f"Failed to load SAHI model: {e}")
            self.use_sahi = False
            return None
            
        return self._sahi_model

    def _load_sam(self) -> Any:
        """Load SAM model for box refinement."""
        if self._sam is not None:
            return self._sam
            
        if not self.use_sam:
            return None
        
        try:
            from ultralytics import SAM
        except ImportError:
            logger.warning("SAM not available in ultralytics, skipping refinement")
            self.use_sam = False
            return None
        
        logger.info(f"Loading SAM model for refinement: {self.sam_model_name}")
        
        # Map variant to model file
        sam_models = {
            "sam_b": "sam_b.pt",
            "sam_l": "sam_l.pt", 
            "sam_h": "sam_h.pt",
            "mobile_sam": "mobile_sam.pt",
        }
        
        model_file = sam_models.get(self.sam_model_name, "sam_b.pt")
        
        try:
            self._sam = SAM(model_file)
            logger.info("SAM model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SAM model: {e}")
            self.use_sam = False
            return None
        
        return self._sam

    def initialize(self) -> None:
        """Eagerly initialize models (call once before processing)."""
        if self._initialized:
            return
            
        self._load_yolo()
        self._resolve_bird_class_id()
        if self.use_sam:
            self._load_sam()
        if self.use_sahi:
            self._load_sahi_model()
        
        self._initialized = True

    def detect_frame(
        self,
        frame: np.ndarray,
        refine_with_sam: bool | None = None,
        skip_bird_filter: bool = False,
    ) -> list[DetectedBox]:
        """Detect objects in a single frame.
        
        Args:
            frame: BGR image as numpy array
            refine_with_sam: Override SAM usage for this frame
            skip_bird_filter: If True, return all detected objects (not just birds)
            
        Returns:
            List of detected bounding boxes
        """
        if not self._initialized:
            self.initialize()
        
        # Resolving dependencies and loading models
        boxes = []
        
        # Step 1: Standard YOLO
        boxes = self._detect_standard_yolo(frame, skip_bird_filter=skip_bird_filter)
        
        if os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1":
            if boxes:
                logger.info(f"Standard YOLO found {len(boxes)} boxes")
            else:
                logger.debug("Standard YOLO found 0 boxes")

        # Step 2: Conditional SAHI "Rescue"
        # Only run SAHI if standard YOLO found nothing (and SAHI is enabled)
        if self.use_sahi and not boxes:
             try:
                 sahi_boxes = self._detect_with_sahi(frame)
                 if sahi_boxes:
                     if os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1":
                         logger.info(f"SAHI rescue found {len(sahi_boxes)} boxes")
                     boxes = sahi_boxes
             except Exception as e:
                 logger.debug(f"SAHI detection failed: {e}")
        
        # Optionally refine with SAM
        use_sam = refine_with_sam if refine_with_sam is not None else self.use_sam
        if use_sam and boxes:
            boxes = self._refine_boxes_with_sam(frame, boxes)
        
        # Post-Process NMS
        if boxes:
            boxes = self._apply_nms(boxes, frame_debug=frame, frame_idx=0)
        
        return boxes

    def _detect_with_sahi(self, frame: np.ndarray) -> list[DetectedBox]:
        """Run detection using SAHI sliced inference."""
        sahi_model = self._load_sahi_model()
        if sahi_model is None:
            return self._detect_standard_yolo(frame)
            
        try:
            from sahi.predict import get_sliced_prediction
        except ImportError:
            return self._detect_standard_yolo(frame)
            
        # Sliced prediction
        t0 = time.time()
        
        result = get_sliced_prediction(
            frame,
            sahi_model,
            slice_height=SAHI_SLICE_HEIGHT,
            slice_width=SAHI_SLICE_WIDTH,
            overlap_height_ratio=SAHI_OVERLAP_RATIO,
            overlap_width_ratio=SAHI_OVERLAP_RATIO,
            verbose=0
        )
        
        t_sahi = (time.time() - t0) * 1000
        debug = os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1"
        if debug:
            backend = os.environ.get("HBMON_INFERENCE_BACKEND", "cpu").upper()
            if "OPENVINO" in backend:
                backend = "OpenVINO-GPU" if "GPU" in backend or "OPEN" in backend else "OpenVINO-CPU"
            
            # Estimate slice count (SAHI doesn't directly expose this in PredictionResult but we can infer from dimensions)
            h_img, w_img = frame.shape[:2]
            cols = math.ceil((w_img - SAHI_SLICE_WIDTH * SAHI_OVERLAP_RATIO) / (SAHI_SLICE_WIDTH * (1 - SAHI_OVERLAP_RATIO)))
            rows = math.ceil((h_img - SAHI_SLICE_HEIGHT * SAHI_OVERLAP_RATIO) / (SAHI_SLICE_HEIGHT * (1 - SAHI_OVERLAP_RATIO)))
            slice_count = max(1, rows * cols)
            
            logger.info(f"SAHI Sliced Inference ({backend}): {t_sahi:.2f}ms for ~{slice_count} slices")
        
        boxes = []
        h_img, w_img = frame.shape[:2]
        
        if debug:
            logger.debug(f"SAHI found {len(result.object_prediction_list)} raw objects")

        for prediction in result.object_prediction_list:
            if debug:
                logger.debug(f"Raw SAHI object: {prediction.category.name} (id={prediction.category.id}) score={prediction.score.value:.2f}")
            
            # Filter by class if needed
            if self.bird_class_id is not None:
                 if prediction.category.id != self.bird_class_id:
                     # Check name as fallback
                     name_lower = prediction.category.name.lower()
                     if 'bird' not in name_lower and 'hummingbird' not in name_lower:
                         continue
            
            # SAHI returns absolute coordinates
            bbox = prediction.bbox
            x_min = bbox.minx
            y_min = bbox.miny
            x_max = bbox.maxx
            y_max = bbox.maxy
            
            # Normalize
            cx = ((x_min + x_max) / 2) / w_img
            cy = ((y_min + y_max) / 2) / h_img
            bw = (x_max - x_min) / w_img
            bh = (y_max - y_min) / h_img
            
            boxes.append(DetectedBox(
                class_id=prediction.category.id,
                x=cx,
                y=cy,
                w=bw,
                h=bh,
                confidence=prediction.score.value,
                source="sahi-auto",
            ))
            
        return boxes

    def _detect_standard_yolo(self, frame: np.ndarray, skip_bird_filter: bool = False) -> list[DetectedBox]:
        """Run standard YOLO detection (original implementation).
        
        Args:
            frame: BGR image as numpy array
            skip_bird_filter: If True, detect all classes (not just birds)
        """
        yolo = self._load_yolo()
        if yolo is None:
            return []
        
        # Resolve image size and device
        imgsz_env = os.environ.get("HBMON_YOLO_IMGSZ", "auto")
        predict_imgsz = resolve_predict_imgsz(imgsz_env, frame.shape)
        
        # Run YOLO detection - optionally filter by bird class
        classes = None if skip_bird_filter else ([self.bird_class_id] if self.bird_class_id is not None else None)
        results = yolo(frame, conf=self.confidence, verbose=False, imgsz=predict_imgsz, classes=classes)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        boxes = []
        
        h, w = frame.shape[:2]
        
        for box in result.boxes:
            # Get normalized coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            # Convert to center format and normalize
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            boxes.append(DetectedBox(
                class_id=cls_id,
                x=cx,
                y=cy,
                w=bw,
                h=bh,
                confidence=conf,
                source="auto",
            ))
        
        return boxes
        


    def _refine_boxes_with_sam(
        self,
        frame: np.ndarray,
        boxes: list[DetectedBox],
    ) -> list[DetectedBox]:
        """Refine bounding boxes using SAM segmentation.
        
        SAM provides precise object masks which can improve box accuracy.
        """
        sam = self._load_sam()
        if sam is None:
            return boxes
        
        h, w = frame.shape[:2]
        refined = []
        
        t0 = time.time()
        for box in boxes:
            try:
                # Convert normalized to pixel coordinates
                x1 = int((box.x - box.w / 2) * w)
                y1 = int((box.y - box.h / 2) * h)
                x2 = int((box.x + box.w / 2) * w)
                y2 = int((box.y + box.h / 2) * h)
                
                # Use box prompt for SAM
                results = sam(frame, bboxes=[[x1, y1, x2, y2]], verbose=False)
                
                if results and len(results) > 0 and results[0].masks is not None:
                    # Get mask and compute tight bounding box
                    mask = results[0].masks.data[0].cpu().numpy()
                    
                    # Find bounding box of mask
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    
                    if rows.any() and cols.any():
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        
                        # Convert back to normalized center format
                        cx = ((xmin + xmax) / 2) / w
                        cy = ((ymin + ymax) / 2) / h
                        bw = (xmax - xmin) / w
                        bh = (ymax - ymin) / h
                        
                        refined.append(DetectedBox(
                            class_id=box.class_id,
                            x=cx,
                            y=cy,
                            w=bw,
                            h=bh,
                            confidence=box.confidence,
                            source="sam-refined",
                        ))
                        continue
                
                # Fallback to original box if SAM failed
                refined.append(box)
                
            except Exception as e:
                logger.debug(f"SAM refinement failed for box: {e}")
                refined.append(box)
        

        
        t_sam = (time.time() - t0) * 1000
        debug = os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1"
        if debug:
            logger.info(f"SAM Refinement ({len(boxes)} boxes): {t_sam:.2f}ms")

        return refined

    def _is_bird(self, class_id: int, class_name: str | None = None) -> bool:
        """Helper to determine if a detection is likely a bird."""
        # Check against resolved class ID
        if self.bird_class_id is not None and class_id == self.bird_class_id:
            return True
        
        # Check by name if available
        if class_name:
             name_str = str(class_name).strip().lower()
             if 'bird' in name_str or 'hummingbird' in name_str:
                 return True
            
        # Fallback for COCO bird ID (14) if not explicitly resolved
        if class_id == 14:
            return True
            
        return False

    def _apply_nms(self, boxes: list[DetectedBox], iou_threshold: float | None = None, frame_debug: np.ndarray | None = None, frame_idx: int = -1) -> list[DetectedBox]:
        """
        Apply Non-Maximum Suppression to merge overlapping boxes.
        Useful after SAM refinement which might output overlapping masks/boxes.
        """
        thresh = iou_threshold if iou_threshold is not None else ANNOTATION_NMS_THRESHOLD
        
        # Filter out invalid/tiny boxes first (less than 0.1% of frame dimension)
        boxes = [b for b in boxes if b.w > 0.001 and b.h > 0.001]
        
        if not boxes:
            return boxes
            
        pre_count = len(boxes)
        
        # Debug Dump Logic (Pre-NMS)
        debug_dump = os.environ.get("HBMON_DEBUG_DUMP_IMAGES", "0") == "1"
        dump_dir = Path("/app/debug_output")
        
        if debug_dump and frame_debug is not None:
             dump_dir.mkdir(parents=True, exist_ok=True)
             import cv2
             # Draw Pre-NMS (Red)
             debug_img = frame_debug.copy()
             h, w = debug_img.shape[:2]
             for b in boxes:
                 x1 = int((b.x - b.w/2) * w)
                 y1 = int((b.y - b.h/2) * h)
                 x2 = int((b.x + b.w/2) * w)
                 y2 = int((b.y + b.h/2) * h)
                 cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                 cv2.putText(debug_img, f"{b.confidence:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
             cv2.imwrite(str(dump_dir / f"frame_{frame_idx}_pre_nms.jpg"), debug_img)

        if len(boxes) < 2:
             if os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1":
                 logger.debug(f"NMS: {pre_count} -> {pre_count} (Single box)")
             return boxes

        try:
            import torch
            import torchvision
            
            # 1. Manual pass for containment (if one box is mostly inside another)
            # This handles cases where IoU is low because of size difference
            kept_after_containment = []
            boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)
            
            indices_to_skip = set()
            for i in range(len(boxes_sorted)):
                if i in indices_to_skip:
                    continue
                b1 = boxes_sorted[i]
                kept_after_containment.append(b1)
                
                for j in range(i + 1, len(boxes_sorted)):
                    if j in indices_to_skip:
                        continue
                    b2 = boxes_sorted[j]
                    
                    # Compute Intersection over Area of the smaller box (Containment)
                    x1min, y1min = b1.x - b1.w/2, b1.y - b1.h/2
                    x1max, y1max = b1.x + b1.w/2, b1.y + b1.h/2
                    x2min, y2min = b2.x - b2.w/2, b2.y - b2.h/2
                    x2max, y2max = b2.x + b2.w/2, b2.y + b2.h/2
                    
                    inter_xmin = max(x1min, x2min)
                    inter_ymin = max(y1min, y2min)
                    inter_xmax = min(x1max, x2max)
                    inter_ymax = min(y1max, y2max)
                    
                    inter_w = max(0, inter_xmax - inter_xmin)
                    inter_h = max(0, inter_ymax - inter_ymin)
                    inter_area = inter_w * inter_h
                    
                    area1 = b1.w * b1.h
                    area2 = b2.w * b2.h
                    smaller_area = min(area1, area2)
                    
                    if smaller_area > 0:
                        containment = inter_area / smaller_area
                        if containment > 0.8: # 80% contained
                            indices_to_skip.add(j)
            
            boxes = kept_after_containment
            if len(boxes) < 2:
                return boxes

            # 2. Standard IoU-based NMS
            
            # Prepare data for NMS
            # torchvision.ops.nms expects boxes in (x1, y1, x2, y2) format
            box_tensors = []
            scores = []
            
            for b in boxes:
                # Convert normalized center-xywh to xyxy
                x1 = b.x - (b.w / 2)
                y1 = b.y - (b.h / 2)
                x2 = b.x + (b.w / 2)
                y2 = b.y + (b.h / 2)
                box_tensors.append([x1, y1, x2, y2])
                scores.append(b.confidence)
            
            t_boxes = torch.tensor(box_tensors, dtype=torch.float32)
            t_scores = torch.tensor(scores, dtype=torch.float32)
            
            # Run NMS
            keep_indices = torchvision.ops.nms(t_boxes, t_scores, thresh)
            keep_indices = keep_indices.tolist()
            
            final_boxes = [boxes[i] for i in keep_indices]
            post_count = len(final_boxes)
            
            if os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1":
                 logger.info(f"NMS Applied: {pre_count} boxes -> {post_count} boxes (Threshold: {thresh})")

            # Debug Dump Logic (Post-NMS)
            if debug_dump and frame_debug is not None:
                 import cv2
                 # Draw Post-NMS (Green)
                 debug_img = frame_debug.copy()
                 h, w = debug_img.shape[:2]
                 for b in final_boxes:
                     x1 = int((b.x - b.w/2) * w)
                     y1 = int((b.y - b.h/2) * h)
                     x2 = int((b.x + b.w/2) * w)
                     y2 = int((b.y + b.h/2) * h)
                     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                     cv2.putText(debug_img, f"{b.confidence:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                 cv2.imwrite(str(dump_dir / f"frame_{frame_idx}_post_nms.jpg"), debug_img)
                 
            return final_boxes
            
        except ImportError:
            logger.warning("torchvision not found, skipping Post-SAM NMS")
            return boxes
        except Exception as e:
            logger.warning(f"Post-SAM NMS failed: {e}")
            return boxes

    def detect_batch(
        self,
        frames: list[np.ndarray],
        refine_with_sam: bool | None = None,
    ) -> list[list[DetectedBox]]:
        """Detect objects in a batch of frames.
        
        Args:
            frames: List of BGR images
            refine_with_sam: Override SAM usage
            
        Returns:
            List of detection lists (one per frame)
        """
        if not self._initialized:
            self.initialize()
        
        yolo = self._load_yolo()
        if yolo is None:
            return [[] for _ in frames]
        
        # Resolve image size
        imgsz_env = os.environ.get("HBMON_YOLO_IMGSZ", "auto")
        predict_imgsz = resolve_predict_imgsz(imgsz_env, frames[0].shape if frames else None)
        
        debug = os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1"
        use_sam = refine_with_sam if refine_with_sam is not None else self.use_sam
        
        # Standard batched YOLO (Run First)
        if not yolo:
             return [[] for _ in frames]

        t0 = time.time()
        # Note: We filter manually below to be more robust than strict YOLO class filtering
        results = yolo(frames, conf=self.confidence, verbose=debug, imgsz=predict_imgsz)
        t_infer = (time.time() - t0) * 1000
        
        backend = os.environ.get("HBMON_INFERENCE_BACKEND", "cpu").upper()
        if "OPENVINO" in backend:
            backend = "OpenVINO-GPU" if "GPU" in backend or "OPEN" in backend else "OpenVINO-CPU"
             
        if debug:
            logger.info(f"YOLO Inference ({backend}): {t_infer:.2f}ms")
            logger.debug(f"YOLO results contain {len(results)} items")

        all_boxes = []
        sahi_rescues = 0
        
        for i, result in enumerate(results):
            frame = frames[i]
            h, w = frame.shape[:2]
            boxes = []
            
            names = getattr(result, 'names', {})
            
            # Process YOLO results
            for box in result.boxes:
                # Get normalized coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id)
                
                # Filter by bird class using robust helper
                if not self._is_bird(cls_id, cls_name):
                    continue
                
                # Convert to center format and normalize
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                boxes.append(DetectedBox(
                    class_id=cls_id,
                    x=cx,
                    y=cy,
                    w=bw,
                    h=bh,
                    confidence=conf,
                    source="auto",
                ))
            
            if debug and boxes:
                logger.info(f"Frame {i}: Standard YOLO found {len(boxes)} boxes")
            
            # Conditional SAHI "Rescue"
            # Only run SAHI if YOLO found no birds
            if self.use_sahi and not boxes:
                if self._sahi_model is not None:
                    try:
                        sahi_boxes = self._detect_with_sahi(frame)
                        if sahi_boxes:
                            if debug:
                                logger.info(f"Frame {i}: SAHI rescue found {len(sahi_boxes)} boxes")
                            boxes = sahi_boxes
                            sahi_rescues += 1
                    except Exception as e:
                         # Log but don't crash batch
                        logger.debug(f"Frame {i}: SAHI rescue failed: {e}")

            # Optionally refine with SAM
            if use_sam and boxes:
                boxes = self._refine_boxes_with_sam(frame, boxes)
                if debug:
                    logger.debug(f"Frame {i}: Final count {len(boxes)} boxes after SAM")
            
            # Post-Process NMS
            if boxes:
                batch_start = locals().get('current_batch_start_idx', 0)
                boxes = self._apply_nms(boxes, frame_debug=frame, frame_idx=batch_start + i)


            all_boxes.append(boxes)
        
        if debug and self.use_sahi:
            logger.info(f"Batch completed: SAHI rescued {sahi_rescues}/{len(frames)} frames")

        return all_boxes


def create_detector(
    yolo_model: str | None = None,
    use_sam: bool | None = None,
    use_sahi: bool | None = None,
) -> AnnotationDetector:
    """Factory function to create an annotation detector.
    
    Args:
        yolo_model: Override YOLO model (default from env)
        use_sam: Override SAM usage (default from env)
        use_sahi: Override SAHI usage (default from env)
        
    Returns:
        Configured AnnotationDetector instance
    """
    return AnnotationDetector(
        yolo_model=yolo_model,
        use_sam=use_sam,
        use_sahi=use_sahi,
    )
