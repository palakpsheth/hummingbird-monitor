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
    ):
        """Initialize the annotation detector.
        
        Args:
            yolo_model: YOLO model path/name (default from env)
            use_sam: Whether to use SAM refinement (default from env)
            sam_model: SAM model variant (default from env)
            confidence: Detection confidence threshold (default from env)
        """
        self.yolo_model_name = yolo_model or ANNOTATION_YOLO_MODEL
        self.use_sam = use_sam if use_sam is not None else ANNOTATION_USE_SAM
        self.sam_model_name = sam_model or ANNOTATION_SAM_MODEL
        self.confidence = confidence if confidence is not None else ANNOTATION_CONFIDENCE
        
        self._yolo = None
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
                if str(v).strip().lower() in ('bird', 'hummingbird'):
                    target_id = int(k)
                    break
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).strip().lower() in ('bird', 'hummingbird'):
                    target_id = int(i)
                    break
                    
        if target_id is not None:
            self.bird_class_id = target_id
            logger.info(f"Resolved annotator bird_class_id={target_id} from model")
        else:
            logger.warning("Could not resolve 'bird' class ID from model names. Detections will be unfiltered.")

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
                self._yolo = YOLO(self.yolo_model_name, task="detect")
                # Export to OpenVINO format for future use
                try:
                    self._export_to_openvino()
                except Exception as e:
                    logger.warning(f"OpenVINO export failed, using PyTorch: {e}")
        else:
            self._yolo = YOLO(self.yolo_model_name, task="detect")
        
        return self._yolo

    def _get_openvino_model_path(self) -> Path | None:
        """Get path to OpenVINO-exported model if it exists."""
        cache_dir = Path(os.environ.get("OPENVINO_CACHE_DIR", "/data/openvino_cache"))
        model_name = Path(self.yolo_model_name).stem
        ov_path = cache_dir / "annotation" / f"{model_name}_openvino_model" / f"{model_name}.xml"
        return ov_path if ov_path.exists() else None

    def _export_to_openvino(self) -> Path | None:
        """Export YOLO model to OpenVINO format."""
        if self._yolo is None:
            return None
            
        cache_dir = Path(os.environ.get("OPENVINO_CACHE_DIR", "/data/openvino_cache"))
        export_dir = cache_dir / "annotation"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Exporting annotation YOLO model to OpenVINO format...")
            self._yolo.export(format="openvino", half=False)
            logger.info("OpenVINO export complete")
        except Exception as e:
            logger.warning(f"OpenVINO export failed: {e}")
        
        return self._get_openvino_model_path()

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
        
        self._initialized = True

    def detect_frame(
        self,
        frame: np.ndarray,
        refine_with_sam: bool | None = None,
    ) -> list[DetectedBox]:
        """Detect objects in a single frame.
        
        Args:
            frame: BGR image as numpy array
            refine_with_sam: Override SAM usage for this frame
            
        Returns:
            List of detected bounding boxes
        """
        if not self._initialized:
            self.initialize()
        
        yolo = self._load_yolo()
        if yolo is None:
            return []
        
        # Resolve image size and device
        imgsz_env = os.environ.get("HBMON_YOLO_IMGSZ", "auto")
        predict_imgsz = resolve_predict_imgsz(imgsz_env, frame.shape)
        
        # Run YOLO detection
        classes = [self.bird_class_id] if self.bird_class_id is not None else None
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
        
        # Optionally refine with SAM
        use_sam = refine_with_sam if refine_with_sam is not None else self.use_sam
        if use_sam and boxes:
            boxes = self._refine_boxes_with_sam(frame, boxes)
        
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
        # Use first frame for auto-size calculation (assuming uniform batch)
        imgsz_env = os.environ.get("HBMON_YOLO_IMGSZ", "auto")
        predict_imgsz = resolve_predict_imgsz(imgsz_env, frames[0].shape if frames else None)
        
        # Run batch detection
        # Resolves logic: if env sends "auto" and we have frame shape, it returns stride-aligned [h, w]
        # Run batch detection
        debug = os.environ.get("HBMON_ANNOTATOR_DEBUG", "0") == "1"
        t0 = time.time()
        classes = [self.bird_class_id] if self.bird_class_id is not None else None
        results = yolo(frames, conf=self.confidence, verbose=debug, imgsz=predict_imgsz, classes=classes)
        t_infer = (time.time() - t0) * 1000
        
        backend = os.environ.get("HBMON_INFERENCE_BACKEND", "cpu").upper()
        # If openvino backend, enhance label
        if "OPENVINO" in backend:
             # Check if we forced GPU
             # Ideally we check device, but env var is close enough for log matching
             backend = "OpenVINO-GPU" if "GPU" in backend or "OPEN" in backend else "OpenVINO-CPU"
             
        if debug:
             logger.info(f"YOLO Inference ({backend}): {t_infer:.2f}ms")
             # Also log image size used for verification
             if isinstance(predict_imgsz, list):
                 sz_str = f"{predict_imgsz[0]}x{predict_imgsz[1]}"
             else:
                 sz_str = f"{predict_imgsz}x{predict_imgsz}"
             logger.debug(f"YOLO predict_imgsz: {sz_str}")
        
        all_boxes = []
        use_sam = refine_with_sam if refine_with_sam is not None else self.use_sam
        
        for i, result in enumerate(results):
            frame = frames[i]
            h, w = frame.shape[:2]
            boxes = []
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
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
            
            # Optionally refine with SAM
            if use_sam and boxes:
                boxes = self._refine_boxes_with_sam(frame, boxes)
            
            all_boxes.append(boxes)
        
        return all_boxes


def create_detector(
    yolo_model: str | None = None,
    use_sam: bool | None = None,
) -> AnnotationDetector:
    """Factory function to create an annotation detector.
    
    Args:
        yolo_model: Override YOLO model (default from env)
        use_sam: Override SAM usage (default from env)
        
    Returns:
        Configured AnnotationDetector instance
    """
    return AnnotationDetector(
        yolo_model=yolo_model,
        use_sam=use_sam,
    )
