#!/usr/bin/env python3
"""
Download annotation models (YOLO + SAM) to cache on container start.
"""
import os
import logging
from ultralytics import YOLO, SAM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

def download_models():
    # 1. Download YOLO model
    yolo_model = os.environ.get("HBMON_ANNOTATION_YOLO_MODEL", "yolo11l.pt")
    logger.info(f"Checking/Downloading YOLO model: {yolo_model}")
    try:
        YOLO(yolo_model)
        logger.info("YOLO model ready.")
    except Exception as e:
        logger.error(f"Failed to download YOLO model {yolo_model}: {e}")
        # Don't exit error, just log. Workers might still try or using cached.

    # 2. Download SAM model if enabled
    use_sam = os.environ.get("HBMON_ANNOTATION_USE_SAM", "1") == "1"
    if use_sam:
        sam_model_name = os.environ.get("HBMON_ANNOTATION_SAM_MODEL", "sam_b")
        sam_map = {
            "sam_b": "sam_b.pt",
            "sam_l": "sam_l.pt",
            "sam_h": "sam_h.pt",
            "mobile_sam": "mobile_sam.pt",
        }
        sam_file = sam_map.get(sam_model_name, "sam_b.pt")
        logger.info(f"Checking/Downloading SAM model: {sam_file}")
        try:
            SAM(sam_file)
            logger.info("SAM model ready.")
        except Exception as e:
            logger.warning(f"Failed to download SAM model {sam_file}: {e}")

if __name__ == "__main__":
    download_models()
