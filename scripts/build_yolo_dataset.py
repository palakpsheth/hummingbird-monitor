#!/usr/bin/env python3
"""
Build YOLO dataset from annotations.

This script exports annotated frames to a YOLO-format dataset, including
hard-negative crops from false-positive boxes.

Usage:
    python scripts/build_yolo_dataset.py [--output-dir /data/exports/yolo/dataset]

Export Rules:
    - bird_present=true: Include valid boxes only in YOLO labels
    - bird_present=false: Generate empty label file (true negative)
    - is_false_positive=true: Exclude from labels, export as hard negatives

Output Structure:
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── hard_negatives/
    ├── dataset.yaml
    └── export_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from annotations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/exports/yolo/dataset"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--include-hard-negatives",
        action="store_true",
        default=True,
        help="Export hard negatives from false-positive boxes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually exporting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    try:
        from hbmon.annotation_storage import (
            FRAMES_DIR,
            LABELS_DIR,
            BOXES_DIR,
            get_false_positive_boxes,
            export_hard_negative_crop,
        )
    except ImportError as e:
        logger.error(f"Failed to import hbmon modules: {e}")
        return 1

    output_dir = args.output_dir
    
    if args.dry_run:
        logger.info("DRY RUN - no files will be created")

    # Create output directories
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    hard_neg_dir = output_dir / "hard_negatives"

    if not args.dry_run:
        for d in [train_images, train_labels, val_images, val_labels, hard_neg_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # Find all observation directories with annotations
    obs_dirs = []
    if FRAMES_DIR.exists():
        obs_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
    
    if not obs_dirs:
        logger.warning("No annotated observations found")
        return 0

    logger.info(f"Found {len(obs_dirs)} observation directories")

    # Collect all annotated frames
    frames_to_export = []
    
    for obs_dir in obs_dirs:
        obs_id = obs_dir.name
        labels_dir = LABELS_DIR / obs_id
        boxes_dir = BOXES_DIR / obs_id
        
        if not labels_dir.exists():
            logger.debug(f"No labels for observation {obs_id}, skipping")
            continue

        for frame_file in sorted(obs_dir.glob("*.jpg")):
            frame_name = frame_file.stem  # e.g., "frame_000001"
            label_file = labels_dir / f"{frame_name}.txt"
            box_file = boxes_dir / f"{frame_name}.json"
            
            # Check if this frame was reviewed (has box JSON)
            if not box_file.exists():
                logger.debug(f"Frame {frame_file} not reviewed, skipping")
                continue

            # Load box metadata
            try:
                box_data = json.loads(box_file.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Failed to read box file {box_file}")
                continue

            bird_present = box_data.get("bird_present", False)
            
            frames_to_export.append({
                "obs_id": obs_id,
                "frame_file": frame_file,
                "label_file": label_file if label_file.exists() else None,
                "box_file": box_file,
                "bird_present": bird_present,
                "box_data": box_data,
            })

    if not frames_to_export:
        logger.warning("No reviewed frames found to export")
        return 0

    logger.info(f"Found {len(frames_to_export)} reviewed frames")

    # Shuffle and split into train/val
    random.seed(args.seed)
    random.shuffle(frames_to_export)
    
    split_idx = int(len(frames_to_export) * args.train_split)
    train_frames = frames_to_export[:split_idx]
    val_frames = frames_to_export[split_idx:]

    logger.info(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    # Export frames
    stats = {
        "train_frames": 0,
        "val_frames": 0,
        "tp_frames": 0,
        "tn_frames": 0,
        "fp_boxes": 0,
        "hard_negatives": 0,
        "valid_boxes": 0,
    }

    def export_frame(frame_info: dict, target_images: Path, target_labels: Path) -> None:
        frame_file = frame_info["frame_file"]
        label_file = frame_info["label_file"]
        obs_id = frame_info["obs_id"]
        
        # Create unique filename
        out_name = f"{obs_id}_{frame_file.stem}"
        
        # Copy image
        out_image = target_images / f"{out_name}.jpg"
        if not args.dry_run:
            shutil.copy(frame_file, out_image)
        
        # Copy or create label
        out_label = target_labels / f"{out_name}.txt"
        if label_file and label_file.exists():
            if not args.dry_run:
                shutil.copy(label_file, out_label)
        else:
            # Empty label file for true negative
            if not args.dry_run:
                out_label.write_text("")

        # Update stats
        box_data = frame_info.get("box_data", {})
        boxes = box_data.get("boxes", [])
        
        if frame_info["bird_present"]:
            stats["tp_frames"] += 1
            stats["valid_boxes"] += sum(1 for b in boxes if not b.get("is_false_positive", False))
        else:
            stats["tn_frames"] += 1

        stats["fp_boxes"] += sum(1 for b in boxes if b.get("is_false_positive", False))

    # Export training frames
    for frame_info in train_frames:
        export_frame(frame_info, train_images, train_labels)
        stats["train_frames"] += 1

    # Export validation frames
    for frame_info in val_frames:
        export_frame(frame_info, val_images, val_labels)
        stats["val_frames"] += 1

    # Export hard negatives
    if args.include_hard_negatives:
        logger.info("Exporting hard negatives from false-positive boxes...")
        
        for obs_dir in obs_dirs:
            obs_id = obs_dir.name
            fp_boxes = get_false_positive_boxes(obs_id)
            
            for frame_idx, box in fp_boxes:
                frame_file = FRAMES_DIR / obs_id / f"frame_{frame_idx:06d}.jpg"
                if frame_file.exists():
                    crop_path = export_hard_negative_crop(
                        obs_id, frame_idx, box, frame_file, margin=0.15
                    )
                    if crop_path:
                        stats["hard_negatives"] += 1
                        if not args.dry_run:
                            # Copy to output hard_negatives dir
                            out_crop = hard_neg_dir / crop_path.name
                            shutil.copy(crop_path, out_crop)

    # Create dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    yaml_content = f"""# YOLO Dataset - Generated {datetime.utcnow().isoformat()}Z
path: {output_dir}
train: images/train
val: images/val

names:
  0: bird

# Statistics
# Train frames: {stats['train_frames']}
# Val frames: {stats['val_frames']}
# Hard negatives: {stats['hard_negatives']}
"""
    
    if not args.dry_run:
        dataset_yaml.write_text(yaml_content)
        logger.info(f"Created {dataset_yaml}")

    # Save export stats
    stats_file = output_dir / "export_stats.json"
    stats["exported_at"] = datetime.utcnow().isoformat() + "Z"
    stats["seed"] = args.seed
    stats["train_split"] = args.train_split
    
    if not args.dry_run:
        stats_file.write_text(json.dumps(stats, indent=2))
        logger.info(f"Saved stats to {stats_file}")

    # Summary
    logger.info("=" * 50)
    logger.info("Export Summary:")
    logger.info(f"  Train frames: {stats['train_frames']}")
    logger.info(f"  Val frames: {stats['val_frames']}")
    logger.info(f"  TP frames: {stats['tp_frames']}")
    logger.info(f"  TN frames: {stats['tn_frames']}")
    logger.info(f"  Valid boxes: {stats['valid_boxes']}")
    logger.info(f"  FP boxes: {stats['fp_boxes']}")
    logger.info(f"  Hard negatives: {stats['hard_negatives']}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
