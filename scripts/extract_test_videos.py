#!/usr/bin/env python3
"""
extract_test_videos.py

Query the hbmon database for observation videos and list/copy them for testing.
Use this to find videos of known bird detections for pipeline verification.

Usage:
    # List all observation videos
    uv run python scripts/extract_test_videos.py

    # List only recent 5 videos
    uv run python scripts/extract_test_videos.py --limit 5

    # List videos with full paths (for use with rtsp_test_server.py)
    uv run python scripts/extract_test_videos.py --full-path --limit 3

    # Copy videos to a staging directory
    uv run python scripts/extract_test_videos.py --copy-to /tmp/test_videos --limit 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_db_url() -> str:
    """Get database URL from environment or default.
    
    Handles Docker hostnames by translating them to localhost.
    """
    # Try async URL first, fall back to sync
    url = os.getenv("HBMON_DB_ASYNC_URL", "")
    if not url:
        url = os.getenv("DATABASE_URL", "postgresql://hbmon:hbmon@localhost:5432/hbmon")
    # Remove async driver suffix
    url = url.replace("+asyncpg", "").replace("+psycopg", "")
    # Translate Docker hostnames to localhost for host-run scripts
    url = url.replace("@hbmon-db:", "@localhost:")
    return url


def get_media_dir() -> Path:
    """Get media directory from environment or default."""
    return Path(os.getenv("HBMON_MEDIA_DIR", "./data/media"))


def query_observations(
    limit: int = 50,
    species_filter: str | None = None,
    true_positive_only: bool = False,
    review_label: str | None = None,
    observation_id: int | None = None,
) -> list[dict]:
    """Query observations from the database.
    
    Args:
        true_positive_only: Only return observations with review label 'true_positive'
        review_label: Filter by specific review label (e.g., 'true_positive', 'false_positive')
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    db_url = get_db_url()
    
    # Parse connection string
    # Format: postgresql://user:pass@host:port/dbname
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    
    conn_params = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "hbmon",
        "password": parsed.password or "hbmon",
        "dbname": parsed.path.lstrip("/") or "hbmon",
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        
        query = """
            SELECT id, ts, species_label, video_path, snapshot_path, 
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   extra_json
            FROM observations
            WHERE video_path IS NOT NULL AND video_path != ''
        """
        params: list = []
        
        if observation_id:
            query += " AND id = %s"
            params.append(observation_id)
        
        if species_filter:
            query += " AND species_label ILIKE %s"
            params.append(f"%{species_filter}%")
        
        if true_positive_only:
            query += " AND extra_json::jsonb->'review'->>'label' = 'true_positive'"
        elif review_label:
            query += " AND extra_json::jsonb->'review'->>'label' = %s"
            params.append(review_label)
        
        query += " ORDER BY id DESC"
        
        if limit > 0:
            query += f" LIMIT {limit}"
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "ts": row[1],
                "species_label": row[2],
                "video_path": row[3],
                "snapshot_path": row[4],
                "bbox": (row[5], row[6], row[7], row[8]) if row[5] else None,
                "extra_json": row[9],
            })
        
        cur.close()
        conn.close()
        return results
        
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        safe_conn_info = {
            "host": conn_params.get("host"),
            "port": conn_params.get("port"),
            "dbname": conn_params.get("dbname"),
        }
        print(f"Connection params (without password): {safe_conn_info}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract observation videos from the database for testing."
    )
    ap.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Maximum number of videos to list (default: 10, 0 for all)"
    )
    ap.add_argument(
        "--species", "-s",
        type=str,
        default=None,
        help="Filter by species label (substring match)"
    )
    ap.add_argument(
        "--full-path", "-f",
        action="store_true",
        help="Print full absolute paths instead of relative"
    )
    ap.add_argument(
        "--copy-to", "-c",
        type=str,
        default=None,
        help="Copy videos to this directory"
    )
    ap.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    ap.add_argument(
        "--paths-only", "-p",
        action="store_true",
        help="Output only video paths, one per line (for piping)"
    )
    ap.add_argument(
        "--true-positive", "-t",
        action="store_true",
        help="Only include observations labeled as 'true_positive' (for e2e testing)"
    )
    ap.add_argument(
        "--review-label", "-r",
        type=str,
        default=None,
        help="Filter by review label (e.g., 'true_positive', 'false_positive')"
    )
    ap.add_argument(
        "--observation-id", "-o",
        type=int,
        default=None,
        help="Get video for specific observation ID"
    )
    ap.add_argument(
        "--export-e2e",
        type=str,
        default=None,
        help="Export single observation as E2E test case to tests/integration/test_data/e2e/<NAME>"
    )
    ap.add_argument(
        "--roi-config",
        type=str,
        default=None,
        help="Optional path to config.json to pull ROI from if missing in metadata"
    )
    args = ap.parse_args()

    media_dir = get_media_dir()
    observations = query_observations(
        limit=args.limit,
        species_filter=args.species,
        true_positive_only=args.true_positive,
        review_label=args.review_label,
        observation_id=args.observation_id
    )
    
    if not observations:
        print("No observations found.", file=sys.stderr)
        return 1
    
    if args.json:
        import json
        # Add full paths to output
        for obs in observations:
            video_path = media_dir / obs["video_path"]
            obs["video_full_path"] = str(video_path)
            obs["video_exists"] = video_path.exists()
            obs["ts"] = str(obs["ts"]) if obs["ts"] else None
        print(json.dumps(observations, indent=2))
        return 0
    
    if args.paths_only:
        for obs in observations:
            video_path = media_dir / obs["video_path"]
            if video_path.exists():
                if args.full_path:
                    print(video_path.resolve())
                else:
                    print(video_path)
        return 0
    
    # Copy mode
    if args.copy_to:
        dest_dir = Path(args.copy_to)
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for obs in observations:
            video_path = media_dir / obs["video_path"]
            if video_path.exists():
                dest_path = dest_dir / f"obs_{obs['id']}_{video_path.name}"
                print(f"Copying: {video_path.name} -> {dest_path}")
                shutil.copy2(video_path, dest_path)
                copied += 1
        print(f"\nCopied {copied} videos to {dest_dir}")
        return 0
    
    # Export E2E mode
    if args.export_e2e:
        if len(observations) != 1:
            print("ERROR: --export-e2e requires exactly one observation. Use --observation-id.", file=sys.stderr)
            return 1
        
        obs = observations[0]
        video_path = media_dir / obs["video_path"]
        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
            return 1
            
        case_name = args.export_e2e
        target_dir = Path("tests/integration/test_data/e2e") / case_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy video
        dest_video = target_dir / "clip.mp4"
        print(f"Copying video to {dest_video}...")
        shutil.copy2(video_path, dest_video)
        
        # Parse extra_json for metadata
        import json
        extra = obs["extra_json"] if isinstance(obs["extra_json"], dict) else json.loads(obs["extra_json"])
        detection = extra.get("detection", {})
        
        # ROI
        roi_coords = detection.get("roi_xyxy")
        if not roi_coords and args.roi_config:
            print(f"Loading fallback ROI from {args.roi_config}...")
            try:
                with open(args.roi_config, "r") as f:
                    prod_config = json.load(f)
                    roi_in = prod_config.get("roi")
                    if roi_in:
                        # Convert normalized ROI to absolute pixels
                        video_meta = extra.get("video", {})
                        w = video_meta.get("width")
                        h = video_meta.get("height")
                        if w and h:
                            roi_coords = [
                                int(roi_in["x1"] * w),
                                int(roi_in["y1"] * h),
                                int(roi_in["x2"] * w),
                                int(roi_in["y2"] * h)
                            ]
                            print(f"Derived ROI from config: {roi_coords}")
                        else:
                            print("ERROR: Cannot derive absolute ROI from config without video dimensions in metadata.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Failed to load ROI from config: {e}", file=sys.stderr)

        if not roi_coords:
            # Fallback/Warning
            print("WARNING: 'roi_xyxy' not found in metadata and no --roi-config provided. ROI config will be missing.", file=sys.stderr)
            roi_coords = None
            
        # Write config.json
        config_data = {}
        if roi_coords:
             # Normalize ROI
             video_meta = extra.get("video", {})
             w = video_meta.get("width")
             h = video_meta.get("height")
             if w and h:
                  x1, y1, x2, y2 = roi_coords
                  config_data["roi"] = {
                      "x1": x1 / w,
                      "y1": y1 / h,
                      "x2": x2 / w,
                      "y2": y2 / h
                  }
             else:
                  print("WARNING: Video dimensions missing in metadata. Cannot normalize ROI.", file=sys.stderr)
                  # Fallback to list? No, Settings expects dict.
                  # We can't safely produce a valid config without normalization.
        
        with open(target_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
            
        # Write metadata.json
        metadata = {
            "description": f"Extracted from observation {obs['id']}",
            "species": obs["species_label"],
            "expect_detection": True, # Assumed if from DB
            "yolo_imgsz": detection.get("yolo_imgsz", "auto"),
            "roi": roi_coords,
            "original_bbox": detection.get("bbox_xyxy"),
            "original_conf": detection.get("box_confidence"),
        }
        with open(target_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Exported test case '{case_name}' to {target_dir}")
        return 0

    # Default: print table
    print(f"Found {len(observations)} observation(s) with videos:\n")
    print(f"{'ID':>5} {'Species':<30} {'Video Path':<50} {'Exists':<6}")
    print("-" * 95)
    
    for obs in observations:
        video_path = media_dir / obs["video_path"]
        exists = "✓" if video_path.exists() else "✗"
        path_str = str(video_path.resolve()) if args.full_path else obs["video_path"]
        print(f"{obs['id']:>5} {obs['species_label'][:29]:<30} {path_str[:49]:<50} {exists:<6}")
    
    print(f"\nMedia directory: {media_dir.resolve()}")
    print("\nTip: Use --paths-only -f to get full paths for rtsp_test_server.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
