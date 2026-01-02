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
    """Get database URL from environment or default."""
    # Try async URL first, fall back to sync
    url = os.getenv("HBMON_DB_ASYNC_URL", "")
    if not url:
        url = os.getenv("DATABASE_URL", "postgresql://hbmon:hbmon@localhost:5432/hbmon")
    # Convert async to sync if needed
    return url.replace("+asyncpg", "")


def get_media_dir() -> Path:
    """Get media directory from environment or default."""
    return Path(os.getenv("HBMON_MEDIA_DIR", "./data/media"))


def query_observations(
    limit: int = 50,
    species_filter: str | None = None,
) -> list[dict]:
    """Query observations from the database."""
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
        
        if species_filter:
            query += " AND species_label ILIKE %s"
            params.append(f"%{species_filter}%")
        
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
        print(f"Connection params: host={conn_params['host']}, port={conn_params['port']}, dbname={conn_params['dbname']}", file=sys.stderr)
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
    args = ap.parse_args()

    media_dir = get_media_dir()
    observations = query_observations(limit=args.limit, species_filter=args.species)
    
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
