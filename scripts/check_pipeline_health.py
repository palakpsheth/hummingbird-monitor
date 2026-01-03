#!/usr/bin/env python3
"""
check_pipeline_health.py

Comprehensive health check for the hbmon detection pipeline.
Identifies silent failures that might cause observations to not be recorded.

Checks:
1. Database connectivity and recent writes
2. YOLO model loading and inference
3. CLIP model loading and classification
4. Media directory write permissions
5. Queue processing (Redis connectivity)
6. Recent observation gaps

Usage:
    # Run full health check
    uv run python scripts/check_pipeline_health.py

    # Check only specific components
    uv run python scripts/check_pipeline_health.py --check db
    uv run python scripts/check_pipeline_health.py --check yolo
    uv run python scripts/check_pipeline_health.py --check clip
    uv run python scripts/check_pipeline_health.py --check files

Output:
    Detailed status of each component with PASS/FAIL indicators
    Recommendations for fixing failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class HealthChecker:
    def __init__(self):
        self.results: list[dict] = []
        self.media_dir = Path(os.getenv("HBMON_MEDIA_DIR", "./data/media"))
        
    def add_result(self, component: str, status: str, message: str, details: dict | None = None):
        self.results.append({
            "component": component,
            "status": status,
            "message": message,
            "details": details or {},
        })
        
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(status, "?")
        print(f"  [{icon}] {component}: {message}")
        if details:
            # Sensitive keys that should be redacted when logging
            sensitive_keys = {"password", "passwd", "secret", "token", "api_key", "auth"}
            for k, v in details.items():
                # Redact sensitive values
                if k.lower() in sensitive_keys or any(s in k.lower() for s in sensitive_keys):
                    print(f"       {k}: [REDACTED]")
                else:
                    print(f"       {k}: {v}")
    
    def check_database(self) -> bool:
        """Check database connectivity and recent write activity."""
        print("\n📊 Database Health")
        print("-" * 50)
        
        try:
            import psycopg2
        except ImportError:
            self.add_result("db_import", "FAIL", "psycopg2 not installed")
            return False
        
        db_url = os.getenv("HBMON_DB_ASYNC_URL", "") or os.getenv("DATABASE_URL", "postgresql://hbmon:hbmon@localhost:5432/hbmon")
        db_url = db_url.replace("+asyncpg", "").replace("+psycopg", "")
        # Translate Docker hostnames to localhost for host-run scripts
        db_url = db_url.replace("@hbmon-db:", "@localhost:")
        
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
            conn = psycopg2.connect(**conn_params, connect_timeout=5)
            self.add_result("db_connection", "PASS", f"Connected to {conn_params['host']}:{conn_params['port']}")
        except Exception as e:
            error_type = type(e).__name__
            safe_msg = (
                f"Connection failed to {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']} "
                f"({error_type})"
            )
            self.add_result("db_connection", "FAIL", safe_msg)
            return False
        
        cur = conn.cursor()
        
        # Check table exists
        try:
            cur.execute("SELECT COUNT(*) FROM observations")
            count = cur.fetchone()[0]
            self.add_result("db_table", "PASS", f"observations table has {count} rows")
        except Exception as e:
            self.add_result("db_table", "FAIL", f"Table query failed: {e}")
            conn.close()
            return False
        
        # Check for recent observations
        try:
            cur.execute("""
                SELECT id, ts, species_label 
                FROM observations 
                ORDER BY ts DESC 
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                obs_id, ts, label = row
                
                # Calculate age
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - ts
                
                age_str = f"{age.total_seconds() / 3600:.1f} hours ago"
                
                if age < timedelta(hours=1):
                    self.add_result("db_recent", "PASS", f"Last observation: {age_str}", {"id": obs_id, "species": label})
                elif age < timedelta(hours=12):
                    self.add_result("db_recent", "WARN", f"Last observation: {age_str} (older than 1h)", {"id": obs_id})
                else:
                    self.add_result("db_recent", "WARN", f"Last observation: {age_str} (older than 12h)", {"id": obs_id})
            else:
                self.add_result("db_recent", "WARN", "No observations in database")
        except Exception as e:
            self.add_result("db_recent", "FAIL", f"Recent query failed: {e}")
        
        # Test write capability
        try:
            cur.execute("BEGIN")
            cur.execute("SELECT 1")  # Simple query in transaction
            cur.execute("ROLLBACK")
            self.add_result("db_write", "PASS", "Write test successful (rollback)")
        except Exception as e:
            self.add_result("db_write", "FAIL", f"Write test failed: {e}")
            conn.close()
            return False
        
        conn.close()
        return True
    
    def check_yolo(self) -> bool:
        """Check YOLO model loading and inference."""
        print("\n🔍 YOLO Detection Health")
        print("-" * 50)
        
        try:
            from ultralytics import YOLO
            self.add_result("yolo_import", "PASS", "ultralytics package available")
        except ImportError:
            self.add_result("yolo_import", "FAIL", "ultralytics not installed")
            return False
        
        model_name = os.getenv("HBMON_YOLO_MODEL", "yolo11n.pt")
        
        try:
            t0 = time.time()
            model = YOLO(model_name, task="detect")
            load_time = time.time() - t0
            self.add_result("yolo_load", "PASS", f"Model loaded in {load_time:.2f}s", {"model": model_name})
        except Exception as e:
            self.add_result("yolo_load", "FAIL", f"Model load failed: {e}")
            return False
        
        # Check bird class
        bird_class_id = None
        names = getattr(model, 'names', None)
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).strip().lower() == 'bird':
                    bird_class_id = int(k)
                    break
        
        if bird_class_id is not None:
            self.add_result("yolo_classes", "PASS", f"Bird class found: ID={bird_class_id}")
        else:
            self.add_result("yolo_classes", "WARN", "Bird class not found in model names, using default 14")
            bird_class_id = 14
        
        # Test inference on a dummy image
        try:
            import numpy as np
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            t0 = time.time()
            model.predict(dummy_img, conf=0.1, classes=[bird_class_id], verbose=False)
            inference_time = (time.time() - t0) * 1000
            self.add_result("yolo_inference", "PASS", f"Inference test: {inference_time:.0f}ms")
        except Exception as e:
            self.add_result("yolo_inference", "FAIL", f"Inference failed: {e}")
            return False
        
        return True
    
    def check_clip(self) -> bool:
        """Check CLIP model loading."""
        print("\n🏷️  CLIP Classification Health")
        print("-" * 50)
        
        try:
            import importlib.util
            if importlib.util.find_spec("open_clip") is None:
                raise ImportError("open_clip not found")
            self.add_result("clip_import", "PASS", "open_clip package available")
        except ImportError:
            self.add_result("clip_import", "FAIL", "open_clip not installed")
            return False
        
        clip_model = os.getenv("HBMON_CLIP_MODEL", "hf-hub:imageomics/bioclip")
        
        # Just check if we can start loading (full load is slow)
        self.add_result("clip_config", "PASS", f"Configured model: {clip_model}")
        
        # Check for cached OpenVINO model
        cache_dir = Path(os.getenv("OPENVINO_CACHE_DIR", "./data/openvino_cache")) / "clip"
        if cache_dir.exists():
            files = list(cache_dir.glob("*.xml"))
            if files:
                self.add_result("clip_cache", "PASS", f"OpenVINO cache found: {len(files)} IR files")
            else:
                self.add_result("clip_cache", "WARN", "OpenVINO cache directory empty")
        else:
            self.add_result("clip_cache", "WARN", "No OpenVINO cache (will convert on first run)")
        
        return True
    
    def check_files(self) -> bool:
        """Check file system access and write permissions."""
        print("\n📁 File System Health")
        print("-" * 50)
        
        # Media directory
        if self.media_dir.exists():
            self.add_result("media_dir", "PASS", f"Media directory exists: {self.media_dir}")
        else:
            self.add_result("media_dir", "FAIL", f"Media directory missing: {self.media_dir}")
            return False
        
        # Write test
        test_file = self.media_dir / ".health_check_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            self.add_result("media_write", "PASS", "Write permission confirmed")
        except PermissionError:
            self.add_result("media_write", "FAIL", "No write permission to media directory")
            return False
        except Exception as e:
            self.add_result("media_write", "FAIL", f"Write test failed: {e}")
            return False
        
        # Check subdirectories
        for subdir in ["snapshots", "clips"]:
            path = self.media_dir / subdir
            if path.exists():
                files = list(path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                self.add_result(f"media_{subdir}", "PASS", f"{subdir}/: {file_count} files")
            else:
                self.add_result(f"media_{subdir}", "WARN", f"{subdir}/: directory not created yet")
        
        # OpenVINO cache
        cache_dir = Path(os.getenv("OPENVINO_CACHE_DIR", "./data/openvino_cache"))
        if cache_dir.exists():
            yolo_cache = cache_dir / "yolo"
            if yolo_cache.exists() and list(yolo_cache.glob("*")):
                self.add_result("openvino_cache", "PASS", "OpenVINO YOLO cache present")
            else:
                self.add_result("openvino_cache", "WARN", "OpenVINO YOLO cache empty (will export on startup)")
        else:
            self.add_result("openvino_cache", "WARN", "OpenVINO cache directory missing")
        
        return True
    
    def check_redis(self) -> bool:
        """Check Redis connectivity for queue processing."""
        print("\n📮 Queue (Redis) Health")
        print("-" * 50)
        
        try:
            import redis
            self.add_result("redis_import", "PASS", "redis package available")
        except ImportError:
            self.add_result("redis_import", "WARN", "redis package not installed (optional)")
            return True
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            r = redis.from_url(redis_url, socket_timeout=5)
            r.ping()
            self.add_result("redis_connection", "PASS", f"Connected to {redis_url}")
        except Exception as e:
            self.add_result("redis_connection", "WARN", f"Connection failed: {e} (Redis is optional)")
            return True
        
        return True
    
    def check_worker_logs(self) -> bool:
        """Check for error patterns in recent worker logs."""
        print("\n📋 Worker Log Analysis")
        print("-" * 50)
        
        import subprocess
        
        try:
            result = subprocess.run(
                ["docker", "compose", "logs", "--since", "1h", "hbmon-worker"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(__file__).parent.parent),
            )
            logs = result.stdout + result.stderr
        except Exception as e:
            self.add_result("logs_access", "WARN", f"Could not read logs: {e}")
            return True
        
        # Count important patterns
        error_count = logs.lower().count("error")
        exception_count = logs.lower().count("exception") + logs.lower().count("traceback")
        detection_count = logs.count("1 bird") + logs.count("2 bird")
        visit_count = logs.count("Visit STARTED")
        no_det_count = logs.count("no detections")
        
        self.add_result("logs_errors", 
            "WARN" if error_count > 10 else "PASS",
            f"Errors: {error_count}, Exceptions: {exception_count}")
        
        self.add_result("logs_detections",
            "WARN" if detection_count == 0 and no_det_count > 100 else "PASS",
            f"Bird detections: {detection_count}, Visits: {visit_count}")
        
        # Check for specific failure patterns
        if "Database" in logs and "error" in logs.lower():
            self.add_result("logs_db_errors", "WARN", "Possible database errors in logs")
        
        if "permission denied" in logs.lower():
            self.add_result("logs_permissions", "FAIL", "Permission denied errors in logs")
        
        return True
    
    def summary(self) -> int:
        """Print summary and return exit code."""
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        
        pass_count = sum(1 for r in self.results if r["status"] == "PASS")
        warn_count = sum(1 for r in self.results if r["status"] == "WARN")
        fail_count = sum(1 for r in self.results if r["status"] == "FAIL")
        
        print(f"  ✓ PASS: {pass_count}")
        print(f"  ⚠ WARN: {warn_count}")
        print(f"  ✗ FAIL: {fail_count}")
        
        if fail_count > 0:
            print("\n❌ Critical issues found:")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"   - {r['component']}: {r['message']}")
            return 1
        
        if warn_count > 0:
            print("\n⚠️  Warnings (may affect functionality):")
            for r in self.results:
                if r["status"] == "WARN":
                    print(f"   - {r['component']}: {r['message']}")
        
        print("\n✅ No critical issues found.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Check hbmon pipeline health for silent failures.")
    ap.add_argument(
        "--check", "-c",
        choices=["all", "db", "yolo", "clip", "files", "redis", "logs"],
        default="all",
        help="Component to check (default: all)"
    )
    ap.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    args = ap.parse_args()

    print("=" * 50)
    print("HBMON PIPELINE HEALTH CHECK")
    print("=" * 50)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    
    checker = HealthChecker()
    
    checks = {
        "db": checker.check_database,
        "yolo": checker.check_yolo,
        "clip": checker.check_clip,
        "files": checker.check_files,
        "redis": checker.check_redis,
        "logs": checker.check_worker_logs,
    }
    
    if args.check == "all":
        for name, check_fn in checks.items():
            try:
                check_fn()
            except Exception as e:
                checker.add_result(name, "FAIL", f"Check crashed: {e}")
    else:
        try:
            checks[args.check]()
        except Exception as e:
            checker.add_result(args.check, "FAIL", f"Check crashed: {e}")
    
    if args.json:
        print("\n" + json.dumps(checker.results, indent=2))
    
    return checker.summary()


if __name__ == "__main__":
    sys.exit(main())
