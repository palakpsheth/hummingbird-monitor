
import logging
import shutil
import subprocess
import time
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError
import json

import psutil

# Import cache functions for GPU stats fallback
try:
    from hbmon.cache import cache_set_json_sync, cache_get_json_sync
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    cache_set_json_sync = None  # type: ignore
    cache_get_json_sync = None  # type: ignore

logger = logging.getLogger(__name__)


def get_system_stats() -> dict:
    """Get current system stats: CPU, memory, and GPU loads.
    
    Returns dict with keys: cpu, mem, gpu_intel, gpu_nvidia.
    GPU fields are None if the corresponding GPU tools are not available.
    """
    cpu_pct = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    mem_pct = mem.percent
    
    intel_load = get_intel_gpu_load() if shutil.which("intel_gpu_top") else None
    nv_load = get_nvidia_gpu_load() if shutil.which("nvidia-smi") else None
    
    # Hybrid approach: Check Redis cache for GPU stats from worker
    # The worker container sees actual GPU usage better than web container
    if _CACHE_AVAILABLE:
        try:
            cached_gpu = cache_get_json_sync("gpu_stats_cache", {}) or {}
            cached_intel = cached_gpu.get("intel")
            cached_nv = cached_gpu.get("nvidia")
            
            # Use max of local vs cached, or cached if local is None/0
            if cached_intel is not None:
                intel_load = max(intel_load or 0.0, cached_intel)
            if cached_nv is not None:
                nv_load = max(nv_load or 0.0, cached_nv)
        except Exception:
            pass
    
    return {
        "cpu": cpu_pct,
        "mem": mem_pct,
        "gpu_intel": intel_load,
        "gpu_nvidia": nv_load
    }


def get_intel_gpu_load() -> Optional[float]:
    """
    Attempt to get Intel GPU load using intel_gpu_top.
    Returns percentage (0-100) or None if unavailable/failed.
    
    Checks multiple engine classes relevant to compute workloads:
    - Render/3D (traditional 3D rendering)
    - Compute (OpenCL/compute shaders, may be used by OpenVINO)
    - Video/VideoEnhance (media engines)
    """
    if not shutil.which("intel_gpu_top"):
        return None
    # Also need timeout command
    if not shutil.which("timeout"):
        return None
    try:
        # Use shell timeout to kill intel_gpu_top after 5 seconds to capture more samples
        # intel_gpu_top runs continuously, so we need to force-kill it
        # -J: JSON output
        # -s 100: 100ms sample period (better for capturing bursty inference load)
        result = subprocess.run(
            ["timeout", "5", "intel_gpu_top", "-J", "-s", "100"],
            capture_output=True,
            text=True,
            timeout=7,  # Python timeout as backup
        )
        # timeout command returns 124 when it kills the process
        if not result.stdout or not result.stdout.strip():
            return 0.0

        if result.stdout.strip():
            import json
            import re
            
            # When killed with timeout, JSON may be incomplete
            # Try to fix common issues: missing closing brackets, trailing commas
            output = result.stdout.strip()
            
            # intel_gpu_top outputs multiple JSON objects
            # They're separated by },\n{ or }\n{ (comma may or may not be present)
            # Use distinct token to split, preserving the braces
            safe_output = re.sub(r'\}\s*,?\s*\n\s*\{', '}|||{', output)
            json_objects = safe_output.split('|||')
            
            # Need at least 1 object
            if json_objects:
                max_busy_overall = 0.0
                found_valid_data = False

                for obj_str in json_objects:
                    obj_str = obj_str.strip()
                    if not obj_str:
                        continue
                        
                    # Fix braces after split
                    if not obj_str.startswith('{'):
                        obj_str = '{' + obj_str
                    if not obj_str.endswith('}'):
                        # Handle incomplete JSON from timeout kill
                        obj_str = obj_str.rstrip(',\n\t ')
                        # Count braces to add missing closing ones
                        open_braces = obj_str.count('{') - obj_str.count('}')
                        obj_str += '}' * max(0, open_braces)
                    
                    try:
                        data = json.loads(obj_str)
                        
                        # 1. Check CLIENTS (per-process usage)
                        clients = data.get("clients", {})
                        for client in clients.values():
                            engine_classes = client.get("engine-classes", {})
                            for engine in engine_classes.values():
                                busy = engine.get("busy")
                                if busy is not None:
                                    max_busy_overall = max(max_busy_overall, float(busy))
                                    found_valid_data = True

                        # 2. Check ENGINES (system-wide)
                        engines = data.get("engines", {})
                        for engine in engines.values():
                            busy = engine.get("busy")
                            if busy is not None:
                                max_busy_overall = max(max_busy_overall, float(busy))
                                found_valid_data = True
                                
                    except (json.JSONDecodeError, ValueError):
                        continue
                
                if found_valid_data:
                    return max_busy_overall
    except Exception:
        pass
    return None


def get_nvidia_gpu_load() -> Optional[float]:
    """
    Attempt to get Nvidia GPU utilization using nvidia-smi.
    Returns percentage (0-100) or None if unavailable/failed.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


_LAST_LOG_TIME = 0.0


def log_system_stats_from_api(api_url: str = "http://hbmon-web:8000/api/system_load") -> None:
    """
    Fetch system stats from the web API and log them.
    
    Throttles to once every 60 seconds to avoid noise.
    Used by worker/annotator containers to log system load periodically.
    """
    global _LAST_LOG_TIME
    now = time.time()
    
    # Throttle: only log every 60 seconds
    if now - _LAST_LOG_TIME < 60:
        return
    _LAST_LOG_TIME = now
    
    try:
        with urlopen(api_url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        stats = [f"CPU: {data['cpu']:.1f}%", f"Mem: {data['mem']:.1f}%"]
        if data.get('gpu_intel') is not None:
            stats.append(f"Intel GPU: {data['gpu_intel']:.1f}%")
        if data.get('gpu_nvidia') is not None:
            stats.append(f"Nvidia GPU: {data['gpu_nvidia']:.1f}%")
        
        logger.info("System Load: " + " | ".join(stats))
    except (URLError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to fetch system stats from API: {e}")
    except Exception as e:
        logger.debug(f"Error logging system stats: {e}")


_LAST_CACHE_TIME = 0.0

def cache_gpu_stats() -> None:
    """
    Fetch local GPU stats and cache them in Redis.
    Called by worker/annotator containers which have better visibility into GPU usage.
    Throttles to once every 10 seconds.
    """
    if not _CACHE_AVAILABLE:
        return

    global _LAST_CACHE_TIME
    now = time.time()
    if now - _LAST_CACHE_TIME < 10:
        return
    _LAST_CACHE_TIME = now
        
    try:
        intel_load = get_intel_gpu_load() if shutil.which("intel_gpu_top") else None
        nv_load = get_nvidia_gpu_load() if shutil.which("nvidia-smi") else None
        
        if intel_load is not None or nv_load is not None:
            stats = {
                "intel": intel_load,
                "nvidia": nv_load,
                "timestamp": time.time()
            }
            # expire in 15 seconds so we don't show stale data if worker dies
            if cache_set_json_sync("gpu_stats_cache", stats, ttl_seconds=15):
                # Log only when we actually see load (reduces spam)
                if intel_load and intel_load > 0:
                    logger.info(f"Pushed GPU stats to Redis: Intel={intel_load}%")
    except Exception as e:
        logger.error(f"Failed to cache GPU stats: {e}")
