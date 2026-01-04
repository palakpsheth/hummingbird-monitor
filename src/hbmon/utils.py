
import logging
import shutil
import subprocess
import time
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError
import json

import psutil

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
    
    return {
        "cpu": cpu_pct,
        "mem": mem_pct,
        "gpu_intel": intel_load,
        "gpu_nvidia": nv_load
    }


def get_intel_gpu_load() -> Optional[float]:
    """
    Attempt to get Intel GPU render load using intel_gpu_top.
    Returns percentage (0-100) or None if unavailable/failed.
    """
    if not shutil.which("intel_gpu_top"):
        return None
    try:
        result = subprocess.run(
            ["intel_gpu_top", "-J", "-s", "500"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            import json
            import re
            # intel_gpu_top outputs multiple JSON objects, take the last complete one
            # Split by },{ pattern but keep the braces
            json_objects = re.split(r'\}\s*,\s*\{', result.stdout.strip())
            if json_objects:
                # Get last object and fix braces
                last_obj = json_objects[-1]
                if not last_obj.startswith('{'):
                    last_obj = '{' + last_obj
                if not last_obj.endswith('}'):
                    last_obj = last_obj.rstrip(',\n ') + '}'
                try:
                    data = json.loads(last_obj)
                    # First try clients section (more accurate per-process usage)
                    clients = data.get("clients", {})
                    for client in clients.values():
                        engine_classes = client.get("engine-classes", {})
                        render = engine_classes.get("Render/3D", {})
                        busy = render.get("busy")
                        if busy:
                            return float(busy)
                    # Fallback to engines section
                    engines = data.get("engines", {})
                    render = engines.get("Render/3D/0", {})
                    busy = render.get("busy", 0.0)
                    return float(busy) if busy else None
                except (json.JSONDecodeError, ValueError):
                    pass
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
        with urlopen(api_url, timeout=5) as resp:
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
