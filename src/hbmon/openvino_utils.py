"""
OpenVINO utilities for Intel GPU device detection and model conversion.

This module provides helper functions to:
- Detect available OpenVINO devices and validate GPU capability
- Convert CLIP models from PyTorch to OpenVINO IR format
- Load and cache converted models for accelerated inference

Environment Variables:
- HBMON_YOLO_BACKEND: "pytorch" (default), "openvino-cpu", or "openvino-gpu"
- HBMON_DEVICE: "cpu", "cuda", "openvino-cpu", or "openvino-gpu"
- OPENVINO_CACHE_DIR: Directory for caching converted models
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional OpenVINO dependency
# ---------------------------------------------------------------------------
try:
    from openvino import Core  # type: ignore
    _OPENVINO_AVAILABLE = True
except ImportError:  # pragma: no cover
    Core = None  # type: ignore
    _OPENVINO_AVAILABLE = False


def is_openvino_available() -> bool:
    """Check if OpenVINO runtime is installed."""
    return _OPENVINO_AVAILABLE


# Global Core instance to be reused across the application
_CORE: Core | None = None


def get_core() -> Core:
    """
    Get a configured OpenVINO Core instance.
    
    Initializes a shared Core instance if it doesn't exist, and configures
    it with the model cache directory.
    """
    global _CORE
    if not _OPENVINO_AVAILABLE:
        raise RuntimeError("OpenVINO is not installed.")
    
    if _CORE is None:
        import os
        from pathlib import Path
        _CORE = Core()
        
        # Enable model caching for faster subsequent compilations
        cache_dir = os.getenv("OPENVINO_CACHE_DIR")
        if not cache_dir:
            data_dir = os.getenv("HBMON_DATA_DIR", "/data")
            cache_dir = str(Path(data_dir) / "openvino_cache")
        
        # Ensure the base cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # OpenVINO's CACHE_DIR property enables binary model caching
        try:
            _CORE.set_property({"CACHE_DIR": cache_dir})
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to set OpenVINO CACHE_DIR: {e}")
            
    return _CORE


def get_available_openvino_devices() -> list[str]:
    """
    Return list of available OpenVINO devices.

    Returns an empty list if OpenVINO is not installed.
    """
    if not _OPENVINO_AVAILABLE:
        return []
    try:
        core = get_core()
        return list(core.available_devices)
    except Exception:  # pragma: no cover
        return []


def validate_openvino_gpu() -> bool:
    """
    Check if Intel GPU is available for OpenVINO inference.

    Returns True if GPU or GPU.0 is in the available devices list.
    """
    devices = get_available_openvino_devices()
    return any(dev.startswith("GPU") for dev in devices)


def get_recommended_backend() -> str:
    """
    Suggest the best inference backend based on available hardware.

    Returns:
    - "openvino-gpu" if Intel GPU is available
    - "openvino-cpu" if OpenVINO is installed (faster than PyTorch CPU)
    - "pytorch" if OpenVINO is not available
    
    Note: OpenVINO CPU is typically 1.5-2x faster than PyTorch CPU for inference.
    """
    if validate_openvino_gpu():
        return "openvino-gpu"
    if is_openvino_available():
        return "openvino-cpu"
    return "pytorch"


def force_openvino_gpu_override() -> None:
    """
    Monkeypatch OpenVINO Core.compile_model to force GPU execution.
    
    This is required because the ultralytics library defaults to 'AUTO' or 'CPU'
    when using the OpenVINO backend, and its own device selection logic triggers
    errors if we try to explicitly pass device='GPU' (due to missing CUDA).
    
    This patch intercepts the low-level OpenVINO compilation call and redirects
    'AUTO' or 'CPU' requests to 'GPU'.
    """
    if not _OPENVINO_AVAILABLE:
        return

    try:
        from openvino.runtime import Core  # type: ignore
        import openvino.runtime as ov_runtime  # type: ignore
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Avoid double-patching
        if getattr(Core, "_is_hbmon_patched", False):
            return

        original_compile_model = Core.compile_model

        def patched_compile_model(self, model, device_name=None, config=None):
            # If logic requests AUTO (standard for ultralytics) or CPU, upgrade to GPU
            target = "GPU"
            
            # Simple check: if we are forcing GPU, we override generic requests
            if device_name in (None, "AUTO", "CPU"):
                logger.info(f"OpenVINO: Intercepting device='{device_name}' -> forcing '{target}' (Intel GPU)")
                device_name = target
            
            return original_compile_model(self, model, device_name, config)

        # Apply patch to class
        Core.compile_model = patched_compile_model
        Core._is_hbmon_patched = True  # type: ignore
        
        # Also patch the module-level alias if it exists
        if hasattr(ov_runtime, "Core"):
            ov_runtime.Core.compile_model = patched_compile_model

        print("[openvino_utils] Patched OpenVINO to force GPU execution")

    except Exception as e:
        print(f"[openvino_utils] Failed to patch OpenVINO: {e}")

# ---------------------------------------------------------------------------
# CLIP Model Conversion and Loading
# ---------------------------------------------------------------------------

def get_clip_cache_dir() -> str:
    """
    Get the cache directory for OpenVINO CLIP models.
    
    Uses OPENVINO_CACHE_DIR environment variable if set, otherwise uses
    a default location in the data directory.
    """
    import os
    from pathlib import Path
    
    cache_dir = os.getenv("OPENVINO_CACHE_DIR")
    if cache_dir:
        return str(Path(cache_dir) / "clip")
    
    # Fallback to data directory
    data_dir = os.getenv("HBMON_DATA_DIR", "/data")
    return str(Path(data_dir) / "openvino_cache" / "clip")


def get_clip_model_path(model_name: str, pretrained: str) -> tuple[str, str]:
    """
    Get the paths for cached OpenVINO CLIP model files.
    
    Returns:
        Tuple of (xml_path, bin_path) for the OpenVINO IR files
    """
    from pathlib import Path
    
    cache_dir = Path(get_clip_cache_dir())
    # Create a safe filename from model name and pretrained source
    safe_name = f"{model_name}_{pretrained}".replace("/", "_").replace(":", "_")
    
    xml_path = cache_dir / f"{safe_name}.xml"
    bin_path = cache_dir / f"{safe_name}.bin"
    
    return str(xml_path), str(bin_path)


def convert_clip_to_openvino(
    pytorch_model: any,
    model_name: str,
    pretrained: str,
    example_image_input: any,
    example_text_input: any,
) -> tuple[any, any]:
    """
    Convert PyTorch CLIP model to OpenVINO IR format.
    
    Args:
        pytorch_model: The PyTorch CLIP model to convert
        model_name: Name of the CLIP model (e.g., "ViT-B-32")
        pretrained: Pretrained weights source (e.g., "openai")
        example_image_input: Example image tensor for tracing
        example_text_input: Example text tensor for tracing
    
    Returns:
        Tuple of (image_model, text_model) as OpenVINO compiled models
    
    Raises:
        RuntimeError: If OpenVINO is not available or conversion fails
    """
    if not _OPENVINO_AVAILABLE:  # pragma: no cover
        raise RuntimeError("OpenVINO is not installed. Cannot convert CLIP model.")
    
    try:
        from openvino import convert_model, save_model  # type: ignore
        from pathlib import Path
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Get cache paths
        xml_path, bin_path = get_clip_model_path(model_name, pretrained)
        cache_dir = Path(xml_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting CLIP model {model_name} ({pretrained}) to OpenVINO IR...")
        
        import torch

        # Define wrappers to ensure correct tracing of sub-modules
        class VisualWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)

        class TextWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model.encode_text(x)

        # Convert image encoder
        logger.info("Tracing and converting CLIP image encoder...")
        # We manually trace with check_trace=False to avoid issues with some specific
        # OpenCLIP ops diverging during trace validation.
        visual_wrapper = VisualWrapper(pytorch_model.visual)
        traced_visual = torch.jit.trace(visual_wrapper, example_image_input, check_trace=False)
        
        # Pass traced model to convert_model WITHOUT example_input to avoid re-tracing
        image_model_ov = convert_model(traced_visual)
        
        # Convert text encoder  
        logger.info("Tracing and converting CLIP text encoder...")
        text_wrapper = TextWrapper(pytorch_model)
        traced_text = torch.jit.trace(text_wrapper, example_text_input, check_trace=False)
        
        text_model_ov = convert_model(traced_text)
        
        # Save models to cache
        logger.info(f"Saving OpenVINO CLIP models to {cache_dir}...")
        save_model(image_model_ov, str(xml_path).replace(".xml", "_image.xml"))
        save_model(text_model_ov, str(xml_path).replace(".xml", "_text.xml"))
        
        logger.info("CLIP model conversion complete")
        
        return image_model_ov, text_model_ov
        
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to convert CLIP model to OpenVINO: {exc}") from exc


def load_openvino_clip(
    model_name: str,
    pretrained: str,
    device: str = "CPU",
) -> tuple[any, any] | None:
    """
    Load cached OpenVINO CLIP model.
    
    Args:
        model_name: Name of the CLIP model (e.g., "ViT-B-32")
        pretrained: Pretrained weights source (e.g., "openai")
        device: OpenVINO device to use ("CPU", "GPU", "GPU.0", etc.)
    
    Returns:
        Tuple of (image_model, text_model) as compiled OpenVINO models,
        or None if cached model doesn't exist
    
    Raises:
        RuntimeError: If OpenVINO is not available or loading fails
    """
    if not _OPENVINO_AVAILABLE:  # pragma: no cover
        raise RuntimeError("OpenVINO is not installed. Cannot load CLIP model.")
    
    try:
        from pathlib import Path
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Get cache paths
        xml_path, _ = get_clip_model_path(model_name, pretrained)
        image_xml = str(xml_path).replace(".xml", "_image.xml")
        text_xml = str(xml_path).replace(".xml", "_text.xml")
        
        # Check if cached models exist
        if not (Path(image_xml).exists() and Path(text_xml).exists()):
            logger.debug(f"Cached OpenVINO CLIP model not found at {xml_path}")
            return None
        
        logger.info(f"Loading cached OpenVINO CLIP model from {Path(xml_path).parent}...")
        
        # Load and compile models
        core = get_core()
        image_model = core.read_model(image_xml)
        text_model = core.read_model(text_xml)
        
        image_compiled = core.compile_model(image_model, device)
        text_compiled = core.compile_model(text_model, device)
        
        logger.info(f"Loaded OpenVINO CLIP model on device: {device}")
        
        return image_compiled, text_compiled
        
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load OpenVINO CLIP model: {exc}") from exc


def select_clip_device(backend: str) -> str:
    """
    Select the appropriate OpenVINO device based on backend string.
    
    Args:
        backend: Backend string ("openvino-gpu", "openvino-cpu", etc.)
    
    Returns:
        OpenVINO device string ("GPU", "CPU", etc.)
    """
    if "gpu" in backend.lower():
        if validate_openvino_gpu():
            return "GPU"
        # Fallback to CPU if GPU not available
        import logging
        logging.getLogger(__name__).warning(
            "Intel GPU not available for OpenVINO, falling back to CPU"
        )
        return "CPU"
    return "CPU"
