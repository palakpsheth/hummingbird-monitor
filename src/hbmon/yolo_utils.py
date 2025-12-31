import numpy as np

def resolve_predict_imgsz(imgsz_env: str, frame_shape: tuple[int, int] | None = None) -> list[int]:
    """
    Resolves the YOLO prediction image size based on environment variable and frame shape.
    
    Args:
        imgsz_env: The raw environment variable string (e.g. "auto", "1088,1920", "640").
        frame_shape: The (H, W) of the input frame, required for "auto" mode.
        
    Returns:
        A list of [H, W] or [sz, sz] for YOLO imgsz parameter.
    """
    if imgsz_env.lower() == "auto":
        if frame_shape is None:
            # Fallback if no frame shape provided for auto
            return [1088, 1920]
        # Snap height/width to nearest stride 32
        h, w = frame_shape[:2]
        target_h = int(np.ceil(h / 32) * 32)
        target_w = int(np.ceil(w / 32) * 32)
        return [target_h, target_w]
    
    # Parse H,W directly
    try:
        parts = imgsz_env.split(",")
        parsed = [int(p) for p in parts if p.strip()]
        if not parsed:
            return [1088, 1920]
        return parsed
    except ValueError:
        # Fallback if parse fails
        return [1088, 1920]
