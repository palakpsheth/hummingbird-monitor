import os

# --- Monkeypatch openvino.runtime.Core.compile_model ---
try:
    # Try importing from the location Ultralytics uses
    import openvino.runtime as ov_runtime
    from openvino.runtime import Core

    original_compile_model = Core.compile_model

    def patched_compile_model(self, model, device_name=None, config=None):
        print(f"[DEBUG-PATCH] compile_model called with device_name='{device_name}'")
        
        # Force GPU if available and requested conceptual 'GPU' override
        # In a real script we would check env vars, here we just force it for test
        if device_name == 'CPU': 
             print("[DEBUG-PATCH] Intercepting CPU request -> forcing GPU")
             device_name = 'GPU'
        
        return original_compile_model(self, model, device_name, config)

    # Apply patch
    Core.compile_model = patched_compile_model
    ov_runtime.Core.compile_model = patched_compile_model
    print("[DEBUG] Applied Core.compile_model monkeypatch")
    
except ImportError:
    print("OpenVINO not installed, cannot patch")

# ---------------------------------

try:
    from ultralytics import YOLO
    model_path = "/data/openvino_cache/yolo/yolo11n_openvino_model"
    if os.path.exists(model_path):
        print(f"Loading YOLO from {model_path}...")
        model = YOLO(model_path)
        
        # Create a dummy image
        import numpy as np
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("Running inference (default device)...")
        try:
            # We let it default to CPU (or pass device='cpu')
            # Ultralytics will tell OpenVINO "CPU"
            # Our patch should swap it to "GPU"
            results = model.predict(img, device='cpu')
            print("Inference completed.")
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"Model path not found: {model_path}")

except Exception as e:
    print(f"YOLO check failed: {e}")
