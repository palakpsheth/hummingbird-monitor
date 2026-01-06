import urllib.request
import urllib.error
import urllib.parse
import time
import json
import itertools

# Configuration
BASE_URL = "http://localhost:8000"
OBS_ID = 53
FRAME_ID = 15579
POINTS = [{"x": 0.5, "y": 0.5}]

# Parameter ranges
CONFIDENCE_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25]
CROP_SIZES = [640, 800, 1024, 1280]

def make_request(url, method="GET", data=None):
    try:
        req = urllib.request.Request(url, method=method)
        req.add_header('Content-Type', 'application/json')
        if data:
            json_data = json.dumps(data).encode('utf-8')
            req.data = json_data
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        print(f"Request Error: {e}")
        return None

def check_job_status(job_id):
    max_retries = 20
    for _ in range(max_retries):
        data = make_request(f"{BASE_URL}/api/annotate/wand_job/{job_id}")
        if not data:
            time.sleep(1)
            continue
            
        status = data.get("status")
        if status == "finished":
            return data.get("boxes", [])
        elif status == "failed":
            print(f"Job {job_id} failed: {data.get('error')}")
            return None
        
        time.sleep(1)
    return None

def run_test():
    print(f"Starting tuning for Frame {FRAME_ID} (Obs {OBS_ID})...")
    
    results = []
    
    for conf, crop in itertools.product(CONFIDENCE_LEVELS, CROP_SIZES):
        print(f"Testing Conf={conf}, Crop={crop}...")
        
        payload = {
            "x": 0.5, 
            "y": 0.5,
            "confidence": conf,
            "crop_size": crop
        }
        
        data = make_request(
            f"{BASE_URL}/api/annotate/{OBS_ID}/frame/{FRAME_ID}/detect_region",
            method="POST",
            data=payload
        )
        
        if not data:
            continue
            
        job_id = data.get("job_id")
        
        if not job_id:
            print("No job ID returned")
            continue
            
        boxes = check_job_status(job_id)
        
        box_count = len(boxes) if boxes is not None else 0
        print(f"-> Found {box_count} boxes (Conf: {conf}, Crop: {crop})")
        
        results.append({
            "conf": conf,
            "crop": crop,
            "boxes": box_count,
            "details": boxes
        })
        
        # If we find something, maybe we don't need to try *everything*? 
        # But user asked to try a variety.
        # Let's keep going.
            
    # Summary
    print("\n=== Tuning Results ===")
    found_any = False
    for r in results:
        if r["boxes"] > 0:
            found_any = True
            print(f"SUCCESS: Conf={r['conf']}, Crop={r['crop']} -> {r['boxes']} boxes")
            
    if not found_any:
        print("No boxes found with any combination.")

if __name__ == "__main__":
    run_test()
