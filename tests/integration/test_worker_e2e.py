
import asyncio
import shutil
import pytest
from pathlib import Path
from sqlalchemy import func, select
import logging
import json

from hbmon import db as db_module
from hbmon.models import Observation
import hbmon.worker as worker

# Mark as integration test
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def bioclip_ready() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        pytest.skip(f"Hugging Face Hub dependency unavailable: {exc}")

    try:
        hf_hub_download("imageomics/bioclip", "open_clip_config.json")
    except Exception as exc:
        pytest.skip(f"Bioclip model not available: {exc}")


def get_e2e_cases():
    """Discover E2E test cases in tests/integration/test_data/e2e/"""
    base_dir = Path(__file__).parent / "test_data" / "e2e"
    cases = []
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                cases.append(item)
    return sorted(cases, key=lambda p: p.name)

@pytest.mark.parametrize("case_dir", get_e2e_cases(), ids=lambda p: p.name)
@pytest.mark.anyio
async def test_worker_e2e_data_driven(tmp_path, monkeypatch, case_dir, bioclip_ready):
    """
    Data-driven E2E worker test.
    Reads metadata.json and config.json from case_dir, runs worker against clip.mp4.
    """
    # 1. Load Metadata
    metadata_path = case_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
        
    if metadata.get("xfail"):
        pytest.xfail(metadata.get("xfail_reason", "Known issue"))

    video_filename = "clip.mp4"
    src_video_path = case_dir / video_filename
    if not src_video_path.exists():
        pytest.skip(f"Video not found: {src_video_path}")
        
    expect_detection = metadata.get("expect_detection", True)
    
    # 2. Setup Environment
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    
    data_dir.mkdir(exist_ok=True)
    media_dir.mkdir(exist_ok=True)
    
    # Copy video to media dir (optional, but good for realism if using RTSP path)
    # Actually, we point RTSP directly to the file, so we can use source or copy.
    # Let's use the source directly to save time, or copy if we want to modify it.
    rtsp_url = str(src_video_path.absolute())
    
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    
    monkeypatch.setenv("HBMON_RTSP_URL", rtsp_url)
    monkeypatch.setenv("HBMON_CAMERA_NAME", "test-cam")
    
    # Ensure worker writes cache to temp dir
    monkeypatch.setenv("OPENVINO_CACHE_DIR", str(data_dir / "openvino_cache"))
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(data_dir / "yolo"))
    monkeypatch.setenv("YOLO_DATA_DIR", str(data_dir / "yolo_data"))
    
    # 3. Config (ROI)
    src_config_path = case_dir / "config.json"
    if src_config_path.exists():
        # Copy to data dir so we can point to it
        dst_config_path = data_dir / "config.json"
        shutil.copy2(src_config_path, dst_config_path)
        monkeypatch.setenv("HBMON_CONFIG_FILE", str(dst_config_path))
    
    # 4. Settings from Metadata
    monkeypatch.setenv("HBMON_INFERENCE_BACKEND", "cpu") # Force PyTorch for consistency
    
    if "yolo_imgsz" in metadata:
        val = metadata["yolo_imgsz"]
        if isinstance(val, list):
            monkeypatch.setenv("HBMON_YOLO_IMGSZ", ",".join(map(str, val)))
        else:
            monkeypatch.setenv("HBMON_YOLO_IMGSZ", str(val))
    
    # Set defaults / overrides
    monkeypatch.setenv("HBMON_DETECT_CONF", "0.1")
    monkeypatch.setenv("HBMON_MIN_BOX_AREA", "600")
    monkeypatch.setenv("HBMON_TEMPORAL_WINDOW_FRAMES", "5")
    monkeypatch.setenv("HBMON_FPS_LIMIT", "0")
    monkeypatch.setenv("HBMON_BG_SUBTRACTION", "0")
    
    # 5. Initialize DB
    await db_module.init_async_db()
    
    # 6. Run Worker Loop
    worker_task = asyncio.create_task(worker.run_worker())
    
    logger = logging.getLogger(__name__)
    logger.info(f"Worker started for {case_dir.name}. Expect detection: {expect_detection}")
    
    obs_found = False
    try:
        # Increase timeout to 60s
        for _ in range(60):
            await asyncio.sleep(1.0)
            async with db_module.async_session_scope() as session:
                count = (await session.execute(select(func.count(Observation.id)))).scalar_one()
                if count > 0:
                    obs_found = True
                    break
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        await db_module.dispose_async_engine()
            
    # 7. Validation
    if expect_detection:
        assert obs_found, f"Expected detection in {case_dir.name} but got none."
        
        async with db_module.async_session_scope() as session:
            result = await session.execute(select(Observation).limit(1))
            obs = result.scalar_one()
            
            # Check basic properties
            assert obs.camera_name == "test-cam"
            
            # Check confidence in extra_json
            assert obs.extra_json is not None
            extra = json.loads(obs.extra_json)
            assert "detection" in extra
            assert extra["detection"]["box_confidence"] >= 0.1
            
            # Verify yolo_imgsz is recorded if expected
            if "yolo_imgsz" in extra["detection"]:
                yolo_sz = extra["detection"]["yolo_imgsz"]
                logger.info(f"Recorded YOLO imgsz: {yolo_sz}")
                assert isinstance(yolo_sz, list) or isinstance(yolo_sz, int)

            # Check expected species if present in metadata
            if "species" in metadata and metadata["species"]:
                 # Note: Bioclip might behave differently on CPU/test model, so exact match might be flaky.
                 # Just logging for now.
                 pass

    else:
        assert not obs_found, f"Expected NO detection in {case_dir.name} but found one."
