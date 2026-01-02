import cv2
import threading
import queue
from pathlib import Path
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

class BackgroundRecorder:
    """
    Writes video frames to disk in a background thread to avoid blocking
    the main processing loop.
    
    Supports optional post-processing compression via FFmpeg for significant
    file size reduction without major quality loss.
    """
    def __init__(self, out_path: Path, fps: float, width: int, height: int, compress: bool = True):
        self.out_path = out_path
        self.fps = fps
        self.width = width
        self.height = height
        self.compress = compress
        self.queue = queue.Queue()
        self.stopped = False
        self.writer = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error = None
        
        # Temporary path for uncompressed video (if compression enabled)
        if compress:
            self.temp_path = out_path.parent / f"{out_path.stem}_temp{out_path.suffix}"
        else:
            self.temp_path = out_path
        
        # Ensure directory exists
        if not self.out_path.parent.exists():
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the background writing thread."""
        self.thread.start()

    def feed(self, frame):
        """Queue a frame for writing."""
        if not self.stopped:
            self.queue.put(frame)

    def stop(self):
        """Stop receiving frames and wait for the queue to drain."""
        self.stopped = True
        self.queue.put(None) # Sentinel to signal exit
        self.thread.join()
    
    def _compress_with_ffmpeg(self, input_path: Path, output_path: Path) -> bool:
        """
        Compress video using FFmpeg with H.264 codec and optimized settings.
        
        Settings optimized for hummingbird footage:
        - CRF 23: Good quality/size balance (18=high quality, 28=smaller files)
        - preset medium: Good compression/speed tradeoff
        - H.264 baseline profile: Maximum browser compatibility
        - YUV420p pixel format: Required for browser playback
        
        Returns True if successful, False otherwise.
        """
        ffmpeg_path = os.getenv("HBMON_FFMPEG_PATH", "ffmpeg")
        
        # CRF (Constant Rate Factor): Lower = better quality, larger file
        # 23 is default, good balance. Range: 18 (high quality) to 28 (smaller)
        crf = int(os.getenv("HBMON_VIDEO_CRF", "23"))
        
        # Preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        # Slower presets = better compression, more CPU time
        preset = os.getenv("HBMON_VIDEO_PRESET", "medium")
        
        try:
            cmd = [
                ffmpeg_path,
                "-i", str(input_path),
                "-c:v", "libx264",           # H.264 codec
                "-crf", str(crf),            # Quality setting
                "-preset", preset,            # Encoding speed/compression tradeoff  
                "-profile:v", "baseline",     # Browser compatibility
                "-pix_fmt", "yuv420p",       # Required for browser playback
                "-movflags", "+faststart",   # Enable streaming (move moov atom to start)
                "-y",                        # Overwrite output
                str(output_path)
            ]
            
            logger.debug(f"Compressing video with FFmpeg: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode == 0:
                # Get file sizes for logging
                original_size = input_path.stat().st_size / (1024 * 1024)  # MB
                compressed_size = output_path.stat().st_size / (1024 * 1024)  # MB
                reduction = ((original_size - compressed_size) / original_size) * 100
                
                logger.info(
                    f"Video compressed: {original_size:.2f}MB → {compressed_size:.2f}MB "
                    f"({reduction:.1f}% reduction)"
                )
                return True
            else:
                logger.error(f"FFmpeg compression failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning(f"FFmpeg not found at {ffmpeg_path}. Skipping compression.")
            return False
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg compression timed out")
            return False
        except Exception as e:
            logger.error(f"FFmpeg compression error: {e}")
            return False

    def _run(self):
        try:
            # Try to initialize VideoWriter with various codecs
            # Ideally use avc1 (H.264) for browser compatibility
            codecs = [
                ("avc1", ".mp4"),
                ("H264", ".mp4"),
                ("mp4v", ".mp4"), # Fallback (might not play in browser)
                ("XVID", ".avi"), # Last resort
            ]
            
            
            # Ensure out_path has correct suffix for the codec (though calling code usually handles this)
            # We strictly respect the out_path suffix if provided, but we iterate codecs compatible with it?
            # Actually, let's just try to open the writer.
            
            for codec, ext in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                # Note: If out_path suffix conflicts, the writer might fail or produce invalid file.
                # Assuming out_path is .mp4 for now.
                
                logger.debug(f"Attempting to open VideoWriter with codec: {codec}")
                try:
                    self.writer = cv2.VideoWriter(
                        str(self.temp_path), 
                        fourcc, 
                        self.fps, 
                        (self.width, self.height)
                    )
                    if self.writer.isOpened():
                        logger.info(f"Initialized VideoWriter with codec: {codec}")
                        break
                    else:
                        if self.writer:
                            self.writer.release()
                except Exception as e:
                    logger.warning(f"Failed VideoWriter init with {codec}: {e}")

            if not self.writer or not self.writer.isOpened():
                self.error = "Failed to initialize any compatible VideoWriter"
                logger.error(self.error)
                return

            # Processing loop
            while True:
                frame = self.queue.get()
                if frame is None:
                    # Sentinel received, draining done (implicit since we consume strictly FIFO)
                    break
                
                self.writer.write(frame)
                self.queue.task_done()
                
            self.writer.release()
            logger.info(f"Video saved to {self.temp_path}")
            
            # Post-process with FFmpeg for compression if enabled
            if self.compress and self.temp_path != self.out_path:
                logger.info("Post-processing video with FFmpeg for compression...")
                if self._compress_with_ffmpeg(self.temp_path, self.out_path):
                    # Compression successful, remove temp file
                    try:
                        self.temp_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temp video file: {e}")
                else:
                    # Compression failed, use the uncompressed version
                    logger.warning("FFmpeg compression failed, using uncompressed video")
                    try:
                        self.temp_path.rename(self.out_path)
                    except Exception as e:
                        logger.error(f"Failed to rename temp video: {e}")

        except Exception as e:
            self.error = str(e)
            logger.error(f"BackgroundRecorder error: {e}", exc_info=True)
