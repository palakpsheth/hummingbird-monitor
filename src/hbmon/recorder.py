import cv2
import threading
import queue
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BackgroundRecorder:
    """
    Writes video frames to disk in a background thread to avoid blocking
    the main processing loop.
    
    Videos are stored uncompressed to preserve quality for ML training.
    Compression is applied on-the-fly during streaming (see web.py).
    """
    def __init__(self, out_path: Path, fps: float, width: int, height: int):
        self.out_path = out_path
        self.fps = fps
        self.width = width
        self.height = height
        self.queue = queue.Queue()
        self.stopped = False
        self.writer = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error = None
        
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
            
            for codec, ext in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                
                logger.debug(f"Attempting to open VideoWriter with codec: {codec}")
                try:
                    self.writer = cv2.VideoWriter(
                        str(self.out_path), 
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

            # Processing loop - write uncompressed frames
            while True:
                frame = self.queue.get()
                if frame is None:
                    # Sentinel received, draining done
                    break
                
                self.writer.write(frame)
                self.queue.task_done()
                
            self.writer.release()
            logger.info(f"Uncompressed video saved to {self.out_path}")

        except Exception as e:
            self.error = str(e)
            logger.error(f"BackgroundRecorder error: {e}", exc_info=True)
