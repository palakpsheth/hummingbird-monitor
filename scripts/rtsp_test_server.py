#!/usr/bin/env python3
"""
rtsp_test_server.py

Serve video files as a mock RTSP stream using FFmpeg for detection pipeline testing.
This allows you to replay known bird observation videos through the worker.

Usage:
    # Serve a single video on loop
    uv run python scripts/rtsp_test_server.py data/media/clips/20260102/xxxxx.mp4

    # Serve with specific port
    uv run python scripts/rtsp_test_server.py --port 8555 /path/to/video.mp4

    # Serve multiple videos in sequence
    uv run python scripts/rtsp_test_server.py video1.mp4 video2.mp4 video3.mp4

    # Use with extract_test_videos.py:
    uv run python scripts/rtsp_test_server.py $(uv run python scripts/extract_test_videos.py --paths-only -f --limit 3)

Requirements:
    - FFmpeg with RTSP server capability (rtsp output format)
    
Note:
    Update HBMON_RTSP_URL in .env to point to the test stream:
    HBMON_RTSP_URL=rtsp://localhost:8555/test
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def find_ffmpeg() -> str:
    """Find FFmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: FFmpeg not found on PATH.", file=sys.stderr)
        print("Install FFmpeg:", file=sys.stderr)
        print("  Ubuntu: sudo apt-get install ffmpeg", file=sys.stderr)
        print("  macOS:  brew install ffmpeg", file=sys.stderr)
        sys.exit(1)
    return ffmpeg


def create_concat_file(videos: list[Path], temp_dir: Path) -> Path:
    """Create FFmpeg concat demuxer file for multiple videos."""
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for video in videos:
            # FFmpeg concat requires escaped paths
            escaped = str(video.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    return concat_file


def start_mediamtx_server(port: int) -> subprocess.Popen | None:
    """Start mediamtx RTSP server container if docker is available."""
    docker = shutil.which("docker")
    if not docker:
        return None
    
    # Stop any existing container
    subprocess.run(
        [docker, "rm", "-f", "hbmon-rtsp-test"],
        capture_output=True,
    )
    
    # Start mediamtx container
    cmd = [
        docker, "run", "--rm", "-d",
        "--name", "hbmon-rtsp-test",
        "-p", f"{port}:8554",
        "bluenviron/mediamtx:latest",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Could not start mediamtx: {result.stderr}", file=sys.stderr)
        return None
    
    # Wait for server to be ready
    time.sleep(2)
    
    # Return a dummy Popen that we can check
    return subprocess.Popen(["sleep", "infinity"])


def stop_mediamtx_server():
    """Stop mediamtx container."""
    docker = shutil.which("docker")
    if docker:
        subprocess.run([docker, "rm", "-f", "hbmon-rtsp-test"], capture_output=True)


def start_rtsp_server(
    ffmpeg: str,
    videos: list[Path],
    port: int,
    stream_name: str,
    loop: bool = True,
    fps: int = 20,
) -> tuple[subprocess.Popen, subprocess.Popen | None]:
    """Start FFmpeg to push to RTSP server.
    
    Returns (ffmpeg_proc, mediamtx_proc) where mediamtx_proc may be None.
    """
    
    # Start mediamtx first
    mediamtx_proc = start_mediamtx_server(port)
    if mediamtx_proc is None:
        print("ERROR: Could not start RTSP server.", file=sys.stderr)
        print("Make sure docker is installed and running.", file=sys.stderr)
        sys.exit(1)
    
    # ffmpeg pushes to mediamtx
    rtsp_url = f"rtsp://localhost:{port}/{stream_name}"
    
    if len(videos) == 1:
        # Single video mode - try to copy stream directly for best quality
        input_args = ["-re", "-stream_loop", "-1" if loop else "0", "-i", str(videos[0])]
        # Check if video is h264, if so use copy, otherwise transcode
        # For now, let's assume input might need transcoding for RTSP if not already H.264
        # But user requested "raw rtsp stream" mimicry.
        # Let's use high-quality re-encoding to be safe on format but lossless-ish
        codec_args = [
             "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",  # Visually lossless
            "-tune", "zerolatency",
            "-g", str(fps * 2),
            "-r", str(fps),
        ]
    else:
        # Multiple videos - use concat demuxer
        temp_dir = Path(tempfile.mkdtemp(prefix="rtsp_test_"))
        concat_file = create_concat_file(videos, temp_dir)
        loop_opts = ["-stream_loop", "-1"] if loop else []
        input_args = ["-re", *loop_opts, "-f", "concat", "-safe", "0", "-i", str(concat_file)]
        codec_args = [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",
            "-tune", "zerolatency",
            "-g", str(fps * 2),
            "-r", str(fps),
        ]
    
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "warning",
        *input_args,
        *codec_args,
        "-pix_fmt", "yuv420p",
        # Audio handling (drop if present)
        "-an",
        # RTSP output to mediamtx
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]
    
    print("Starting RTSP server...")
    print(f"  Stream URL: rtsp://0.0.0.0:{port}/{stream_name}")
    print(f"  Videos: {len(videos)}")
    print(f"  Loop: {loop}")
    print(f"  FPS: {fps}")
    print()
    print("To test with VLC or ffplay:")
    print(f"  ffplay rtsp://localhost:{port}/{stream_name}")
    print()
    print("To use with hbmon-worker, update .env:")
    print(f"  HBMON_RTSP_URL=rtsp://172.17.0.1:{port}/{stream_name}")
    print()
    print("Press Ctrl+C to stop...")
    print("-" * 60)
    
    # Start FFmpeg
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    return proc, mediamtx_proc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Serve video files as a mock RTSP stream for testing."
    )
    ap.add_argument(
        "videos",
        nargs="+",
        help="Video file(s) to serve"
    )
    ap.add_argument(
        "--port", "-p",
        type=int,
        default=8555,
        help="RTSP server port (default: 8555)"
    )
    ap.add_argument(
        "--stream-name", "-s",
        type=str,
        default="test",
        help="RTSP stream name (default: test)"
    )
    ap.add_argument(
        "--no-loop",
        action="store_true",
        help="Don't loop videos (exit after playback)"
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output frame rate (default: 20)"
    )
    args = ap.parse_args()

    # Validate videos
    videos: list[Path] = []
    for v in args.videos:
        path = Path(v)
        if not path.exists():
            print(f"ERROR: Video not found: {path}", file=sys.stderr)
            return 1
        videos.append(path)
    
    if not videos:
        print("ERROR: No videos specified", file=sys.stderr)
        return 1
    
    ffmpeg = find_ffmpeg()
    
    # Set up signal handler for graceful shutdown
    proc: subprocess.Popen | None = None
    mediamtx_proc: subprocess.Popen | None = None
    
    def signal_handler(sig, frame):
        print("\nShutting down...")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if mediamtx_proc:
            mediamtx_proc.terminate()
        stop_mediamtx_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    proc, mediamtx_proc = start_rtsp_server(
        ffmpeg=ffmpeg,
        videos=videos,
        port=args.port,
        stream_name=args.stream_name,
        loop=not args.no_loop,
        fps=args.fps,
    )
    
    # Monitor process
    try:
        while proc.poll() is None:
            # Read and print stderr (FFmpeg logs)
            if proc.stderr:
                line = proc.stderr.readline()
                if line:
                    print(f"[ffmpeg] {line.rstrip()}")
            time.sleep(0.1)
        
        # Process exited
        returncode = proc.returncode
        if returncode != 0:
            print(f"\nFFmpeg exited with code {returncode}", file=sys.stderr)
            if proc.stderr:
                remaining = proc.stderr.read()
                if remaining:
                    print(remaining, file=sys.stderr)
            stop_mediamtx_server()
            return returncode
            
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    finally:
        stop_mediamtx_server()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
