#!/usr/bin/env python3
"""
rtsp_diagnostics.py

RTSP stream diagnostics:
1) Raw RTSP handshake probe (OPTIONS + DESCRIBE) via socket
2) ffprobe JSON summary (streams/format) + optional frame count

Usage:
  uv run python scripts/rtsp_diagnostics.py --seconds 10 rtsp://192.168.1.31:554/stream0
  uv run python scripts/rtsp_diagnostics.py --transport tcp --loglevel warning rtsp://...
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _run(cmd: list[str], timeout_s: float | None = None) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout_s,
    )
    return p.returncode, p.stdout, p.stderr


def sanitize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if parts.username is None:
            return url
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        userinfo = parts.username
        if parts.password is not None:
            userinfo += ":***"
        netloc = f"{userinfo}@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return url


def detect_ffprobe() -> str:
    fp = shutil.which("ffprobe")
    if not fp:
        raise SystemExit(
            "ERROR: ffprobe not found on PATH.\n"
            "Install FFmpeg (includes ffprobe):\n"
            "  Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
            "  macOS:  brew install ffmpeg\n"
        )
    return fp


def ffprobe_version(ffprobe: str) -> str:
    rc, out, err = _run([ffprobe, "-version"])
    if rc != 0:
        return f"(failed)\n{err.strip()}"
    return out.strip().splitlines()[0] if out.strip() else "(unknown)"


def detect_supported_timeout_option(ffprobe: str) -> str | None:
    rc, out, err = _run([ffprobe, "-h", "full"])
    text = out + "\n" + err
    for opt in ("rw_timeout", "stimeout", "timeout"):
        if re.search(rf"(?m)^\s*-{re.escape(opt)}\b", text):
            return opt
    return None


def build_ffprobe_cmd(
    ffprobe: str,
    url: str,
    transport: str,
    timeout_us: int,
    timeout_opt: str | None,
    loglevel: str,
) -> list[str]:
    cmd = [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_programs",
        "-show_error",
        "-analyzeduration",
        "5000000",
        "-probesize",
        "5000000",
        "-rtsp_transport",
        transport,
    ]
    if timeout_opt:
        cmd += [f"-{timeout_opt}", str(timeout_us)]
    cmd += [url]
    return cmd


def parse_json_maybe(s: str) -> dict[str, Any]:
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def fmt_rate(rate: str | None) -> str:
    if not rate or rate in ("0/0", "N/A"):
        return "N/A"
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", rate)
    if not m:
        return rate
    num = int(m.group(1))
    den = int(m.group(2))
    if den == 0:
        return rate
    return f"{rate} ({num/den:.3f} fps)"


def summarize(probe: dict[str, Any]) -> str:
    lines: list[str] = []
    fmt = probe.get("format", {}) or {}
    streams = probe.get("streams", []) or []

    if fmt:
        lines.append("Format:")
        lines.append(f"  format_name:      {fmt.get('format_name', 'N/A')}")
        lines.append(f"  format_long_name: {fmt.get('format_long_name', 'N/A')}")
        lines.append(f"  start_time:       {fmt.get('start_time', 'N/A')}")
        lines.append(f"  duration:         {fmt.get('duration', 'N/A')}")
        lines.append(f"  bit_rate:         {fmt.get('bit_rate', 'N/A')}")
        lines.append(f"  nb_streams:       {fmt.get('nb_streams', 'N/A')}")
        lines.append("")

    if not streams:
        lines.append("No streams found.")
        err = probe.get("error")
        if err:
            lines.append("\nffprobe error object:")
            lines.append(json.dumps(err, indent=2))
        return "\n".join(lines)

    lines.append("Streams:")
    for st in streams:
        idx = st.get("index", "N/A")
        st_type = st.get("codec_type", "N/A")
        codec = st.get("codec_name", "N/A")
        lines.append(f"  - stream #{idx} ({st_type}): {codec}")

        if st_type == "video":
            lines.append(f"      size:           {st.get('width','N/A')}x{st.get('height','N/A')}")
            lines.append(f"      pix_fmt:        {st.get('pix_fmt','N/A')}")
            lines.append(f"      profile:        {st.get('profile','N/A')}")
            lines.append(f"      r_frame_rate:   {fmt_rate(st.get('r_frame_rate'))}")
            lines.append(f"      avg_frame_rate: {fmt_rate(st.get('avg_frame_rate'))}")
            lines.append(f"      bit_rate:       {st.get('bit_rate','N/A')}")
        lines.append("")

    err = probe.get("error")
    if err:
        lines.append("ffprobe error object:")
        lines.append(json.dumps(err, indent=2))

    return "\n".join(lines).rstrip()


def _recv_all(sock: socket.socket, max_bytes: int = 64_000) -> bytes:
    sock.settimeout(2.0)
    chunks: list[bytes] = []
    total = 0
    try:
        while total < max_bytes:
            b = sock.recv(4096)
            if not b:
                break
            chunks.append(b)
            total += len(b)
            # stop early if we likely got full RTSP headers + SDP
            if b"\r\n\r\n" in b"".join(chunks):
                # allow a bit more for SDP body
                sock.settimeout(0.2)
    except socket.timeout:
        # Timeout is expected; return whatever data we have collected so far.
        pass
    return b"".join(chunks)


def rtsp_handshake_probe(url: str, timeout_s: float = 2.0) -> dict[str, Any]:
    """
    Sends RTSP OPTIONS and DESCRIBE to the server and captures raw responses.
    This tells you immediately if the endpoint is truly speaking RTSP.
    """
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or 554
    if not host:
        return {"ok": False, "error": "Could not parse host from URL"}

    path = parts.path or "/"
    # RTSP request URI should be full URL (common), not just path
    req_uri = urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))

    result: dict[str, Any] = {
        "ok": False,
        "host": host,
        "port": port,
        "req_uri": sanitize_url(req_uri),
    }

    def do_req(method: str, cseq: int, extra_headers: list[str] | None = None) -> str:
        headers = [
            f"{method} {req_uri} RTSP/1.0",
            f"CSeq: {cseq}",
            "User-Agent: rtsp_diagnostics/1.0",
        ]
        if extra_headers:
            headers.extend(extra_headers)
        msg = "\r\n".join(headers) + "\r\n\r\n"
        return msg

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)

            # OPTIONS
            sock.sendall(do_req("OPTIONS", 1).encode("utf-8"))
            resp1 = _recv_all(sock).decode("utf-8", errors="replace")

        # new connection for DESCRIBE (some servers are picky)
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(
                do_req("DESCRIBE", 2, ["Accept: application/sdp"]).encode("utf-8")
            )
            resp2 = _recv_all(sock).decode("utf-8", errors="replace")

        result["ok"] = True
        result["options_response"] = resp1.strip()
        result["describe_response"] = resp2.strip()

        # quick classification
        first_line = (resp1.splitlines()[:1] or [""])[0]
        result["options_first_line"] = first_line
        if first_line.startswith("HTTP/"):
            result["note"] = "Server responded with HTTP, not RTSP (URL/port likely wrong)."
        elif not first_line.startswith("RTSP/"):
            result["note"] = "Server did not respond with RTSP status line (may be wrong service/port)."
        else:
            result["note"] = "Server appears to speak RTSP."

        return result

    except Exception as e:
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def frame_count_probe(
    ffprobe: str,
    url: str,
    transport: str,
    timeout_us: int,
    timeout_opt: str | None,
    seconds: int,
) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-count_packets",
        "-count_frames",
        "-read_intervals",
        f"0%+{seconds}",
        "-select_streams",
        "v:0",
        "-rtsp_transport",
        transport,
    ]
    if timeout_opt:
        cmd += [f"-{timeout_opt}", str(timeout_us)]
    cmd += [url]

    rc, out, err = _run(cmd, timeout_s=float(seconds + 10))
    probe = parse_json_maybe(out)
    if rc != 0 and not probe:
        probe = {"_stderr": err.strip(), "_returncode": rc}
    return probe


def main() -> int:
    ap = argparse.ArgumentParser(description="RTSP stream diagnostics (handshake + ffprobe).")
    ap.add_argument("url", help="RTSP URL (rtsp://...)")
    ap.add_argument("--transport", choices=["auto", "tcp", "udp"], default="auto")
    ap.add_argument("--timeout-us", type=int, default=5_000_000)
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--out", default="", help="Write full ffprobe JSON to this file.")
    ap.add_argument("--no-frame-count", action="store_true")
    ap.add_argument("--no-handshake", action="store_true")
    ap.add_argument(
        "--loglevel",
        default="warning",
        choices=["quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug"],
        help="ffprobe loglevel (default: warning). Use debug to see RTSP negotiation.",
    )
    args = ap.parse_args()

    ffprobe = detect_ffprobe()
    safe_url = sanitize_url(args.url)

    print("== RTSP Diagnostics ==")
    print(f"URL:              {safe_url}")
    print(f"ffprobe:          {ffprobe}")
    print(f"ffprobe version:  {ffprobe_version(ffprobe)}")

    timeout_opt = detect_supported_timeout_option(ffprobe)
    print(f"timeout option:   {timeout_opt or 'none detected'}")
    print(f"timeout_us:       {args.timeout_us}")

    if not args.no_handshake:
        print("\n== Raw RTSP handshake probe (OPTIONS + DESCRIBE) ==")
        hs = rtsp_handshake_probe(args.url)
        print(f"handshake ok:     {hs.get('ok')}")
        if not hs.get("ok"):
            print(f"error:            {hs.get('error')}")
        else:
            print(f"note:             {hs.get('note')}")
            print("\n-- OPTIONS response --")
            print(hs.get("options_response", "(none)"))
            print("\n-- DESCRIBE response --")
            print(hs.get("describe_response", "(none)"))

    transports = ["tcp", "udp"] if args.transport == "auto" else [args.transport]

    best_probe: dict[str, Any] = {}
    best_transport = transports[0]
    for tr in transports:
        print(f"\n-- ffprobe ({tr}) --")
        cmd = build_ffprobe_cmd(ffprobe, args.url, tr, args.timeout_us, timeout_opt, args.loglevel)
        t0 = time.time()
        rc, out, err = _run(cmd, timeout_s=25.0)
        elapsed = time.time() - t0
        probe = parse_json_maybe(out)

        print(f"elapsed:          {elapsed:.3f}s")
        print(f"returncode:       {rc}")
        if err.strip():
            print("stderr:")
            print(err.strip())

        streams = probe.get("streams", []) if probe else []
        has_video = any(s.get("codec_type") == "video" for s in (streams or []))
        best_probe = probe or {}
        best_transport = tr
        if has_video:
            break

    print("\n== Summary ==")
    if best_probe:
        print(summarize(best_probe))
    else:
        print("No JSON output from ffprobe (connection likely failed).")

    if args.out and best_probe:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(best_probe, f, indent=2)
        print(f"\nWrote full ffprobe JSON to: {args.out}")

    if not args.no_frame_count:
        print(f"\n== Short frame-count probe (v:0, {args.seconds}s) ==")
        fc = frame_count_probe(ffprobe, args.url, best_transport, args.timeout_us, timeout_opt, args.seconds)
        if "_stderr" in fc:
            print("Frame-count probe failed:")
            print(f"returncode: {fc.get('_returncode')}")
            print(fc.get("_stderr", ""))
        else:
            streams = fc.get("streams", []) or []
            if streams:
                st = streams[0]
                print(f"codec:            {st.get('codec_name', 'N/A')}")
                print(f"size:             {st.get('width', 'N/A')}x{st.get('height', 'N/A')}")
                print(f"avg_frame_rate:   {fmt_rate(st.get('avg_frame_rate'))}")
                print(f"r_frame_rate:     {fmt_rate(st.get('r_frame_rate'))}")
                print(f"nb_read_frames:   {st.get('nb_read_frames', 'N/A')}")
                print(f"nb_read_packets:  {st.get('nb_read_packets', 'N/A')}")
            print("\n(frame-count JSON)")
            print(json.dumps(fc, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
