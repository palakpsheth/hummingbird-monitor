"""
Integration tests for proxy and stream service wiring.

These tests validate that the docker-compose and nginx configuration
keep the MJPEG stream isolated on its dedicated service.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_file(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _load_compose() -> dict[str, object]:
    compose = _read_repo_file("docker-compose.yml")
    data = yaml.safe_load(compose)
    if not isinstance(data, dict):
        raise AssertionError("docker-compose.yml should parse into a mapping")
    return data


def _extract_nginx_location_blocks(nginx: str) -> dict[str, str]:
    locations: dict[str, str] = {}
    current_location: str | None = None
    brace_depth = 0
    buffer: list[str] = []

    for line in nginx.splitlines():
        stripped = line.strip()
        if current_location is None:
            match = re.match(r"location\s+(\S+)\s*\{", stripped)
            if match:
                current_location = match.group(1)
                brace_depth = stripped.count("{") - stripped.count("}")
                buffer = []
            continue

        brace_depth += stripped.count("{") - stripped.count("}")
        buffer.append(line)

        if brace_depth <= 0 and current_location is not None:
            locations[current_location] = "\n".join(buffer)
            current_location = None
            buffer = []
            brace_depth = 0

    return locations


@pytest.mark.integration
def test_docker_compose_defines_stream_service() -> None:
    compose = _load_compose()
    services = compose.get("services", {})

    assert isinstance(services, dict)
    assert "hbmon-stream" in services

    stream_service = services["hbmon-stream"]
    assert isinstance(stream_service, dict)
    assert stream_service.get("container_name") == "hbmon-stream"

    command = stream_service.get("command") or []
    assert isinstance(command, list)
    assert "--bind" in command
    assert "0.0.0.0:8001" in command


@pytest.mark.integration
def test_docker_compose_proxy_depends_on_stream() -> None:
    compose = _load_compose()
    services = compose.get("services", {})

    assert isinstance(services, dict)
    proxy_service = services.get("hbmon-proxy")

    assert isinstance(proxy_service, dict)
    depends_on = proxy_service.get("depends_on") or {}
    assert "hbmon-stream" in depends_on, "hbmon-proxy should depend on hbmon-stream"


@pytest.mark.integration
def test_nginx_routes_stream_to_stream_service() -> None:
    nginx = _read_repo_file("nginx.conf")
    locations = _extract_nginx_location_blocks(nginx)
    stream_block = locations.get("/api/stream.mjpeg")

    assert stream_block is not None, "nginx should define /api/stream.mjpeg location"
    assert "proxy_pass http://hbmon-stream:8001;" in stream_block


@pytest.mark.integration
def test_nginx_routes_root_to_web_service() -> None:
    nginx = _read_repo_file("nginx.conf")
    locations = _extract_nginx_location_blocks(nginx)
    root_block = locations.get("/")

    assert root_block is not None, "nginx should define / location"
    assert "proxy_pass http://hbmon-web:8000;" in root_block
