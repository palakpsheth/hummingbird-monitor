"""
Integration tests for proxy service wiring.

These tests validate that the docker-compose and nginx configuration
are correctly set up for the web service.
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
def test_docker_compose_does_not_have_stream_service() -> None:
    """Verify that hbmon-stream service has been removed."""
    compose = _load_compose()
    services = compose.get("services", {})

    assert isinstance(services, dict)
    assert "hbmon-stream" not in services, "hbmon-stream service should be removed"


@pytest.mark.integration
def test_docker_compose_proxy_depends_on_web_only() -> None:
    """Verify that hbmon-proxy only depends on hbmon-web."""
    compose = _load_compose()
    services = compose.get("services", {})

    assert isinstance(services, dict)
    proxy_service = services.get("hbmon-proxy")

    assert isinstance(proxy_service, dict)
    depends_on = proxy_service.get("depends_on") or {}
    assert "hbmon-web" in depends_on, "hbmon-proxy should depend on hbmon-web"
    assert "hbmon-stream" not in depends_on, "hbmon-proxy should not depend on hbmon-stream"


@pytest.mark.integration
def test_nginx_routes_root_to_web_service() -> None:
    """Verify that nginx routes root to hbmon-web."""
    nginx = _read_repo_file("nginx.conf")
    locations = _extract_nginx_location_blocks(nginx)
    root_block = locations.get("/")

    assert root_block is not None, "nginx should define / location"
    assert "proxy_pass http://hbmon-web:8000;" in root_block


@pytest.mark.integration
def test_nginx_does_not_route_stream_mjpeg() -> None:
    """Verify that nginx does not have a separate stream.mjpeg route."""
    nginx = _read_repo_file("nginx.conf")
    locations = _extract_nginx_location_blocks(nginx)
    stream_block = locations.get("/api/stream.mjpeg")

    assert stream_block is None, "nginx should not define /api/stream.mjpeg location"


@pytest.mark.integration
def test_docker_compose_uses_yaml_anchors() -> None:
    """Verify that docker-compose.yml uses YAML anchors to reduce duplication."""
    compose_text = _read_repo_file("docker-compose.yml")
    # Check for anchor definitions
    assert "x-common-env:" in compose_text, "Should define x-common-env anchor"
    assert "x-bg-subtraction-env:" in compose_text, "Should define x-bg-subtraction-env anchor"
    assert "x-rtsp-env:" in compose_text, "Should define x-rtsp-env anchor"
    
    # Check for anchor usage (using array syntax for merging multiple anchors)
    # Using regex to allow optional whitespace around anchors and bracket
    anchor_pattern = r"<<:\s*\[\s*\*common-env,\s*\*rtsp-env,\s*\*bg-subtraction-env\s*\]"
    assert re.search(anchor_pattern, compose_text), "Services should use anchor array merge syntax (regex match failed)"
