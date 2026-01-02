"""
UI smoke tests for the dashboard page.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_dashboard_loads(live_server_url: str, ui_page) -> None:
    ui_page.goto(f"{live_server_url}/", wait_until="networkidle")

    expect(ui_page.get_by_role("heading", name="Dashboard")).to_be_visible()
    expect(ui_page.locator("#live-feed-img")).to_be_visible()
    expect(ui_page.get_by_role("link", name="Calibrate ROI")).to_have_attribute("href", "/calibrate")
