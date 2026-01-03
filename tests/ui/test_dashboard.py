"""
UI smoke tests for the dashboard page.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_dashboard_loads(live_server_url: str, ui_page) -> None:
    # Navigate and wait for load
    response = ui_page.goto(f"{live_server_url}/", wait_until="domcontentloaded")
    
    # Verify we got a successful response
    assert response is not None and response.ok, f"Page load failed with status {response.status if response else 'None'}"
    
    # Wait for body to ensure page is rendered
    ui_page.wait_for_selector("body", state="attached")
    
    # Check for dashboard heading - use simpler locator
    expect(ui_page.locator("h1:has-text('Dashboard')")).to_be_visible()
    expect(ui_page.locator("#live-feed-img")).to_be_visible()
    # Use .first() to select the first matching link (avoids strict mode violation with multiple matches)
    expect(ui_page.get_by_role("link", name="Calibrate ROI").first).to_have_attribute("href", "/calibrate")
