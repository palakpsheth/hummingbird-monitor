"""
UI flow checks for ROI calibration.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect


@pytest.mark.ui
def test_roi_save_flow(live_server_url: str, ui_page) -> None:
    ui_page.goto(f"{live_server_url}/calibrate", wait_until="domcontentloaded")

    expect(ui_page.get_by_role("heading", name="Calibrate ROI")).to_be_visible()
    ui_page.fill("#x1", "0.1")
    ui_page.fill("#y1", "0.2")
    ui_page.fill("#x2", "0.8")
    ui_page.fill("#y2", "0.9")

    with ui_page.expect_navigation():
        ui_page.get_by_role("button", name="Save ROI").click()

    expect(ui_page.get_by_text("Current ROI:")).to_contain_text("0.1000,0.2000,0.8000,0.9000")
