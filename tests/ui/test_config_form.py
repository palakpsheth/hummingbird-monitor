"""
UI checks for the configuration form.
"""

from __future__ import annotations

from playwright.sync_api import expect


def test_config_form_save(live_server_url: str, ui_page) -> None:
    ui_page.goto(f"{live_server_url}/config", wait_until="domcontentloaded")

    expect(ui_page.get_by_role("heading", name="Config")).to_be_visible()
    ui_page.fill("input[name=\"fps_limit\"]", "12")
    ui_page.fill("input[name=\"cooldown_seconds\"]", "2")

    with ui_page.expect_navigation():
        ui_page.get_by_role("button", name="Save").click()

    expect(ui_page.get_by_text("Settings saved.")).to_be_visible()
