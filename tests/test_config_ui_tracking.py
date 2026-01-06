"""
Integration tests for config UI tracking mode toggle behavior.

These tests verify that the config page JavaScript correctly shows/hides
temporal voting and tracking fields based on the tracking mode checkbox.
"""

import pytest
from playwright.sync_api import Page, expect


@pytest.mark.integration
class TestConfigUITrackingToggle:
    """Tests for config UI field visibility based on tracking mode."""

    def test_tracking_enabled_hides_temporal_fields(self, page: Page, live_server_url: str):
        """When tracking checkbox is checked, temporal fields should be hidden."""
        page.goto(f"{live_server_url}/config")
        
        # Check the tracking checkbox
        tracking_checkbox = page.locator("#use_tracking_checkbox")
        tracking_checkbox.check()
        
        # Temporal fields should be hidden
        temporal_fields = page.locator("#temporal-voting-fields")
        expect(temporal_fields).not_to_be_visible()
        
        # Tracking fields should be visible
        tracking_fields = page.locator("#tracking-fields")
        expect(tracking_fields).to_be_visible()

    def test_tracking_disabled_shows_temporal_fields(self, page: Page, live_server_url: str):
        """When tracking checkbox is unchecked, temporal fields should be visible."""
        page.goto(f"{live_server_url}/config")
        
        # Uncheck the tracking checkbox
        tracking_checkbox = page.locator("#use_tracking_checkbox")
        tracking_checkbox.uncheck()
        
        # Temporal fields should be visible
        temporal_fields = page.locator("#temporal-voting-fields")
        expect(temporal_fields).to_be_visible()
        
        # Tracking fields should be hidden
        tracking_fields = page.locator("#tracking-fields")
        expect(tracking_fields).not_to_be_visible()

    def test_initial_state_respects_saved_setting(self, page: Page, live_server_url: str):
        """Initial field visibility should match the saved tracking mode setting."""
        page.goto(f"{live_server_url}/config")
        
        # Get the initial state of the checkbox
        tracking_checkbox = page.locator("#use_tracking_checkbox")
        is_checked = tracking_checkbox.is_checked()
        
        # Verify field visibility matches checkbox state
        temporal_fields = page.locator("#temporal-voting-fields")
        tracking_fields = page.locator("#tracking-fields")
        
        if is_checked:
            expect(temporal_fields).not_to_be_visible()
            expect(tracking_fields).to_be_visible()
        else:
            expect(temporal_fields).to_be_visible()
            expect(tracking_fields).not_to_be_visible()

    def test_toggle_updates_visibility_dynamically(self, page: Page, live_server_url: str):
        """Toggling the checkbox should update field visibility without page reload."""
        page.goto(f"{live_server_url}/config")
        
        tracking_checkbox = page.locator("#use_tracking_checkbox")
        temporal_fields = page.locator("#temporal-voting-fields")
        tracking_fields = page.locator("#tracking-fields")
        
        # Start with tracking enabled
        tracking_checkbox.check()
        expect(temporal_fields).not_to_be_visible()
        expect(tracking_fields).to_be_visible()
        
        # Toggle to disabled
        tracking_checkbox.uncheck()
        expect(temporal_fields).to_be_visible()
        expect(tracking_fields).not_to_be_visible()
        
        # Toggle back to enabled
        tracking_checkbox.check()
        expect(temporal_fields).not_to_be_visible()
        expect(tracking_fields).to_be_visible()

    def test_temporal_fields_exist_in_dom(self, page: Page, live_server_url: str):
        """Temporal fields should exist in DOM even when hidden."""
        page.goto(f"{live_server_url}/config")
        
        # Check temporal fields still exist (just hidden)
        temporal_window = page.locator("input[name='temporal_window_frames']")
        temporal_min = page.locator("input[name='temporal_min_detections']")
        
        expect(temporal_window).to_be_attached()
        expect(temporal_min).to_be_attached()

    def test_tracking_fields_exist_in_dom(self, page: Page, live_server_url: str):
        """Tracking fields should exist in DOM even when hidden."""
        page.goto(f"{live_server_url}/config")
        
        # Check tracking fields exist
        track_high = page.locator("input[name='track_high_thresh']")
        track_low = page.locator("input[name='track_low_thresh']")
        
        expect(track_high).to_be_attached()
        expect(track_low).to_be_attached()

    def test_form_submission_includes_hidden_fields(self, page: Page, live_server_url: str):
        """Form submission should include values from hidden fields."""
        page.goto(f"{live_server_url}/config")
        
        # Set a value in a temporal field while it's still visible
        temporal_window = page.locator("input[name='temporal_window_frames']")
        temporal_window.fill("10")

        # Enable tracking (hides temporal fields)
        tracking_checkbox = page.locator("#use_tracking_checkbox")
        tracking_checkbox.check()
        
        # Verify temporal fields are now hidden
        temporal_fields = page.locator("#temporal-voting-fields")
        expect(temporal_fields).not_to_be_visible()
        
        # The value should still be set even though field is hidden
        assert temporal_window.input_value() == "10"
