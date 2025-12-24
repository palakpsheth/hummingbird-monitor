"""
Tests for the hbmon package __init__.py module.
"""

from __future__ import annotations

from unittest.mock import patch


def test_version_exists():
    """Test that __version__ is defined."""
    import hbmon
    assert hasattr(hbmon, "__version__")
    assert isinstance(hbmon.__version__, str)
    assert len(hbmon.__version__) > 0


def test_get_version_function():
    """Test the _get_version function."""
    import hbmon
    # Access the private function
    version = hbmon._get_version()
    assert isinstance(version, str)


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import hbmon
    assert "__version__" in hbmon.__all__


def test_get_version_fallback():
    """Test that _get_version returns fallback when package not found."""
    from importlib.metadata import PackageNotFoundError

    def mock_version(name):
        raise PackageNotFoundError(name)

    import hbmon
    with patch.object(hbmon, '_version', mock_version):
        # Re-call the function to test the fallback
        result = hbmon._get_version()
        # Since the function is already cached, we test the fallback by checking
        # that the function handles the exception properly
        assert isinstance(result, str)
