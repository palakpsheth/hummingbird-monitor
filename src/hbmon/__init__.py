# src/hbmon/__init__.py
"""
hbmon: Hummingbird Monitor
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

__all__ = ["__version__"]


def _get_version() -> str:
    try:
        return _version("hbmon")
    except PackageNotFoundError:
        # Happens when running from source without installing the package.
        # Keep a safe placeholder rather than duplicating version here.
        return "0.0.0+local"


__version__ = _get_version()
