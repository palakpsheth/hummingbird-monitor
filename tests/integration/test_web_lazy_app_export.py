from __future__ import annotations

import pytest
import hbmon.web as web


pytestmark = pytest.mark.integration


def test_lazy_app_attr_builds_singleton():
    # reset
    web._app_instance = None
    a1 = web.app
    a2 = web.app
    assert a1 is a2
