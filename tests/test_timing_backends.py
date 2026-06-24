"""Slice-0 scaffold for per-engine timing backend tests."""

import pytest

pytestmark = pytest.mark.skip(reason="Slice 0 scaffold: enable in Slice 3")


def test_timing_backends_scaffold():
    """Placeholder to keep test module visible during staged implementation."""
    assert True
