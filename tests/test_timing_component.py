"""Slice-0 scaffold for nonlinear timing component tests."""

import pytest

pytestmark = [
    pytest.mark.requires_discovery,
    pytest.mark.skip(reason="Slice 0 scaffold: enable in Slice 5"),
]


def test_timing_component_scaffold():
    """Placeholder to keep test module visible during staged implementation."""
    assert True
