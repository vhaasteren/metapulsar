"""Slice-0 scaffold for JUG/JAX timing backend tests."""

import pytest

pytestmark = [
    pytest.mark.requires_jug,
    pytest.mark.skip(reason="Slice 0 scaffold: enable in Slice 3"),
]


def test_timing_jug_jax_backend_scaffold():
    """Placeholder to keep test module visible during staged implementation."""
    assert True
