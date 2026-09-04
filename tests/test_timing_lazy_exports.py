"""Façade identity checks for MetaPulsar's lazy nltiming re-exports."""

from __future__ import annotations

import importlib

import pytest

import metapulsar
from metapulsar import _TIMING_LAZY_EXPORTS


@pytest.mark.parametrize("name", sorted(_TIMING_LAZY_EXPORTS))
def test_timing_lazy_export_matches_nltiming_target(name):
    module_name, attr = _TIMING_LAZY_EXPORTS[name]
    target = getattr(importlib.import_module(module_name), attr)
    assert getattr(metapulsar, name) is target
