"""Unit conversion helpers for timing-space parameters."""

from __future__ import annotations

import numpy as np


_RAD_PER_DEG = np.pi / 180.0
_RAD_PER_HOUR = np.pi / 12.0


def _normalize_name(name: str) -> str:
    """Normalize parameter names for unit lookups."""
    # Handle pulsar-qualified names like "J1713+0747_ntm_RAJ".
    return name.split("_")[-1].upper()


def to_native(name: str, display_value):
    """Convert display-unit values to native timing units."""
    key = _normalize_name(name)
    value = np.asarray(display_value)
    if key in {"RAJ", "LAMBDA", "ELONG"}:
        return value * _RAD_PER_HOUR
    if key in {"DECJ", "BETA", "ELAT"}:
        return value * _RAD_PER_DEG
    if key in {"T0", "TASC"}:
        # Keep native/storage convention in MJD days.
        return value
    return value


def to_display(name: str, native_value):
    """Convert native timing-unit values to display units."""
    key = _normalize_name(name)
    value = np.asarray(native_value)
    if key in {"RAJ", "LAMBDA", "ELONG"}:
        return value / _RAD_PER_HOUR
    if key in {"DECJ", "BETA", "ELAT"}:
        return value / _RAD_PER_DEG
    if key in {"T0", "TASC"}:
        return value
    return value


def display_unit(name: str) -> str:
    """Human-readable display unit label."""
    key = _normalize_name(name)
    if key in {"RAJ", "LAMBDA", "ELONG"}:
        return "hourangle"
    if key in {"DECJ", "BETA", "ELAT"}:
        return "deg"
    if key in {"T0", "TASC"}:
        return "MJD"
    return "native"
