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


# Hard physical domains in native units. Bounds here are interpreted in this
# module's native convention (e.g. KIN in degrees).
_NON_NEGATIVE = {"PB", "FB0", "A1", "M2", "MTOT", "MP", "H3", "PX", "PBDOT"}
_UNIT_INTERVAL = {"ECC", "E", "SINI", "STIGMA"}
_KIN_INTERVAL_DEG = (0.0, 180.0)


def native_physical_bounds(name: str) -> tuple[float | None, float | None]:
    """Return ``(lower, upper)`` native physical bounds, ``None`` where unbounded."""
    key = _normalize_name(name)
    if key == "KIN":
        return _KIN_INTERVAL_DEG
    if key in _UNIT_INTERVAL:
        return 0.0, 1.0
    if key in _NON_NEGATIVE:
        return 0.0, None
    return None, None
