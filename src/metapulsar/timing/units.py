"""Unit conversion helpers for timing-space parameters."""

from __future__ import annotations

import numpy as np

from metapulsar.pint_helpers import resolve_parameter_alias


_RAD_PER_DEG = np.pi / 180.0
_RAD_PER_HOUR = np.pi / 12.0
_HOURANGLE_PARAMS = {"RAJ", "LAMBDA", "ELONG"}
_DEGREE_PARAMS = {"DECJ", "BETA", "ELAT"}
_IDENTITY_DISPLAY_PARAMS = {"T0", "TASC"}
_NON_NEGATIVE = {"PB", "FB0", "A1", "M2", "MTOT", "MP", "H3", "PX", "PBDOT"}
_UNIT_INTERVAL = {"ECC", "E", "SINI", "STIGMA"}
_KIN_INTERVAL_DEG = (0.0, 180.0)
_KNOWN_UNIT_KEYS = (
    _HOURANGLE_PARAMS
    | _DEGREE_PARAMS
    | _IDENTITY_DISPLAY_PARAMS
    | _NON_NEGATIVE
    | _UNIT_INTERVAL
    | {"KIN"}
)


def _normalize_name(name: str) -> str:
    """Normalize parameter names for unit lookups."""
    text = str(name)
    parts = text.split("_")
    candidates = [text]
    if len(parts) > 1:
        candidates.append("_".join(parts[:-1]))  # Strip PTA suffix, e.g. M2_ng5.
        candidates.append(parts[-1])  # Strip signal prefix, e.g. psr_ntm_M2.
        candidates.extend(reversed(parts))
        for start in range(len(parts)):
            candidates.append("_".join(parts[start:]))
        for start in range(len(parts)):
            for stop in range(start + 1, len(parts) + 1):
                candidates.append("_".join(parts[start:stop]))

    seen: set[str] = set()
    for candidate in candidates:
        key = resolve_parameter_alias(candidate).upper()
        if key in seen:
            continue
        seen.add(key)
        if key in _KNOWN_UNIT_KEYS:
            return key
    return resolve_parameter_alias(text).upper()


def to_native(name: str, display_value):
    """Convert display-unit values to native timing units."""
    key = _normalize_name(name)
    value = np.asarray(display_value)
    if key in _HOURANGLE_PARAMS:
        return value * _RAD_PER_HOUR
    if key in _DEGREE_PARAMS:
        return value * _RAD_PER_DEG
    if key in _IDENTITY_DISPLAY_PARAMS:
        # Keep native/storage convention in MJD days.
        return value
    return value


def to_display(name: str, native_value):
    """Convert native timing-unit values to display units."""
    key = _normalize_name(name)
    value = np.asarray(native_value)
    if key in _HOURANGLE_PARAMS:
        return value / _RAD_PER_HOUR
    if key in _DEGREE_PARAMS:
        return value / _RAD_PER_DEG
    if key in _IDENTITY_DISPLAY_PARAMS:
        return value
    return value


def display_unit(name: str) -> str:
    """Human-readable display unit label."""
    key = _normalize_name(name)
    if key in _HOURANGLE_PARAMS:
        return "hourangle"
    if key in _DEGREE_PARAMS:
        return "deg"
    if key in _IDENTITY_DISPLAY_PARAMS:
        return "MJD"
    return "native"


# Hard physical domains in native units. Bounds here are interpreted in this
# module's native convention (e.g. KIN in degrees).
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
