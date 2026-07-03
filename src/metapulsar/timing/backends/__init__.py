"""Timing-engine builders and public timing-engine adapter classes."""

from __future__ import annotations

from .composite import PulsarSession, build_composite_backend
from .jug import JugEngine, LinearizedJugEngine
from .pint import LinearizedPintEngine, PintEngine
from .tempo2 import LinearizedLibstempoEngine, LibstempoEngine

_ENGINE_CHOICES = {"tempo2": ("libstempo", "jug"), "pint": ("pint", "jug")}
_IMPL_FAMILY = {"libstempo": "tempo2", "pint": "pint", "jug": "jug"}


def normalize_engines(engines):
    """Return ``{'tempo2': impl, 'pint': impl}`` for an engine selection."""
    if isinstance(engines, str):
        engines = {"tempo2": engines, "pint": engines}
    else:
        engines = dict(engines)
    extra = set(engines) - set(_ENGINE_CHOICES)
    if extra:
        raise ValueError(f"Unknown engine compatibility keys: {sorted(extra)}")
    out = {}
    for native, choices in _ENGINE_CHOICES.items():
        impl = engines.get(native, "jug")
        if impl not in choices:
            raise ValueError(
                f"engines[{native!r}] must be one of {choices}, got {impl!r}"
            )
        out[native] = impl
    return out


def build_backend(
    *,
    fitpars: tuple[str, ...],
    nrows: int,
    sessions: list[PulsarSession],
    host_design=None,
):
    """Build the per-pulsar composite over per-PTA engine sessions."""
    return build_composite_backend(
        fitpars=fitpars,
        nrows=nrows,
        sessions=sessions,
        host_design=host_design,
    )


__all__ = [
    "PulsarSession",
    "JugEngine",
    "LinearizedJugEngine",
    "LinearizedPintEngine",
    "LinearizedLibstempoEngine",
    "PintEngine",
    "LibstempoEngine",
    "build_backend",
    "build_composite_backend",
    "normalize_engines",
    "_ENGINE_CHOICES",
    "_IMPL_FAMILY",
]
