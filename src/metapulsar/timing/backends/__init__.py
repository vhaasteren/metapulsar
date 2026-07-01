"""Timing-backend builders and public timing-engine adapter classes."""

from __future__ import annotations

from .composite import BackendSession, build_composite_backend
from .jug import JugTimingBackend, LinearizedJugTimingBackend
from .pint import LinearizedPintTimingBackend, PintTimingBackend
from .tempo2 import LinearizedTempo2TimingBackend, Tempo2TimingBackend


def build_backend(
    *,
    name: str,
    fitpars: tuple[str, ...],
    nrows: int,
    sessions: list[BackendSession],
    missing_param_policy: str = "linear_fallback",
    host_design=None,
):
    """Build a composite backend for the requested family name."""
    if name not in {"jug", "pint", "tempo2"}:
        raise ValueError(f"Unsupported backend name: {name}")
    mismatched = [
        session.name
        for session in sessions
        if getattr(session.backend, "backend_name", None) != name
    ]
    if mismatched:
        raise ValueError(f"Sessions cannot honor backend '{name}': {mismatched}")
    return build_composite_backend(
        fitpars=fitpars,
        nrows=nrows,
        sessions=sessions,
        missing_param_policy=missing_param_policy,
        host_design=host_design,
    )


__all__ = [
    "BackendSession",
    "JugTimingBackend",
    "LinearizedJugTimingBackend",
    "LinearizedPintTimingBackend",
    "LinearizedTempo2TimingBackend",
    "PintTimingBackend",
    "Tempo2TimingBackend",
    "build_backend",
]
