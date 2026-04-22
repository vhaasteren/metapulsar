"""Tiny shims so Dave's vendored modules import cleanly even when their
CLI/parallel-pool helpers are missing.

This module is a tutorial-only fallback. In Dave's upstream environment
both `schwimmbad` and `cyclopts` are present and these shims are unused.
"""

from __future__ import annotations


class MultiPool:
    """Serial drop-in replacement for `schwimmbad.MultiPool`."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def map(self, fn, iterable):
        return list(map(fn, iterable))

    def close(self):
        return None


class MPIPool:
    """Stub MPI pool that raises if instantiated."""

    def __init__(self, *_, **__):
        raise RuntimeError(
            "MPIPool unavailable in tutorial environment. "
            "Set use_mpi=False (notebook default) or install schwimmbad+mpi4py."
        )


class _DummyApp:
    """Minimal stand-in for `cyclopts.App` so `@app.command` decorators no-op."""

    def command(self, fn=None, **_):
        if fn is None:
            return lambda f: f
        return fn

    def __call__(self, *_, **__):
        raise RuntimeError(
            "cyclopts CLI is disabled in the tutorial shim. "
            "Call the underlying functions directly from Python instead."
        )


class _CycloptsModule:
    """Quacks like the `cyclopts` module for the limited surface Dave uses."""

    App = _DummyApp
