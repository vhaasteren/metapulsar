"""Pylk — interactive pulsar-timing workbench (incubated in MetaPulsar).

Pylk is the scientific workspace proposed in ``feature_pylk.md``: interactive
exploration, waveform inspection, and point optimization over MetaPulsar /
``nltiming`` / Discovery / Enterprise stacks.

This package is intentionally thin today. The first shipped engine under the
Pylk umbrella is :mod:`pylk.flexfit` (quick-look empirical-Bayes timing + GP
fits). Workspace/UI layers land later; see ``src/pylk/feature_pylk.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metapulsar")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
