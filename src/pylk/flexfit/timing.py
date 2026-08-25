"""Frontend-neutral timing-model interface for the relinearization loop.

The flexible-``Phi`` solver treats the timing block like any other basis block,
but a *nonlinear* timing model needs its ``J_z`` rebuilt as the anchor moves.
``fastfit`` drives that outer loop through this small protocol so it never has
to import a specific timing backend; concrete implementations live in
``adapters`` (e.g. an ``nltiming`` ``TimingSignal``/``TimingEvaluator`` model).
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .basis import BasisBlock


@runtime_checkable
class TimingModel(Protocol):
    """A timing block anchored at a coordinate, rebuildable under relinearization."""

    @property
    def sampled_block(self) -> str | None:
        """Name of the learnable ``J_z`` block, or ``None`` for pure marginalization."""

    def residuals(self) -> np.ndarray:
        """Residual vector ``y`` at the current timing anchor."""

    def blocks(self) -> tuple[BasisBlock, ...]:
        """Timing basis blocks (marginalized and/or sampled) at the current anchor."""

    def advance(
        self, sampled_mean: np.ndarray, *, damping: float
    ) -> "TimingModel | None":
        """Return a new anchor from the sampled coefficient mean, or ``None``.

        ``None`` signals that no further relinearization is available (a linear
        or purely-marginalized model) or that the step is negligible.
        """

    def summary(self, sampled_mean: np.ndarray) -> Mapping[str, object]:
        """Report timing estimates (``z``, ``delta``, physical values) at the fit."""
