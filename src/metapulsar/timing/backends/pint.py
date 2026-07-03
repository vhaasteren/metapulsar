"""Per-session PINT timing engine adapter."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import LinearModel, LinearTimingBackend
from .engines import PintDeltaEngine


class PintEngine:
    """Native PINT residual-delta adapter."""

    backend_name = "pint"

    def __init__(self, *, engine: PintDeltaEngine, linear_model: LinearModel):
        self._engine = engine
        self._model = linear_model
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)

    @classmethod
    def from_session(
        cls, model: Any, toas: Any, *, linear_model: LinearModel
    ) -> "PintEngine":
        return cls(
            engine=PintDeltaEngine(model, toas, isort=None),
            linear_model=linear_model,
        )

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return self._engine.delta_residuals(_delta_dict(self.fitpars, delta_theta))

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)


def _delta_dict(fitpars: tuple[str, ...], delta_theta: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta_theta, dtype=float).reshape(-1)
    if delta.shape != (len(fitpars),):
        raise ValueError("delta_theta shape mismatch with fitpars")
    return {name: float(delta[i]) for i, name in enumerate(fitpars)}


class LinearizedPintEngine(LinearTimingBackend):
    """Explicit linearized PINT test double using a frozen design matrix."""

    backend_name = "pint"

    @classmethod
    def from_linear_model(cls, model: LinearModel) -> "LinearizedPintEngine":
        return cls(model)
