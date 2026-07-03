"""Per-session libstempo timing engine adapter."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import LinearModel, LinearTimingBackend
from .engines import Tempo2DeltaEngine


class LibstempoEngine:
    """Native libstempo residual-delta adapter."""

    backend_name = "tempo2"

    def __init__(
        self,
        *,
        engine: Tempo2DeltaEngine,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
    ):
        self._engine = engine
        self._model = linear_model
        self._param_mapping = dict(param_mapping or {})
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)

    @classmethod
    def from_session(
        cls,
        lt_psr,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
    ) -> "LibstempoEngine":
        return cls(
            engine=Tempo2DeltaEngine(lt_psr),
            linear_model=linear_model,
            param_mapping=param_mapping,
        )

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = _delta_dict(self.fitpars, delta_theta)
        if self._param_mapping:
            delta = {
                self._param_mapping.get(name, name): value
                for name, value in delta.items()
            }
        return self._engine.delta_residuals(delta)

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)


def _delta_dict(fitpars: tuple[str, ...], delta_theta: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta_theta, dtype=float).reshape(-1)
    if delta.shape != (len(fitpars),):
        raise ValueError("delta_theta shape mismatch with fitpars")
    return {name: float(delta[i]) for i, name in enumerate(fitpars)}


class LinearizedLibstempoEngine(LinearTimingBackend):
    """Explicit linearized libstempo test double using a frozen design matrix."""

    backend_name = "tempo2"

    @classmethod
    def from_linear_model(cls, model: LinearModel) -> "LinearizedLibstempoEngine":
        return cls(model)
