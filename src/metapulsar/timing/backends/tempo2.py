"""Per-session libstempo timing engine adapter."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import LinearModel, LinearTimingBackend, is_exact_linear_param
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
        native_fitpars: tuple[str, ...] | None = None,
        exact_linear_fitpars: frozenset[str] | set[str] | None = None,
    ):
        self._engine = engine
        self._model = linear_model
        self._param_mapping = dict(param_mapping or {})
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self._native_fitpars = (
            self.fitpars if native_fitpars is None else tuple(native_fitpars)
        )
        self._native_indices = tuple(
            self.fitpars.index(name) for name in self._native_fitpars
        )
        self._exact_linear_fitpars = frozenset(exact_linear_fitpars or frozenset())
        self._exact_linear_indices = tuple(
            self.fitpars.index(name) for name in self._exact_linear_fitpars
        )

    @classmethod
    def from_session(
        cls,
        lt_psr,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
    ) -> "LibstempoEngine":
        engine = Tempo2DeltaEngine(lt_psr)
        mapping = dict(param_mapping or {})
        native_fitpars: list[str] = []
        exact_linear: list[str] = []
        settable = set(getattr(engine, "_reference_values", {}))

        for name in tuple(linear_model.fitpars):
            backend_name = mapping.get(name, name)
            if is_exact_linear_param(backend_name):
                exact_linear.append(name)
                continue
            if backend_name not in settable:
                exact_linear.append(name)
                continue
            native_fitpars.append(name)

        return cls(
            engine=engine,
            linear_model=linear_model,
            param_mapping=mapping,
            native_fitpars=tuple(native_fitpars),
            exact_linear_fitpars=frozenset(exact_linear),
        )

    def exact_linear_fitpars(self) -> frozenset[str]:
        """Pulsar fitpars evaluated exactly via the design matrix."""
        return self._exact_linear_fitpars

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")

        delta_native = delta[np.asarray(self._native_indices, dtype=int)]
        delta = _delta_dict(self._native_fitpars, delta_native)
        if self._param_mapping:
            delta = {
                self._param_mapping.get(name, name): value
                for name, value in delta.items()
            }
        return self._engine.delta_residuals(delta) + self._exact_linear_delta(
            delta_theta
        )

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def _exact_linear_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        if not self._exact_linear_indices:
            return np.zeros(self.design_matrix().shape[0], dtype=float)
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        columns = np.asarray(
            self._model.design[:, list(self._exact_linear_indices)], dtype=float
        )
        return columns @ delta[np.asarray(self._exact_linear_indices, dtype=int)]


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
