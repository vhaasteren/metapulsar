"""Per-PTA libstempo timing engine."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from nltiming.protocols import GaugeProvenance

from nltiming.engine_support import (
    LinearModel,
    LinearTimingEngine,
    is_exact_linear_param,
)
from .delta import Tempo2DeltaEngine
from nltiming.hybrid import (
    is_hybrid_engine_axis,
    resolve_hybrid_partition,
    validate_nonlinear_params,
)


class LibstempoEngine:
    """Native libstempo residual-deltan engine.

    libstempo centres residuals internally with no switch. Provenance
    reports ``export="applied-unknown"``; ``gauge_applied`` is derived from that.

    ``nonlinear_params`` (``None`` | ``"binary"`` | ``"binary+"``) is the same
    hybrid residual mode JUG executes: under a hybrid mode only the binary
    axes (plus ``PX`` for ``"binary+"``) reach tempo2; every other fitpar is
    evaluated through its design-matrix column (see :mod:`.hybrid`).
    """

    engine_name = "tempo2"

    def __init__(
        self,
        *,
        engine: Tempo2DeltaEngine,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        engine_fitpars: tuple[str, ...] | None = None,
        exact_linear_fitpars: frozenset[str] | set[str] | None = None,
        nonlinear_params: str | None = None,
    ):
        self._engine = engine
        self._model = linear_model
        self._param_mapping = dict(param_mapping or {})
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)
        self.nonlinear_params = validate_nonlinear_params(nonlinear_params)
        # A stamped mode always implies its partition, even for a direct
        # construction that passed no explicit native/exact-linear lists.
        engine_fitpars, exact_linear_fitpars = resolve_hybrid_partition(
            fitpars=self.fitpars,
            param_mapping=self._param_mapping,
            mode=self.nonlinear_params,
            engine_fitpars=engine_fitpars,
            exact_linear_fitpars=exact_linear_fitpars,
        )
        self._engine_fitpars = tuple(engine_fitpars)
        self._native_indices = tuple(
            self.fitpars.index(name) for name in self._engine_fitpars
        )
        self._exact_linear_fitpars = frozenset(exact_linear_fitpars)
        self._exact_linear_indices = tuple(
            self.fitpars.index(name) for name in self._exact_linear_fitpars
        )

    @classmethod
    def from_contribution(
        cls,
        lt_psr,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        nonlinear_params: str | None = None,
    ) -> "LibstempoEngine":
        engine = Tempo2DeltaEngine(lt_psr)
        mapping = dict(param_mapping or {})
        mode = validate_nonlinear_params(nonlinear_params)
        engine_fitpars: list[str] = []
        exact_linear: list[str] = []
        settable = set(getattr(engine, "_reference_values", {}))

        for name in tuple(linear_model.fitpars):
            engine_param = mapping.get(name, name)
            if is_exact_linear_param(engine_param):
                exact_linear.append(name)
                continue
            if engine_param not in settable:
                exact_linear.append(name)
                continue
            if not is_hybrid_engine_axis(engine_param, mode):
                exact_linear.append(name)
                continue
            engine_fitpars.append(name)

        return cls(
            engine=engine,
            linear_model=linear_model,
            param_mapping=mapping,
            engine_fitpars=tuple(engine_fitpars),
            exact_linear_fitpars=frozenset(exact_linear),
            nonlinear_params=mode,
        )

    def exact_linear_fitpars(self) -> frozenset[str]:
        """Pulsar fitpars evaluated exactly via the design matrix."""
        return self._exact_linear_fitpars

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Fitpars whose engine delay is affine in delta."""
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
        delta_dict = _delta_dict(self._engine_fitpars, delta_native)
        if self._param_mapping:
            delta_dict = {
                self._param_mapping.get(name, name): value
                for name, value in delta_dict.items()
            }
        return self._engine.delta_residuals(delta_dict) + self._exact_linear_delta(
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
        return -(columns @ delta[np.asarray(self._exact_linear_indices, dtype=int)])

    def gauge_provenance(self) -> GaugeProvenance:
        return GaugeProvenance(
            export="applied-unknown",
            reference_mode="unknown",
            reporting_mode="mean",
            reporting_weighted=False,
        )

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"


def _delta_dict(fitpars: tuple[str, ...], delta_theta: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta_theta, dtype=float).reshape(-1)
    if delta.shape != (len(fitpars),):
        raise ValueError("delta_theta shape mismatch with fitpars")
    return {name: float(delta[i]) for i, name in enumerate(fitpars)}


class LinearizedLibstempoEngine(LinearTimingEngine):
    """Explicit linearized libstempo test double using a frozen design matrix."""

    engine_name = "tempo2"

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance | None = None,
    ) -> "LinearizedLibstempoEngine":
        if gauge_provenance is None:
            # Residual product is gauge-free (LinearModel); host reference gauge
            # is not known from a bare design matrix — do not claim "none".
            gauge_provenance = GaugeProvenance(
                export="none",
                reference_mode="unknown",
                reporting_mode="unknown",
            )
        return cls(model, gauge_provenance=gauge_provenance)
