"""Per-PTA PINT timing engine."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from nltiming.protocols import GaugeProvenance

from nltiming.engine_support import LinearModel, LinearTimingEngine
from .delta import PintDeltaEngine


# Kepler ECC/OM/T0-parameterized binary families the chart can chart (ELL1/ELL1H
# expose EPS1/EPS2 directly and are filtered earlier as ``already_laplace``).
_PINT_DD_FAMILY = frozenset({"DD", "DDH", "DDS", "DDK", "DDGR", "BT", "BTX", "T2"})
# Binary families whose secular (post-Keplerian) rates are GR-derived from the
# model name alone (no explicit fitpar); the seam guard must engage for them.
_PINT_GR_DERIVED = frozenset({"DDGR"})
_PINT_SECULAR_PARAMS = ("OMDOT", "PBDOT", "EDOT", "A1DOT", "XDOT")


class PintEngine:
    """Native PINT residual-deltan engine."""

    engine_name = "pint"

    def __init__(
        self,
        *,
        engine: PintDeltaEngine,
        linear_model: LinearModel,
        pint_model: Any = None,
    ):
        self._engine = engine
        self._model = linear_model
        self._pint_model = pint_model
        self.fitpars = tuple(linear_model.fitpars)
        self.native_units = dict(linear_model.native_units)

    @classmethod
    def from_contribution(
        cls, model: Any, toas: Any, *, linear_model: LinearModel
    ) -> "PintEngine":
        return cls(
            engine=PintDeltaEngine(model, toas, isort=None),
            linear_model=linear_model,
            pint_model=model,
        )

    def binary_chart_capability(self, chart_family: str, suffix: str):
        """Authoritative binary-chart capability for the Kepler↔Laplace chart, derived
        directly from the wrapped PINT binary model (no MetaPulsar involvement).

        Returns ``None`` (→ candidacy uses its conservative name-search fallback)
        when the family is not ours, no PINT model is held, or the pulsar carries
        no binary. Otherwise reports the Kepler convention family, and whether the
        epoch-shift identity is exact — ``False`` when any secular rate is active,
        either an explicit nonzero fitpar/param value or a GR-derived family
        (DDGR) whose OMDOT/PBDOT are computed internally and invisible to a name
        search.
        """
        if chart_family != "kepler_laplace":
            return None
        model = self._pint_model
        if model is None:
            return None
        binary_param = getattr(model, "BINARY", None)
        binary_name = getattr(binary_param, "value", None)
        if not isinstance(binary_name, str) or not binary_name:
            return None
        from nltiming.protocols import BinaryChartCapability

        name = binary_name.upper()
        convention = "dd" if name in _PINT_DD_FAMILY else "other"
        secular = self._active_secular_terms(model, name)
        return BinaryChartCapability(
            kepler_convention=convention,
            epoch_shift_exact=not secular,
            secular_terms=tuple(sorted(secular)),
            origin_certified=False,  # flip only via a passing origin-cert PR
            supports_domain=True,
        )

    @staticmethod
    def _active_secular_terms(model: Any, binary_name: str) -> set[str]:
        active: set[str] = set()
        for base in _PINT_SECULAR_PARAMS:
            param = getattr(model, base, None)
            value = getattr(param, "value", None) if param is not None else None
            if value is None:
                continue
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fvalue) and fvalue != 0.0:
                active.add(base)
        if binary_name in _PINT_GR_DERIVED:
            active.update({"OMDOT", "PBDOT"})
        return active

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return self._engine.delta_residuals(_delta_dict(self.fitpars, delta_theta))

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def gauge_provenance(self) -> GaugeProvenance:
        # Constructed with subtract_mean=False (gauge-free export).
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode="mean",
            reporting_weighted=True,
        )

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"


def _delta_dict(fitpars: tuple[str, ...], delta_theta: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta_theta, dtype=float).reshape(-1)
    if delta.shape != (len(fitpars),):
        raise ValueError("delta_theta shape mismatch with fitpars")
    return {name: float(delta[i]) for i, name in enumerate(fitpars)}


class LinearizedPintEngine(LinearTimingEngine):
    """Explicit linearized PINT test double using a frozen design matrix."""

    engine_name = "pint"

    @classmethod
    def from_linear_model(
        cls,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance | None = None,
    ) -> "LinearizedPintEngine":
        if gauge_provenance is None:
            gauge_provenance = GaugeProvenance(
                export="none",
                reference_mode="unknown",
                reporting_mode="mean",
                reporting_weighted=True,
            )
        return cls(model, gauge_provenance=gauge_provenance)
