"""Per-PTA Vela.jl (pyvela) timing engine.

Vela evaluates residuals through ``SPNTA.time_residuals`` on its internal
(raw) parameter vector; ``scale_factors`` map PINT-unit values to raw units
componentwise. Working in *deltas* around the par-file reference sidesteps
Vela's float64 storage conventions entirely: the F0 big/small split and the
epoch-from-PEPOCH offsets are additive constants that cancel, so a native
PINT-unit delta scales directly into a raw-vector delta.

Not JAX-capable: use with the Enterprise/PTMCMC likelihood interface or for cross-engine
validation; NUTS needs a JAX engine (JUG).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nltiming.protocols import GaugeProvenance

from nltiming.engine_support import LinearModel, is_exact_linear_param
from .delta import _is_zero_delta


def _ensure_par_uncertainties(par_file) -> Path:
    """Return a par file where every fitted parameter carries an uncertainty.

    pyvela refuses to construct an ``SPNTA`` when a fitted parameter lacks a
    frequentist uncertainty (it cannot build its default cheat prior). The
    engine only evaluates residuals — priors never enter — so fitted lines
    missing an uncertainty get a placeholder one in a patched temp copy.
    """
    par_file = Path(par_file)
    lines = par_file.read_text().splitlines()
    patched: list[str] = []
    changed = False
    for line in lines:
        tokens = line.split()
        if len(tokens) == 3 and tokens[2] == "1":
            line = f"{line} 1.0"
            changed = True
        patched.append(line)
    if not changed:
        return par_file
    out = Path(tempfile.mkdtemp(prefix="metapulsar_vela_")) / par_file.name
    out.write_text("\n".join(patched) + "\n")
    return out


class VelaDeltaEngine:
    """pyvela-backed residual-deviation engine.

    ``Vela.form_residuals`` does not remove a phase mean. The default
    ``phase_mean_mode=None`` therefore exports gauge-free residual deltas.
    Optional ``"weighted"`` / ``"unweighted"`` modes remain available for
    diagnostics that deliberately re-apply a mean.
    """

    def __init__(
        self,
        spnta: Any,
        *,
        isort: np.ndarray | None = None,
        phase_mean_mode: str | None = None,
        weights: np.ndarray | None = None,
    ):
        self._spnta = spnta
        self.param_names = [str(name) for name in spnta.param_names]
        self.fitpars = list(self.param_names)
        self._index = {name: i for i, name in enumerate(self.param_names)}
        self._scale = np.asarray(spnta.scale_factors, dtype=float)
        self._raw_ref = np.asarray(spnta.default_params, dtype=float)
        self._isort = None if isort is None else np.asarray(isort, dtype=int)
        reference = np.asarray(spnta.time_residuals(self._raw_ref), dtype=float)
        if self._isort is not None:
            reference = reference[self._isort]
        self._reference_residuals = reference

        if phase_mean_mode not in (None, "weighted", "unweighted"):
            raise ValueError(
                "phase_mean_mode must be None, 'weighted', or 'unweighted'; "
                f"got {phase_mean_mode!r}"
            )
        if phase_mean_mode is None:
            self._weights = None
        elif phase_mean_mode == "unweighted":
            self._weights = np.ones_like(self._reference_residuals)
        elif weights is not None:
            self._weights = np.asarray(weights, dtype=float).reshape(-1)
        else:
            errors = np.asarray(
                spnta.scaled_toa_unceritainties(self._raw_ref), dtype=float
            )
            self._weights = 1.0 / errors**2

    def delta_residuals(self, delta_params: dict[str, float]) -> np.ndarray:
        if _is_zero_delta(delta_params):
            return np.zeros_like(self._reference_residuals)

        raw = self._raw_ref.copy()
        for name, delta in delta_params.items():
            if name not in self._index:
                raise KeyError(f"Vela model has no free parameter '{name}'")
            idx = self._index[name]
            raw[idx] = raw[idx] + float(delta) * self._scale[idx]
        residuals = np.asarray(self._spnta.time_residuals(raw), dtype=float)
        if self._isort is not None:
            residuals = residuals[self._isort]
        delta_residuals = residuals - self._reference_residuals
        if self._weights is None:
            return delta_residuals
        mean = (self._weights @ delta_residuals) / self._weights.sum()
        return delta_residuals - mean


class VelaEngine:
    """Native Vela.jl residual-deltan engine.

    Nonlinear residual deltas come from ``VelaDeltaEngine``; the design matrix
    and reference theta metadata are served from the pulsar-derived
    ``LinearModel`` so the composite pulsar engine uses the same canonical
    columns as the pulsar design matrix. Fit parameters Vela cannot evaluate
    natively are routed to the exact-linear path.
    """

    engine_name = "vela"

    def __init__(
        self,
        *,
        engine: VelaDeltaEngine,
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
    def from_contribution(
        cls,
        spnta: Any,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        isort: np.ndarray | None = None,
        phase_mean_mode: str | None = None,
        weights: np.ndarray | None = None,
    ) -> "VelaEngine":
        """Build a native Velan engine from an already-created ``SPNTA``."""
        engine = VelaDeltaEngine(
            spnta, isort=isort, phase_mean_mode=phase_mean_mode, weights=weights
        )
        mapping = dict(param_mapping or {})
        settable = set(engine.param_names)

        native_fitpars: list[str] = []
        exact_linear: list[str] = []
        for name in tuple(linear_model.fitpars):
            engine_param = mapping.get(name, name)
            if is_exact_linear_param(engine_param):
                exact_linear.append(name)
                continue
            if engine_param not in settable:
                exact_linear.append(name)
                continue
            native_fitpars.append(name)

        if not native_fitpars:
            raise ValueError(
                "No Vela-evaluable fit parameters remain after filtering; "
                f"exact-linear candidates: {exact_linear}"
            )

        return cls(
            engine=engine,
            linear_model=linear_model,
            param_mapping=mapping,
            native_fitpars=tuple(native_fitpars),
            exact_linear_fitpars=frozenset(exact_linear),
        )

    @classmethod
    def from_files(
        cls,
        par_file,
        tim_file,
        *,
        linear_model: LinearModel,
        param_mapping: Mapping[str, str] | None = None,
        isort: np.ndarray | None = None,
        phase_mean_mode: str | None = None,
        weights: np.ndarray | None = None,
        spnta_kwargs: Mapping[str, Any] | None = None,
    ) -> "VelaEngine":
        """Build a native Velan engine directly from par/tim files."""
        from pyvela import SPNTA

        kwargs: dict[str, Any] = {"center_epochs": False, "check": False}
        kwargs.update(dict(spnta_kwargs or {}))
        spnta = SPNTA(str(_ensure_par_uncertainties(par_file)), str(tim_file), **kwargs)
        return cls.from_contribution(
            spnta,
            linear_model=linear_model,
            param_mapping=param_mapping,
            isort=isort,
            phase_mean_mode=phase_mean_mode,
            weights=weights,
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
        delta_params = {
            self._param_mapping.get(name, name): float(value)
            for name, value in zip(self._native_fitpars, delta_native, strict=True)
        }
        return self._engine.delta_residuals(delta_params) + self._exact_linear_delta(
            delta
        )

    def design_matrix(self, params=None) -> np.ndarray:
        return np.asarray(self._model.design, dtype=float)

    def _exact_linear_delta(self, delta: np.ndarray) -> np.ndarray:
        if not self._exact_linear_indices:
            return np.zeros(self.design_matrix().shape[0], dtype=float)
        columns = np.asarray(
            self._model.design[:, list(self._exact_linear_indices)], dtype=float
        )
        return -(columns @ delta[np.asarray(self._exact_linear_indices, dtype=int)])

    def gauge_provenance(self) -> GaugeProvenance:
        return GaugeProvenance(
            export="none",
            reference_mode="none",
            reporting_mode="mean",
            reporting_weighted=True,
        )

    @property
    def gauge_applied(self) -> bool:
        return self.gauge_provenance().export != "none"
