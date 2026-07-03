"""Frozen JUG timing state and JAX residual evaluators for traced timing backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.fitting.jax_residual_delta import (
    _normalize_ref_params,
    _reference_param_value,
    make_residual_delta_jax_fn,
)
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_cache,
    _compute_designmatrix_from_setup,
    _compute_full_model_residuals,
    _update_param,
)
from jug.utils.constants import HOURANGLE_PER_RAD, RAD_TO_DEG
from jug.utils.units import validate_column_units


@dataclass(frozen=True)
class JaxTimingState:
    """Convention-frozen timing state exported from a JUG session."""

    fit_params: tuple[str, ...]
    param_mapping: tuple[tuple[str, str], ...]
    ref_params: dict[str, float]
    ref_theta: np.ndarray
    reference_residuals_sec: np.ndarray
    subtract_tzr: bool
    compatibility: str
    phase_mean_mode: str
    isort: np.ndarray | None
    design_matrix: np.ndarray
    column_units: tuple[str, ...]
    setup: Any
    _residual_delta_jax_fn: Any

    def residual_delta_np(self, delta_theta: np.ndarray) -> np.ndarray:
        delta_theta = np.asarray(delta_theta, dtype=np.float64).reshape(-1)
        params = dict(self.ref_params)
        for idx, name in enumerate(self.fit_params):
            backend = self._backend_name(name)
            current = float(params.get(backend, self.ref_theta[idx]))
            _update_param(params, backend, current + float(delta_theta[idx]))
        residuals_sec, _, _, _ = _compute_full_model_residuals(params, self.setup)
        residuals_sec = np.asarray(residuals_sec, dtype=np.float64)
        if self.isort is not None:
            return residuals_sec[self.isort] - self.reference_residuals_sec
        return residuals_sec - self.reference_residuals_sec

    def residual_delta_jax(self, delta_theta):
        return self._residual_delta_jax_fn(delta_theta)

    def linearized_residual_delta_jax(self, delta_theta):
        # ``design_matrix`` is the timing design matrix ``d(residual)/d(theta)``
        # at theta=0, already in the engine's native-delta / output (isort) order
        # (identical convention to ``residual_delta_np``), so the linearized
        # residual is a plain matmul -- no sign flip, no unit rescale, no isort.
        delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
        matrix = jnp.asarray(self.design_matrix, dtype=jnp.float64)
        return matrix @ delta_theta

    def linearized_residual_delta_np(self, delta_theta: np.ndarray) -> np.ndarray:
        delta_theta = np.asarray(delta_theta, dtype=np.float64).reshape(-1)
        return np.asarray(self.design_matrix, dtype=np.float64) @ delta_theta

    def _backend_name(self, canonical: str) -> str:
        for canon, backend in self.param_mapping:
            if canon == canonical:
                return backend
        return canonical


def _phase_mean_mode(compatibility: str) -> str:
    mode = str(compatibility).lower()
    if mode in ("tempo2", "tempo2-compatible", "tempo2_compatible"):
        return "unweighted"
    return "weighted"


def _fit_unit_column_to_native_delta(param: str, column: np.ndarray) -> np.ndarray:
    """Convert a JUG design-matrix column from API fit units to native deltas."""
    param_upper = param.upper()
    if param_upper == "RAJ":
        return column * HOURANGLE_PER_RAD
    if param_upper == "DECJ":
        return column * RAD_TO_DEG
    return column


def export_jax_timing_state(
    session,
    *,
    fit_params: Sequence[str],
    subtract_tzr: bool = True,
    compatibility: str | None = None,
    param_mapping: Mapping[str, str] | None = None,
    isort: np.ndarray | None = None,
    phase_mean_mode: str | None = None,
    design_matrix_method: str = "analytic",
) -> JaxTimingState:
    """Export a frozen JAX timing state from a populated JUG ``TimingSession``."""
    fit_params = tuple(str(name) for name in fit_params)
    if not fit_params:
        raise ValueError("fit_params must be non-empty.")

    compatibility = compatibility or getattr(session, "compatibility", "pint")
    phase_mean_mode = phase_mean_mode or _phase_mean_mode(compatibility)
    mapping = dict(param_mapping or {})

    cached = session._cached_result_by_mode.get(subtract_tzr)
    if cached is None:
        session.compute_residuals(subtract_tzr=subtract_tzr, force_recompute=False)
        cached = session._cached_result_by_mode.get(subtract_tzr)
    if cached is None or "dt_sec" not in cached:
        raise RuntimeError(
            "TimingSession cache is unavailable; call compute_residuals() first."
        )

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    errors_us = np.array([toa.error_us for toa in session.toas_data])
    toa_flags = [toa.flags for toa in session.toas_data]
    session_cached_data = {
        "dt_sec": cached["dt_sec"],
        "dt_sec_ld": cached.get("dt_sec_ld"),
        "tdb_mjd": cached["tdb_mjd"],
        "model_mjd": cached.get("model_mjd"),
        "bbat_mjd": cached.get("bbat_mjd"),
        "engine_conventions": cached.get("engine_conventions"),
        "diagnostic_conventions": cached.get("diagnostic_conventions"),
        "freq_bary_mhz": cached["freq_bary_mhz"],
        "toas_mjd": toas_mjd,
        "errors_us": errors_us,
        "toa_flags": toa_flags,
        "roemer_shapiro_sec": cached.get("roemer_shapiro_sec"),
        "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
        "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
        "earth_ssb_ls": cached.get("earth_ssb_ls"),
        "observatory_earth_ls": cached.get("observatory_earth_ls"),
        "sw_geometry_pc": cached.get("sw_geometry_pc"),
        "jump_phase": cached.get("jump_phase"),
        "tzr_phase": cached.get("tzr_phase"),
    }

    mapping = dict(param_mapping or {})
    jug_fit_params = [mapping.get(name, name) for name in fit_params]

    setup = _build_general_fit_setup_from_cache(
        session_cached_data,
        session.params,
        jug_fit_params,
        compatibility=compatibility,
        design_matrix_method=design_matrix_method,
    )

    ref_params = _normalize_ref_params(session.params)
    ref_theta = np.array(
        [
            _reference_param_value(
                ref_params,
                mapping.get(name, name),
            )
            for name in fit_params
        ],
        dtype=np.float64,
    )
    reference_residuals_sec, _, _, _ = _compute_full_model_residuals(ref_params, setup)
    reference_residuals_sec = np.asarray(reference_residuals_sec, dtype=np.float64)

    residual_fn_base = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=tuple(jug_fit_params),
        ref_params=ref_params,
        ref_theta=ref_theta,
        phase_mean_mode=phase_mean_mode,
    )
    if isort is not None:
        isort_j = jnp.asarray(np.asarray(isort, dtype=int), dtype=jnp.int32)

        @jax.jit
        def residual_fn(delta_theta):
            return residual_fn_base(delta_theta)[isort_j]

    else:
        residual_fn = residual_fn_base

    ref_for_delta = reference_residuals_sec
    if isort is not None:
        ref_for_delta = reference_residuals_sec[np.asarray(isort, dtype=int)]

    # Linearized timing design matrix = d(residual_delta)/d(native theta) at
    # theta=0, assembled once from the analytic derivative blocks of the cached
    # ``setup`` (no second par/tim read, no parameter perturbation). JUG's basis
    # is the fitter timing basis in API fit units (M ~= -d residual / d fit-unit
    # param), so exporting residual deltas requires one sign flip plus the
    # RAJ/DECJ fit-unit conversion back to the native delta convention used by
    # ``parameter_space``.
    design_matrix = -np.asarray(
        _compute_designmatrix_from_setup(setup, jug_fit_params), dtype=np.float64
    )
    for col, name in enumerate(fit_params):
        design_matrix[:, col] = _fit_unit_column_to_native_delta(
            name, design_matrix[:, col]
        )
    # Match the residual convention used by JUG's phase residuals.  Tempo2 mode
    # removes the unweighted phase mean; PINT mode removes the weighted mean.
    if phase_mean_mode == "unweighted":
        col_means = np.mean(design_matrix, axis=0)
    else:
        weights = np.asarray(setup.weights, dtype=np.float64)
        col_means = (weights @ design_matrix) / weights.sum()
    design_matrix = design_matrix - col_means
    if isort is not None:
        design_matrix = design_matrix[np.asarray(isort, dtype=int), :]
    column_units = tuple(validate_column_units(list(fit_params)))

    return JaxTimingState(
        fit_params=fit_params,
        param_mapping=tuple(sorted(mapping.items())),
        ref_params=ref_params,
        ref_theta=ref_theta,
        reference_residuals_sec=ref_for_delta,
        subtract_tzr=subtract_tzr,
        compatibility=str(compatibility),
        phase_mean_mode=phase_mean_mode,
        isort=None if isort is None else np.asarray(isort, dtype=int),
        design_matrix=design_matrix,
        column_units=column_units,
        setup=setup,
        _residual_delta_jax_fn=residual_fn,
    )
