"""Frozen JUG timing state and JAX residual evaluators for traced timing backends."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.fitting.derivatives_astrometry import compute_astrometric_delay
from jug.fitting.derivatives_dd import _compute_dd_binary_delay_jit
from jug.fitting.derivatives_fd import compute_fd_delay
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_cache,
    _compute_designmatrix_from_setup,
    _compute_full_model_residuals,
    _update_param,
)
from jug.utils.constants import HOURANGLE_PER_RAD, K_DM_SEC, RAD_TO_DEG, SECS_PER_DAY
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


def _phase_change_residual_delta_jax(
    dt_base,
    delay_change,
    f_coeffs,
    weights,
    *,
    mean_mode: str,
    f0,
):
    """Precision-safe JAX residual *delta* from a (small) delay change.

    JAX has no longdouble, but JUG's host ``compute_phase_residuals`` relies on
    longdouble because the absolute spin phase ``F0*dt`` is ~1e11 turns while a
    per-parameter residual change is ~1e-4 turns. Forming the absolute phase in
    float64 quantises away the signal (the observed jax-vs-host divergence).

    Instead we compute the residual *change* relative to the reference directly.
    With ``phase(x) = sum_k F_k x^(k+1)/(k+1)!`` and a delay change ``d`` (so the
    emission time becomes ``x - d``), the exact phase change is

        phase(x - d) - phase(x) = sum_{j>=1} (-d)^j / j! * G_j(x),
        G_j(x) = sum_{m>=0} F_{m+j-1} x^m / m!.

    Every factor here is well-scaled in float64 (``d`` is small; ``G_j`` is a
    low-order polynomial with tiny high-order coefficients), so no precision is
    lost. The reference residual, its pulse numbers, the TZR phase, and any
    non-fitted JUMP phase all cancel analytically in the difference, leaving an
    exact, fully nonlinear residual delta. Pulse numbers are assumed unchanged,
    which holds whenever the delay change stays well within half a spin period.
    """
    x = jnp.asarray(dt_base, dtype=jnp.float64)
    d = jnp.asarray(delay_change, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    n_coeffs = len(f_coeffs)

    phase_change = jnp.zeros_like(x)
    for j in range(1, n_coeffs + 1):
        g_j = jnp.zeros_like(x)
        for m in range(0, n_coeffs - (j - 1)):
            coeff = jnp.asarray(f_coeffs[m + j - 1], dtype=jnp.float64)
            g_j = g_j + coeff * (x**m) / float(math.factorial(m))
        phase_change = phase_change + ((-d) ** j) / float(math.factorial(j)) * g_j

    f0_val = jnp.asarray(f0, dtype=jnp.float64)
    residual_delta = phase_change / f0_val

    if mean_mode == "unweighted":
        residual_delta = residual_delta - jnp.mean(residual_delta)
    else:
        residual_delta = residual_delta - jnp.sum(residual_delta * weights) / jnp.sum(
            weights
        )
    return residual_delta


def _dm_delay_jax(tdb_mjd, freq_mhz, dm_values, dm_epoch: float):
    dt_years = (tdb_mjd - dm_epoch) / 365.25
    dm_eff = jnp.zeros_like(tdb_mjd)
    for i, coeff in enumerate(dm_values):
        dm_eff = dm_eff + coeff * (dt_years**i) / float(math.factorial(i))
    return K_DM_SEC * dm_eff / (freq_mhz**2)


def _reference_param_value(params: Mapping[str, object], param: str) -> float:
    """Return a fit parameter value in native numeric storage units."""
    param_upper = param.upper()
    key = param_upper if param_upper in params else param
    if key not in params:
        for candidate in (param, param_upper):
            if candidate in params:
                key = candidate
                break
        else:
            return 0.0
    value = params[key]
    if param_upper == "RAJ" and isinstance(value, str):
        from jug.io.par_reader import parse_ra

        return float(parse_ra(value))
    if param_upper == "DECJ" and isinstance(value, str):
        from jug.io.par_reader import parse_dec

        return float(parse_dec(value))
    return float(value)


def _normalize_ref_params(params: Mapping[str, object]) -> dict[str, object]:
    """Return session params with string RAJ/DECJ converted to radians."""
    normalized = dict(params)
    for key in ("RAJ", "DECJ"):
        if key in normalized and isinstance(normalized[key], str):
            normalized[key] = _reference_param_value(normalized, key)
    return normalized


def _build_params_from_delta(
    ref_params: dict[str, float],
    fit_params: Sequence[str],
    param_mapping: Mapping[str, str],
    ref_theta: np.ndarray,
    delta_theta,
):
    params = dict(ref_params)
    delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
    ref_theta_j = jnp.asarray(ref_theta, dtype=jnp.float64)
    for idx, name in enumerate(fit_params):
        backend = param_mapping.get(name, name)
        key = backend.upper()
        params[key] = ref_theta_j[idx] + delta_theta[idx]
    return params


def _param_scalar(params: dict, name: str, default: float = 0.0):
    key = name.upper()
    if key in params:
        return params[key]
    return default


def _compute_residual_delta_jax(params: dict, setup, *, phase_mean_mode: str):
    """Residual delta (perturbed - reference) mirroring JUG's host model.

    This accumulates the per-component delay *change* relative to the cached
    reference (``new - initial`` for each fitted group), exactly as JUG's host
    ``_compute_full_model_residuals`` does, then converts that small change into
    a residual delta via the precision-safe phase-change formulation. The huge
    absolute emission time ``dt_base`` is never added to the small change, which
    is what makes the float64 result match JUG's longdouble host path.
    """
    dt_base_np = (
        setup.dt_sec_ld
        if setup.dt_sec_ld is not None
        else np.array(setup.dt_sec_cached, dtype=np.float64)
    )
    dt_base = jnp.asarray(np.asarray(dt_base_np, dtype=np.float64), dtype=jnp.float64)
    tdb_mjd = jnp.asarray(setup.tdb_mjd, dtype=jnp.float64)
    freq_mhz = jnp.asarray(setup.freq_mhz, dtype=jnp.float64)
    weights = jnp.asarray(setup.weights, dtype=jnp.float64)

    # Accumulated delay change (seconds, small) relative to the reference.
    delay_change = jnp.zeros_like(dt_base)

    if setup.dm_params and setup.initial_dm_delay is not None:
        dm_epoch = float(params.get("DMEPOCH", params.get("PEPOCH", 55000.0)))
        dm_values = [
            _param_scalar(params, p if p != "DM0" else "DM", 0.0)
            for p in setup.dm_params
        ]
        new_dm = _dm_delay_jax(tdb_mjd, freq_mhz, dm_values, dm_epoch)
        init_dm = jnp.asarray(setup.initial_dm_delay, dtype=jnp.float64)
        delay_change = delay_change + (new_dm - init_dm)

    if (
        setup.dmx_design_matrix is not None
        and setup.dmx_labels
        and setup.initial_dmx_delay is not None
    ):
        current_dmx = jnp.array(
            [_param_scalar(params, label, 0.0) for label in setup.dmx_labels],
            dtype=jnp.float64,
        )
        matrix = jnp.asarray(setup.dmx_design_matrix, dtype=jnp.float64)
        new_dmx = matrix @ current_dmx
        init_dmx = jnp.asarray(setup.initial_dmx_delay, dtype=jnp.float64)
        delay_change = delay_change + (new_dmx - init_dmx)

    if setup.binary_params and setup.initial_binary_delay is not None:
        toas_prebinary = (
            tdb_mjd
            - jnp.asarray(setup.prebinary_delay_sec, dtype=jnp.float64) / SECS_PER_DAY
        )
        sini_raw = _param_scalar(params, "SINI", 0.0)
        kin = _param_scalar(params, "KIN", 0.0)
        sini = jax.lax.cond(
            jnp.asarray(sini_raw) == 0.0,
            lambda _: jnp.sin(jnp.deg2rad(kin)),
            lambda _: jnp.asarray(sini_raw, dtype=jnp.float64),
            None,
        )
        new_binary = _compute_dd_binary_delay_jit(
            toas_prebinary,
            _param_scalar(params, "A1"),
            _param_scalar(params, "PB"),
            _param_scalar(params, "T0"),
            _param_scalar(params, "ECC"),
            _param_scalar(params, "OM"),
            _param_scalar(params, "OMDOT"),
            _param_scalar(params, "PBDOT"),
            _param_scalar(params, "GAMMA"),
            sini,
            _param_scalar(params, "M2"),
            _param_scalar(params, "XDOT"),
            _param_scalar(params, "EDOT"),
        )
        init_binary = jnp.asarray(setup.initial_binary_delay, dtype=jnp.float64)
        delay_change = delay_change + (new_binary - init_binary)

    if setup.astrometry_params and setup.initial_astrometric_delay is not None:
        new_astro = compute_astrometric_delay(
            params,
            tdb_mjd,
            jnp.asarray(setup.ssb_obs_pos_ls, dtype=jnp.float64),
            obs_sun_pos_ls=(
                None
                if setup.obs_sun_pos_ls is None
                else jnp.asarray(setup.obs_sun_pos_ls, dtype=jnp.float64)
            ),
            obs_planet_pos_ls=setup.obs_planet_pos_ls,
        )
        init_astro = jnp.asarray(setup.initial_astrometric_delay, dtype=jnp.float64)
        delay_change = delay_change + (new_astro - init_astro)

    if setup.fd_params and setup.initial_fd_delay is not None:
        current_fd = {
            p: _param_scalar(params, p) for p in setup.fd_params if p in params
        }
        new_fd = jnp.asarray(compute_fd_delay(freq_mhz, current_fd), dtype=jnp.float64)
        init_fd = jnp.asarray(setup.initial_fd_delay, dtype=jnp.float64)
        delay_change = delay_change + (new_fd - init_fd)

    if setup.sw_params and setup.initial_sw_delay is not None:
        ne_sw = _param_scalar(params, "NE_SW", _param_scalar(params, "NE1AU", 0.0))
        sw_geom = jnp.asarray(setup.sw_geometry_pc, dtype=jnp.float64)
        new_sw = K_DM_SEC * ne_sw * sw_geom / (freq_mhz**2)
        init_sw = jnp.asarray(setup.initial_sw_delay, dtype=jnp.float64)
        delay_change = delay_change + (new_sw - init_sw)

    f_terms = []
    for i in range(10):
        key = f"F{i}"
        if key in params:
            f_terms.append(_param_scalar(params, key))
        elif i == 0:
            f_terms.append(_param_scalar(params, "F0", 1.0))
        else:
            break

    return _phase_change_residual_delta_jax(
        dt_base,
        delay_change,
        f_terms,
        weights,
        mean_mode=phase_mean_mode,
        f0=_param_scalar(params, "F0", f_terms[0]),
    )


def _make_residual_delta_jax_fn(
    *,
    ref_params: dict[str, float],
    fit_params: tuple[str, ...],
    param_mapping: dict[str, str],
    ref_theta: np.ndarray,
    reference_residuals_sec: np.ndarray,
    setup,
    phase_mean_mode: str,
    isort: np.ndarray | None,
):
    ref_params = dict(ref_params)
    param_mapping = dict(param_mapping)

    @jax.jit
    def _fn(delta_theta):
        params = _build_params_from_delta(
            ref_params, fit_params, param_mapping, ref_theta, delta_theta
        )
        # The residual delta is computed directly relative to the reference; the
        # reference residuals (and their pulse numbers / TZR phase) cancel
        # analytically in the phase-change formulation, so no longdouble-precision
        # reference subtraction is required here.
        delta = _compute_residual_delta_jax(
            params, setup, phase_mean_mode=phase_mean_mode
        )
        if isort is not None:
            isort_j = jnp.asarray(isort, dtype=jnp.int32)
            return delta[isort_j]
        return delta

    return _fn


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
        "freq_bary_mhz": cached["freq_bary_mhz"],
        "toas_mjd": toas_mjd,
        "errors_us": errors_us,
        "toa_flags": toa_flags,
        "roemer_shapiro_sec": cached.get("roemer_shapiro_sec"),
        "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
        "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
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

    residual_fn = _make_residual_delta_jax_fn(
        ref_params=ref_params,
        fit_params=fit_params,
        param_mapping=mapping,
        ref_theta=ref_theta,
        reference_residuals_sec=reference_residuals_sec,
        setup=setup,
        phase_mean_mode=phase_mean_mode,
        isort=None if isort is None else np.asarray(isort, dtype=int),
    )

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
    # Match the residual convention: the host ``_compute_full_model_residuals``
    # always removes the *weighted* prefit mean (``compute_phase_residuals`` with
    # its default ``mean_mode="weighted"``), which is equivalent to the timing
    # offset/phase parameter being analytically marginalized in the GLS likelihood. Centering
    # each column the same way makes ``design_matrix @ delta`` tangent to the
    # mean-subtracted nonlinear residual delta. Done once here, in setup
    # (pre-isort) row order so the weights line up with the residual computation.
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
