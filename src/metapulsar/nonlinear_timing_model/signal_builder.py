"""Enterprise-compatible signal assembly for nonlinear timing modeling."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from enterprise.signals import deterministic_signals, gp_signals, parameter
from enterprise.signals.parameter import Function

from .partitioning import TimingPartition, compute_timing_partition
from .transforms import TransformRegistry


def _make_parameter_priors(
    sampled_params: Sequence[str], parameter_priors: Mapping[str, object] | None
) -> dict[str, object]:
    priors: dict[str, object] = {}
    parameter_priors = parameter_priors or {}
    for param in sampled_params:
        if param in parameter_priors:
            priors[param] = parameter_priors[param]
        else:
            priors[param] = parameter.Normal(0.0, 1.0)
    return priors


def _call_engine_timing_delta(
    engine, delta_params: dict[str, float], strict_missing: bool
) -> np.ndarray:
    """Return a timing delay for Enterprise/Discovery residual updates.

    Engines report ``r(theta0 + delta) - r(theta0)``.  Enterprise uses
    ``detres = residuals - delay`` (see ``enterprise_extensions.tm_delay``), so the
    delay must be ``r(theta0) - r(theta0 + delta)``, i.e. the negated engine delta.
    """
    if hasattr(engine, "timing_delta"):
        if strict_missing:
            try:
                raw = np.asarray(
                    engine.timing_delta(
                        delta_params, missing_param_policy="strict_error"
                    ),
                    dtype=float,
                )
            except TypeError:
                raw = np.asarray(engine.timing_delta(delta_params), dtype=float)
            else:
                return -raw
        raw = np.asarray(engine.timing_delta(delta_params), dtype=float)
    elif hasattr(engine, "delta_residuals"):
        raw = np.asarray(engine.delta_residuals(delta_params), dtype=float)
    else:
        raise TypeError(
            "Engine must provide either 'timing_delta(delta_params, ...)' or "
            "'delta_residuals(delta_params)'."
        )

    return -raw


def _build_nonlinear_waveform(
    *,
    engine,
    sampled_params: Sequence[str],
    transform_registry: TransformRegistry,
    strict_missing_sampled_params: bool,
    parameter_priors: Mapping[str, object] | None,
):
    priors = _make_parameter_priors(sampled_params, parameter_priors)
    func_kwargs = dict(priors)

    def nonlinear_delay(toas, mask=None, psr=None, **z_params):
        del toas, psr  # supplied by enterprise but not used by this waveform
        delta_params = transform_registry.to_physical(z_params)
        full_delay = _call_engine_timing_delta(
            engine,
            delta_params=delta_params,
            strict_missing=strict_missing_sampled_params,
        )
        if mask is None:
            return full_delay

        mask_array = np.asarray(mask, dtype=bool)
        if full_delay.shape[0] == mask_array.shape[0]:
            return full_delay[mask_array]
        if full_delay.shape[0] == int(mask_array.sum()):
            return full_delay
        raise ValueError(
            "Nonlinear delay waveform output length does not align with Enterprise mask "
            f"({full_delay.shape[0]} vs {mask_array.shape[0]})."
        )

    return Function(nonlinear_delay, **func_kwargs)


def build_nonlinear_timing_signal(
    *,
    engine,
    sampled_params: Sequence[str],
    marginalized_params: Sequence[str] | None = None,
    mode: str = "nmat",
    standardization: Mapping[str, object] | None = None,
    idx_from_fitpars: Mapping[str, int | Sequence[int]] | None = None,
    name: str = "nonlinear_timing_model",
    coefficients: bool = False,
    strict_missing_sampled_params: bool = True,
    parameter_priors: Mapping[str, object] | None = None,
):
    """Build an enterprise-composable nonlinear timing signal."""

    fitpars = list(getattr(engine, "fitpars", []))
    if not fitpars:
        raise ValueError("Engine must expose canonical 'fitpars' for partitioning.")

    partition: TimingPartition = compute_timing_partition(
        fitpars=fitpars,
        sampled_params=sampled_params,
        marginalized_params=marginalized_params,
        idx_from_fitpars=idx_from_fitpars,
    )
    transform_registry = TransformRegistry(partition.sampled_params, standardization)
    transform_registry.validate_roundtrip()

    nonlinear_waveform = _build_nonlinear_waveform(
        engine=engine,
        sampled_params=partition.sampled_params,
        transform_registry=transform_registry,
        strict_missing_sampled_params=strict_missing_sampled_params,
        parameter_priors=parameter_priors,
    )
    nonlinear_signal = deterministic_signals.Deterministic(
        nonlinear_waveform,
        name=f"{name}_nonlinear",
    )

    if mode not in {"nmat", "basis"}:
        raise ValueError("mode must be either 'nmat' or 'basis'.")

    if not partition.marginalized_params:
        return nonlinear_signal

    if mode == "nmat":
        linear_signal = gp_signals.MarginalizingTimingModel(
            name=f"{name}_linear_nmat",
            idx_exclude=partition.idx_sampled,
        )
    else:
        linear_signal = gp_signals.TimingModel(
            name=f"{name}_linear_basis",
            idx_exclude=partition.idx_sampled,
            coefficients=coefficients,
        )

    return nonlinear_signal + linear_signal
