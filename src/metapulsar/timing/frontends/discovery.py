"""Discovery likelihood-frontend adapter for nonlinear timing.

This module builds Discovery-native likelihood signals from a bound
``NonLinearTimingModel`` host: an optional improper GP for analytically
marginalized linear fit parameters, plus a JAX nonlinear delay for the
numerically sampled block.

Priors
------
Discovery delay keys carry timing-backend-native ``delta_theta`` values; this module
does **not** attach timing priors to the likelihood. Sampled-parameter priors
live in ``ParameterSpace`` (from ``NonLinearTimingModel.space``) and are
added separately—for example via ``contribute_timing``, which evaluates
``space.logprior_coord`` as a NumPyro factor.

With ``prior_policy="fallback"``, unresolved sampled priors use the reference-stack
*cheat* prior convention: a flat ``uniform`` on ``[center ± cheat_prior_scale · σ]`` in
delta space (center = par-file reference, ``σ`` = par-file uncertainty with
WLS fallback), clipped to ``native_physical_bounds`` when applicable. The
whitening/standardized linear layer is a sampler reparameterization only; it
does not change the physical prior measure.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from metapulsar.timing.partition import PartitionResult
from metapulsar.timing.protocols import JaxTimingBackend


def _sample_key(host_name: str, name: str, fitpar: str) -> str:
    return f"{host_name}_{name}_{fitpar}"


def _build_delay_callable(
    *,
    host,
    backend: JaxTimingBackend,
    partition: PartitionResult,
    name: str,
) -> Callable[[dict[str, object]], object]:
    sampled_names = tuple(partition.sampled)
    sampled_indices = tuple(partition.idx_sampled)
    keys = [_sample_key(host.name, name, fitpar) for fitpar in sampled_names]
    ndim = len(partition.fitpars)

    def delay(params: dict[str, object]):
        try:
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - environment-specific import path
            raise RuntimeError(
                "Discovery timing delay requires JAX (jax.numpy) on the NumPyro NUTS tier"
            ) from exc

        delta_sampled = jnp.asarray([params[key] for key in keys], dtype=float)
        full_delta = jnp.zeros((ndim,), dtype=delta_sampled.dtype)
        for i, col in enumerate(sampled_indices):
            full_delta = full_delta.at[col].set(delta_sampled[i])

        # Discovery uses detres = residuals - delay, so delay = -delta_residual.
        return -backend.residual_delta_jax(full_delta)

    delay.params = list(keys)
    return delay


def discovery_signals(
    *, host, space, backend, partition: PartitionResult, name: str
) -> list:
    """Return Discovery-native timing signals: GP (optional) + nonlinear delay.

    Parameters
    ----------
    host
        Timing host with residuals, design matrix, and TOA metadata.
    space
        ``ParameterSpace`` for the sampled block. Passed for API symmetry with
        ``NonLinearTimingModel.discovery_signals``; delay keys already use
        backend-native ``delta_theta``, so the likelihood path here does not
        consume ``space`` directly. Priors from ``space`` are applied outside
        this builder (see module docstring).
    backend
        JAX-capable timing backend used to evaluate ``residual_delta_jax``.
    partition
        Numerically sampled vs analytically marginalized fit-parameter partition in host
        column order.
    name
        Component name prefix for emitted signal keys.

    Returns
    -------
    list
        Discovery signal factories: ``makegp_improper`` for analytically marginalized
        columns (when present) and a delay callable keyed by
        ``{host.name}_{name}_{fitpar}`` for each sampled parameter.

    Notes
    -----
    Discovery uses ``detres = residuals - delay``, so the emitted delay is
    ``-residual_delta(full_delta)``. Analytically marginalized linear parameters are
    represented with a flat improper GP (``constant=1e40``), matching the
    Woodbury/Schur analytical-marginalization path used elsewhere in the stack.
    """
    from discovery import signals as discovery_signals

    _ = space  # Discovery delay keys are already backend-facing delta_theta values.

    if tuple(partition.fitpars) != tuple(host.fitpars):
        raise ValueError("partition.fitpars must match host.fitpars in canonical order")

    signals: list = []
    if partition.idx_analytically_marginalized:
        basis = np.asarray(
            host.Mmat[:, list(partition.idx_analytically_marginalized)], dtype=float
        )
        signals.append(
            discovery_signals.makegp_improper(
                host,
                basis,
                constant=1.0e40,
                name=f"{name}_timingmodel",
            )
        )

    if not partition.sampled:
        return signals

    if not isinstance(backend, JaxTimingBackend):
        raise ValueError(
            "discovery_signals requires a JAX-capable backend when nonlinear delay is emitted"
        )

    signals.append(
        _build_delay_callable(
            host=host,
            backend=backend,
            partition=partition,
            name=name,
        )
    )
    return signals
