"""Discovery frontend adapter for nonlinear timing."""

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
                "Discovery timing delay requires JAX (jax.numpy) for nonlinear path"
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
    """Return Discovery-native timing signals: GP (optional) + nonlinear delay."""
    from discovery import signals as discovery_signals

    _ = space  # Discovery delay keys are already backend-facing delta_theta values.

    if tuple(partition.fitpars) != tuple(host.fitpars):
        raise ValueError("partition.fitpars must match host.fitpars in canonical order")

    signals: list = []
    if partition.idx_marginalized:
        basis = np.asarray(host.Mmat[:, list(partition.idx_marginalized)], dtype=float)
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
