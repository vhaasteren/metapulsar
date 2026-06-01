"""Discovery-compatible nonlinear timing assembly (mode='nmat')."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .partitioning import TimingPartition, compute_timing_partition
from .signal_builder import _call_engine_timing_delta
from .transforms import TransformRegistry


@dataclass(frozen=True)
class DiscoveryNonlinearTimingComponents:
    """Container for Discovery nonlinear timing components."""

    delay: Any
    timing_gp: Any | None
    partition: TimingPartition
    transform_registry: TransformRegistry
    sampled_parameter_names: dict[str, str]


def _require_discovery():
    try:
        from discovery import likelihood as discovery_likelihood
        from discovery import signals as discovery_signals
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Discovery is required for the Discovery nonlinear timing builder. "
            "Install Discovery to use this API."
        ) from exc

    return discovery_signals, discovery_likelihood


def _z_parameter_name(psr, name: str, param: str) -> str:
    psr_name = getattr(psr, "name", "psr")
    return f"{psr_name}_{name}_{param}"


def _build_delay_callable(
    *,
    engine,
    partition: TimingPartition,
    transform_registry: TransformRegistry,
    sampled_parameter_names: Mapping[str, str],
    strict_missing_sampled_params: bool,
):
    """Build a Discovery delay term compatible with JAX-based samplers (NumPyro NUTS)."""
    import jax
    import jax.numpy as jnp

    ref_residuals = getattr(engine, "_reference_residuals", None)
    if ref_residuals is None and hasattr(engine, "_mmat"):
        ref_residuals = np.zeros(int(engine._mmat.shape[0]), dtype=float)
    ref_residuals = np.asarray(ref_residuals, dtype=float)
    if ref_residuals.ndim != 1:
        raise ValueError(
            "Engine must expose 1D '_reference_residuals' (or '_mmat') for Discovery delay."
        )

    param_order = list(partition.sampled_params)
    param_name_order = [sampled_parameter_names[param] for param in param_order]
    result_shape = (int(ref_residuals.shape[0]),)
    result_dtype = jnp.float64

    def _delay_host(z_flat: np.ndarray) -> np.ndarray:
        z_flat = np.asarray(z_flat, dtype=float).reshape(-1)
        z_params = {param: float(z_flat[idx]) for idx, param in enumerate(param_order)}
        delta_params = transform_registry.to_physical(z_params)
        delay_sec = _call_engine_timing_delta(
            engine,
            delta_params=delta_params,
            strict_missing=strict_missing_sampled_params,
        )
        return np.asarray(delay_sec, dtype=float)

    def _delay_vjp_fwd(z_flat):
        y = jax.pure_callback(
            _delay_host,
            jnp.zeros(result_shape, dtype=result_dtype),
            z_flat,
        )
        return y, z_flat

    def _delay_grad_host(
        z_flat_host: np.ndarray, cotangent_host: np.ndarray
    ) -> np.ndarray:
        z_np = np.asarray(z_flat_host, dtype=float).reshape(-1)
        g_np = np.asarray(cotangent_host, dtype=float).reshape(-1)
        grad = np.zeros(z_np.shape[0], dtype=float)
        f0 = _delay_host(z_np)
        step = 1.0e-6
        for idx in range(z_np.shape[0]):
            z_plus = z_np.copy()
            z_plus[idx] += step
            f_plus = _delay_host(z_plus)
            grad[idx] = float(np.dot(g_np, (f_plus - f0) / step))
        return grad

    def _delay_vjp_bwd(z_flat, cotangent):
        grad = jax.pure_callback(
            _delay_grad_host,
            jnp.zeros((len(param_order),), dtype=result_dtype),
            z_flat,
            cotangent,
        )
        return (grad,)

    @jax.custom_vjp
    def _delay_jax(z_flat):
        return _delay_vjp_fwd(z_flat)[0]

    _delay_jax.defvjp(_delay_vjp_fwd, _delay_vjp_bwd)

    def _delay_eager(params):
        z_flat = np.array(
            [float(params[name]) for name in param_name_order],
            dtype=float,
        )
        return _delay_host(z_flat)

    def delay(params):
        if not any(isinstance(v, jax.core.Tracer) for v in params.values()):
            return _delay_eager(params)

        z_flat = jnp.stack(
            [
                jnp.asarray(params[name], dtype=result_dtype)
                for name in param_name_order
            ],
            axis=0,
        )
        return _delay_jax(z_flat)

    delay.params = param_name_order
    return delay


def _build_timing_gp(
    *,
    psr,
    partition: TimingPartition,
    constant: float,
    svd: bool,
    scale: float,
    name: str,
    discovery_signals,
):
    if not partition.marginalized_params:
        return None

    mmat = np.asarray(psr.Mmat, dtype=np.float64)
    if mmat.ndim != 2:
        raise ValueError("Discovery pulsar must provide a 2D Mmat array.")

    subset = mmat[:, partition.idx_marginalized]
    if subset.shape[1] == 0:
        return None

    if svd:
        fmat, _, _ = np.linalg.svd(scale * subset, full_matrices=False)
    else:
        norms = np.sqrt(np.sum(subset**2, axis=0))
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        fmat = np.asarray(subset / safe_norms, dtype=np.float64)

    return discovery_signals.makegp_improper(
        psr,
        fmat,
        constant=float(constant),
        name=f"{name}_timingmodel",
        variable=False,
    )


def build_discovery_nonlinear_timing_components(
    *,
    psr,
    engine,
    sampled_params: Sequence[str],
    marginalized_params: Sequence[str] | None = None,
    mode: str = "nmat",
    standardization: Mapping[str, object] | None = None,
    idx_from_fitpars: Mapping[str, int | Sequence[int]] | None = None,
    name: str = "nonlinear_timing_model",
    constant: float = 1.0e40,
    svd: bool = False,
    scale: float = 1.0,
    strict_missing_sampled_params: bool = True,
) -> DiscoveryNonlinearTimingComponents:
    """Build Discovery delay + marginalized timing-GP components."""

    if mode != "nmat":
        raise ValueError(
            "Discovery nonlinear timing add-on currently supports mode='nmat' only."
        )

    fitpars = list(getattr(engine, "fitpars", []))
    if not fitpars:
        raise ValueError("Engine must expose canonical 'fitpars' for partitioning.")

    discovery_signals, _ = _require_discovery()

    partition = compute_timing_partition(
        fitpars=fitpars,
        sampled_params=sampled_params,
        marginalized_params=marginalized_params,
        idx_from_fitpars=idx_from_fitpars,
    )
    transform_registry = TransformRegistry(partition.sampled_params, standardization)
    transform_registry.validate_roundtrip()

    sampled_parameter_names = {
        param: _z_parameter_name(psr, name, param) for param in partition.sampled_params
    }
    delay = _build_delay_callable(
        engine=engine,
        partition=partition,
        transform_registry=transform_registry,
        sampled_parameter_names=sampled_parameter_names,
        strict_missing_sampled_params=strict_missing_sampled_params,
    )
    timing_gp = _build_timing_gp(
        psr=psr,
        partition=partition,
        constant=constant,
        svd=svd,
        scale=scale,
        name=name,
        discovery_signals=discovery_signals,
    )

    return DiscoveryNonlinearTimingComponents(
        delay=delay,
        timing_gp=timing_gp,
        partition=partition,
        transform_registry=transform_registry,
        sampled_parameter_names=sampled_parameter_names,
    )


def build_discovery_nonlinear_timing_likelihood(
    *,
    psr,
    noise,
    engine,
    sampled_params: Sequence[str],
    marginalized_params: Sequence[str] | None = None,
    mode: str = "nmat",
    standardization: Mapping[str, object] | None = None,
    idx_from_fitpars: Mapping[str, int | Sequence[int]] | None = None,
    name: str = "nonlinear_timing_model",
    residuals=None,
    constant: float = 1.0e40,
    svd: bool = False,
    scale: float = 1.0,
    strict_missing_sampled_params: bool = True,
    red_noise_signal: Any | None = None,
    extra_signals: Sequence[Any] | None = None,
    return_components: bool = False,
):
    """Build a Discovery ``PulsarLikelihood`` with nonlinear + linear timing components."""

    components = build_discovery_nonlinear_timing_components(
        psr=psr,
        engine=engine,
        sampled_params=sampled_params,
        marginalized_params=marginalized_params,
        mode=mode,
        standardization=standardization,
        idx_from_fitpars=idx_from_fitpars,
        name=name,
        constant=constant,
        svd=svd,
        scale=scale,
        strict_missing_sampled_params=strict_missing_sampled_params,
    )
    _, discovery_likelihood = _require_discovery()

    signals = [psr.residuals if residuals is None else residuals, noise]
    if red_noise_signal is not None:
        signals.append(red_noise_signal)
    if components.timing_gp is not None:
        signals.append(components.timing_gp)
    signals.append(components.delay)
    if extra_signals:
        signals.extend(extra_signals)

    likelihood = discovery_likelihood.PulsarLikelihood(signals)
    if return_components:
        return likelihood, components
    return likelihood
