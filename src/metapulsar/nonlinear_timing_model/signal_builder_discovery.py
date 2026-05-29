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
    def delay(params):
        z_params = {
            sampled_param: float(params.get(param_name, 0.0))
            for sampled_param, param_name in sampled_parameter_names.items()
        }
        delta_params = transform_registry.to_physical(z_params)
        return _call_engine_timing_delta(
            engine,
            delta_params=delta_params,
            strict_missing=strict_missing_sampled_params,
        )

    delay.params = [
        sampled_parameter_names[param] for param in partition.sampled_params
    ]
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
    if components.timing_gp is not None:
        signals.append(components.timing_gp)
    signals.append(components.delay)
    if extra_signals:
        signals.extend(extra_signals)

    likelihood = discovery_likelihood.PulsarLikelihood(signals)
    if return_components:
        return likelihood, components
    return likelihood
