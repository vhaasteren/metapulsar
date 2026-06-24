"""Timing fit-parameter partition and default marginalization policy."""

from __future__ import annotations

from dataclasses import dataclass

from metapulsar.pint_helpers import (
    get_parameters_by_type_from_models,
    resolve_parameter_alias,
)


@dataclass(frozen=True)
class PartitionResult:
    """Resolved sampled/marginalized fit-parameter partition."""

    fitpars: tuple[str, ...]
    marginalized: tuple[str, ...]
    sampled: tuple[str, ...]
    idx_marginalized: tuple[int, ...]
    idx_sampled: tuple[int, ...]


def _discover_dispersion_dmx_params(pint_model) -> set[str]:
    discovered: set[str] = set()
    for comp in getattr(pint_model, "components", {}).values():
        if getattr(comp, "category", None) != "dispersion_dmx":
            continue
        for name in getattr(comp, "params", []):
            discovered.add(resolve_parameter_alias(name))
    return discovered


def default_marginalized_fitpars(host) -> tuple[str, ...]:
    """Default policy: astrometry + spindown + dispersion(+dmx) categories."""
    fitpars = [resolve_parameter_alias(p) for p in host.fitpars]
    model = host.pint_model()
    if model is None:
        raise ValueError("host.pint_model() is required for marginalize='default'")

    discovered: set[str] = set()
    for category in ("astrometry", "spindown", "dispersion"):
        names = get_parameters_by_type_from_models(category, {"ref": model})
        discovered.update(resolve_parameter_alias(name) for name in names)
    discovered.update(_discover_dispersion_dmx_params(model))

    return tuple(name for name in fitpars if name in discovered)


def resolve_partition(
    host,
    marginalize: str | list[str] | tuple[str, ...] | None = "default",
) -> PartitionResult:
    """Resolve sampled/marginalized names and canonical index mappings."""
    fitpars = tuple(resolve_parameter_alias(p) for p in host.fitpars)
    if len(set(fitpars)) != len(fitpars):
        raise ValueError("Duplicate fit parameters after alias normalization")

    if marginalize == "default":
        marginalized = default_marginalized_fitpars(host)
    elif marginalize is None:
        marginalized = tuple()
    elif isinstance(marginalize, str):
        raise ValueError(
            "marginalize must be 'default', None, or a sequence of fitpars"
        )
    else:
        normalized = tuple(resolve_parameter_alias(p) for p in marginalize)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate entries in marginalize list")
        unknown = [p for p in normalized if p not in fitpars]
        if unknown:
            raise ValueError(f"Unknown fit parameters in marginalize list: {unknown}")
        marginalized = normalized

    marginalized_set = set(marginalized)
    sampled = tuple(p for p in fitpars if p not in marginalized_set)

    idx_marginalized = tuple(i for i, p in enumerate(fitpars) if p in marginalized_set)
    idx_sampled = tuple(i for i, p in enumerate(fitpars) if p not in marginalized_set)

    return PartitionResult(
        fitpars=fitpars,
        marginalized=marginalized,
        sampled=sampled,
        idx_marginalized=idx_marginalized,
        idx_sampled=idx_sampled,
    )
