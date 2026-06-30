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


# Component categories whose fit parameters are linear timing nuisances that
# marginalize="default" peels off. Anything outside this set (most importantly
# the binary "pulsar_system" parameters) is sampled nonlinearly. JUMP/FD/DMX
# delays are exactly linear in the design matrix, so marginalizing them is exact
# and avoids the near-degenerate sampled blocks that wreck NUTS conditioning.
_MARGINALIZED_CATEGORIES = (
    # "astrometry",
    # "spindown",
    "dispersion_constant",
    "dispersion_dmx",
    "phase_jump",
    "fd",
    "frequency_dependent",
)


def _discover_category_params(pint_model, categories) -> set[str]:
    wanted = set(categories)
    discovered: set[str] = set()
    for comp in getattr(pint_model, "components", {}).values():
        if getattr(comp, "category", None) not in wanted:
            continue
        for name in getattr(comp, "params", []):
            discovered.add(name)
            discovered.add(resolve_parameter_alias(name))
    return discovered


# Linear nuisance families that marginalize="default" must always peel off when
# present. Used only as a sanity guard against silent partition failures (e.g. a
# host whose fitpar names carry a PTA suffix that breaks canonical name matching).
_LINEAR_FAMILY_PREFIXES = ("DMX", "JUMP", "FD")


def _base_param_candidates(host, name: str) -> set[str]:
    """Return the canonical base-name candidates for a (possibly suffixed) fitpar.

    Composite hosts expose PTA-suffixed fitpars (e.g. ``RAJ_ng5``) while PINT
    category discovery yields unsuffixed canonical names (e.g. ``RAJ``). The host
    carries the suffixed -> per-PTA base mapping in ``_fitparameters``; use it so
    membership tests resolve to the underlying PINT parameter names.
    """
    candidates: set[str] = {name, resolve_parameter_alias(name)}
    mapping = getattr(host, "_fitparameters", None) or {}
    for base in mapping.get(name, {}).values():
        candidates.add(base)
        candidates.add(resolve_parameter_alias(base))
    return candidates


def default_marginalized_fitpars(host) -> tuple[str, ...]:
    """Default policy: astrometry + spindown + dispersion(+dmx) categories."""
    model = host.pint_model()
    if model is None:
        raise ValueError("host.pint_model() is required for marginalize='default'")

    discovered: set[str] = set()
    # for category in ("astrometry", "spindown", "dispersion"):
    for category in ("spindown", "dispersion"):
        names = get_parameters_by_type_from_models(category, {"ref": model})
        for name in names:
            discovered.add(name)
            discovered.add(resolve_parameter_alias(name))
    discovered.update(_discover_category_params(model, _MARGINALIZED_CATEGORIES))

    marginalized: list[str] = []
    has_linear_family = False
    for raw in host.fitpars:
        candidates = _base_param_candidates(host, raw)
        if any(c.startswith(_LINEAR_FAMILY_PREFIXES) for c in candidates):
            has_linear_family = True
        if candidates & discovered:
            marginalized.append(resolve_parameter_alias(raw))

    if has_linear_family and not marginalized:
        raise ValueError(
            "marginalize='default' resolved to an empty marginalized set even though "
            "the host carries linear nuisance families (DMX/JUMP/FD). This usually "
            "means fitpar name matching against PINT categories failed (e.g. an "
            "unmapped PTA suffix). Refusing to silently sample every timing parameter."
        )

    return tuple(marginalized)


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
