"""Timing fit-parameter partition and default analytical-marginalization policy."""

from __future__ import annotations

from dataclasses import dataclass

from metapulsar.pint_helpers import (
    get_parameters_by_type_from_models,
    resolve_parameter_alias,
)


@dataclass(frozen=True)
class PartitionResult:
    """Resolved numerically sampled vs analytically marginalized fit-parameter partition."""

    fitpars: tuple[str, ...]
    analytically_marginalized: tuple[str, ...]
    sampled: tuple[str, ...]
    idx_analytically_marginalized: tuple[int, ...]
    idx_sampled: tuple[int, ...]


# Component categories whose fit parameters are identically linear timing nuisances that
# analytically_marginalize="default" integrates out analytically. Anything outside this set
# (most importantly the binary "pulsar_system" parameters) is numerically sampled. JUMP/FD/DMX
# delays are exactly linear in the design matrix, so analytical marginalization is exact
# and avoids the near-degenerate sampled blocks that wreck NUTS conditioning.
_ANALYTICALLY_MARGINALIZED_CATEGORIES = (
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


# Linear nuisance families that analytically_marginalize="default" must always peel off when
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


def default_analytically_marginalized_fitpars(host) -> tuple[str, ...]:
    """Default policy: astrometry + spindown + dispersion(+dmx) categories."""
    model = host.pint_model()
    if model is None:
        raise ValueError(
            "host.pint_model() is required for analytically_marginalize='default'"
        )

    discovered: set[str] = set()
    # for category in ("astrometry", "spindown", "dispersion"):
    for category in ("spindown", "dispersion"):
        names = get_parameters_by_type_from_models(category, {"ref": model})
        for name in names:
            discovered.add(name)
            discovered.add(resolve_parameter_alias(name))
    discovered.update(
        _discover_category_params(model, _ANALYTICALLY_MARGINALIZED_CATEGORIES)
    )

    analytically_marginalized: list[str] = []
    has_linear_family = False
    for raw in host.fitpars:
        candidates = _base_param_candidates(host, raw)
        if any(c.startswith(_LINEAR_FAMILY_PREFIXES) for c in candidates):
            has_linear_family = True
        if candidates & discovered:
            analytically_marginalized.append(resolve_parameter_alias(raw))

    if has_linear_family and not analytically_marginalized:
        raise ValueError(
            "analytically_marginalize='default' resolved to an empty "
            "analytically marginalized set even though the host carries linear nuisance "
            "families (DMX/JUMP/FD). This usually means fitpar name matching against PINT "
            "categories failed (e.g. an unmapped PTA suffix). Refusing to silently sample "
            "every timing parameter."
        )

    return tuple(analytically_marginalized)


def resolve_partition(
    host,
    analytically_marginalize: str | list[str] | tuple[str, ...] | None = "default",
) -> PartitionResult:
    """Resolve numerically sampled vs analytically marginalized names and index mappings."""
    fitpars = tuple(resolve_parameter_alias(p) for p in host.fitpars)
    if len(set(fitpars)) != len(fitpars):
        raise ValueError("Duplicate fit parameters after alias normalization")

    if analytically_marginalize == "default":
        analytically_marginalized = default_analytically_marginalized_fitpars(host)
    elif analytically_marginalize is None:
        analytically_marginalized = tuple()
    elif isinstance(analytically_marginalize, str):
        raise ValueError(
            "analytically_marginalize must be 'default', None, or a sequence of fitpars"
        )
    else:
        normalized = tuple(resolve_parameter_alias(p) for p in analytically_marginalize)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Duplicate entries in analytically_marginalize list")
        unknown = [p for p in normalized if p not in fitpars]
        if unknown:
            raise ValueError(
                f"Unknown fit parameters in analytically_marginalize list: {unknown}"
            )
        analytically_marginalized = normalized

    analytically_marginalized_set = set(analytically_marginalized)
    sampled = tuple(p for p in fitpars if p not in analytically_marginalized_set)

    idx_analytically_marginalized = tuple(
        i for i, p in enumerate(fitpars) if p in analytically_marginalized_set
    )
    idx_sampled = tuple(
        i for i, p in enumerate(fitpars) if p not in analytically_marginalized_set
    )

    return PartitionResult(
        fitpars=fitpars,
        analytically_marginalized=analytically_marginalized,
        sampled=sampled,
        idx_analytically_marginalized=idx_analytically_marginalized,
        idx_sampled=idx_sampled,
    )
