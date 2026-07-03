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
# (most importantly astrometric kinematics and binary "pulsar_system" parameters) is
# numerically sampled. Astrometry is handled by explicit position-only registries below
# because PINT's astrometry category also includes proper motion and parallax.
_ANALYTICALLY_MARGINALIZED_CATEGORIES = (
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
# pulsar whose fitpar names carry a PTA suffix that breaks canonical name matching).
_LINEAR_FAMILY_PREFIXES = ("DMX", "JUMP", "FD")
_LINEAR_EXACT = frozenset({"DM", "DM1", "DM2", "OFFSET", "PHOFF"})
_ASTROMETRY_POSITION_ONLY = frozenset(
    {"RAJ", "DECJ", "ELONG", "ELAT", "RA", "DEC", "LAMBDA", "BETA"}
)
_ASTROMETRY_SAMPLED_BY_DEFAULT = frozenset({"PMRA", "PMDEC", "PMELONG", "PMELAT", "PX"})


def _base_param_candidates(pulsar, name: str) -> set[str]:
    """Return the canonical base-name candidates for a (possibly suffixed) fitpar.

    Composite pulsars expose PTA-suffixed fitpars (e.g. ``RAJ_ng5``) while PINT
    category discovery yields unsuffixed canonical names (e.g. ``RAJ``). The pulsar
    carries the suffixed -> per-PTA base mapping in ``_fitparameters``; use it so
    membership tests resolve to the underlying PINT parameter names.
    """
    candidates: set[str] = {name, resolve_parameter_alias(name)}
    mapping = getattr(pulsar, "_fitparameters", None) or {}
    for base in mapping.get(name, {}).values():
        candidates.add(base)
        candidates.add(resolve_parameter_alias(base))
    return candidates


def _exact_linear_policy_candidates(pulsar, name: str) -> set[str]:
    """Return candidates eligible for exact-linear/prefix registries.

    Composite pulsars with PTA-suffixed names must provide ``_fitparameters`` mappings.
    Without that mapping, broad prefix matching would silently treat ``DMX_0001_ng5``
    as a valid DMX parameter instead of surfacing broken suffix resolution.
    """
    mapping = getattr(pulsar, "_fitparameters", None)
    if mapping is not None:
        bases = mapping.get(name, {})
        candidates: set[str] = set()
        for base in bases.values():
            candidates.add(base)
            candidates.add(resolve_parameter_alias(base))
        return candidates
    return {name, resolve_parameter_alias(name)}


def default_analytically_marginalized_fitpars(pulsar) -> tuple[str, ...]:
    """Default policy: linear timing nuisances plus astrometry positions."""
    model = pulsar.pint_model()
    if model is None:
        raise ValueError(
            "pulsar.pint_model() is required for analytically_marginalize='default'"
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
    for raw in pulsar.fitpars:
        candidates = _base_param_candidates(pulsar, raw)
        exact_linear_candidates = _exact_linear_policy_candidates(pulsar, raw)
        canonical_candidates = {resolve_parameter_alias(c).upper() for c in candidates}
        canonical_fallbacks = {
            resolve_parameter_alias(c).upper() for c in exact_linear_candidates
        }
        if canonical_candidates & _LINEAR_EXACT or any(
            c.startswith(_LINEAR_FAMILY_PREFIXES) for c in canonical_candidates
        ):
            has_linear_family = True
        if canonical_candidates & _ASTROMETRY_SAMPLED_BY_DEFAULT:
            continue
        if (
            candidates & discovered
            or canonical_candidates & _ASTROMETRY_POSITION_ONLY
            or canonical_fallbacks & _LINEAR_EXACT
            or any(c.startswith(_LINEAR_FAMILY_PREFIXES) for c in canonical_fallbacks)
        ):
            analytically_marginalized.append(resolve_parameter_alias(raw))

    if has_linear_family and not analytically_marginalized:
        raise ValueError(
            "analytically_marginalize='default' resolved to an empty "
            "analytically marginalized set even though the pulsar carries linear nuisance "
            "families (DMX/JUMP/FD). This usually means fitpar name matching against PINT "
            "categories failed (e.g. an unmapped PTA suffix). Refusing to silently sample "
            "every timing parameter."
        )

    return tuple(analytically_marginalized)


def resolve_partition(
    pulsar,
    analytically_marginalize: str | list[str] | tuple[str, ...] | None = "default",
) -> PartitionResult:
    """Resolve numerically sampled vs analytically marginalized names and index mappings."""
    fitpars = tuple(resolve_parameter_alias(p) for p in pulsar.fitpars)
    if len(set(fitpars)) != len(fitpars):
        raise ValueError("Duplicate fit parameters after alias normalization")

    if analytically_marginalize == "default":
        analytically_marginalized = default_analytically_marginalized_fitpars(pulsar)
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
