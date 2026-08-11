"""PINT parameter-name and unit utilities (MetaPulsar-owned).

Pure functions over PINT's alias tables and component registry: alias
resolution, category discovery, and the SI/MJD par-value boundary
Par-value unit boundary helpers. MetaPulsar's core combination path — shared-par
merging, canonical tim/par writing, binary-family conversion — depends on
these, so they live here rather than in the nonlinear-timing package: basic
MetaPulsar operations must not import ``nltiming``.

This is a self-contained vendored twin of ``nltiming.pint_compat``. Both are
pure functions of the installed PINT registry, so they behave identically for a
given PINT version; keep them in sync when either changes. The dependency rule
is one-directional — consumer packages may import ``nltiming``, never the
reverse — so MetaPulsar owns its own copy rather than importing nltiming's.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import astropy.units as u
import numpy as np
from pint.models.parameter import MJDParameter, floatParameter
from pint.models.timing_model import AllComponents


class KeyReturningDict(dict):
    """Dictionary that returns the key itself when key is not found."""

    def __missing__(self, key):
        return key


def get_category_mapping_from_pint() -> Dict[str, str]:
    """Get component category mappings from PINT.

    Returns:
        Dictionary mapping parameter type names to PINT category names
    """
    mapping = {
        "astrometry": "astrometry",
        "spindown": "spindown",
        "binary": "pulsar_system",
        "dispersion": "dispersion_constant",
    }

    return KeyReturningDict(mapping)


def get_extra_top_level_params_for_category() -> Dict[str, List[str]]:
    """Return extra top-level parameters to include per logical component.

    Some parameters (e.g., BINARY) are defined at the TimingModel top level in
    PINT and are not listed under any component's ``params``. This registry
    allows discovery to include such parameters in a declarative way.
    """
    return {
        "binary": ["BINARY"],
    }


@lru_cache(maxsize=1)
def _get_all_components():
    """Get cached AllComponents instance.

    Uses lru_cache to ensure AllComponents() is only created once,
    avoiding the ~10ms creation cost on subsequent calls.
    """
    return AllComponents()


def resolve_parameter_alias(param_name: str) -> str:
    """Resolve a single parameter alias to canonical name using cached AllComponents.

    This function provides fast on-demand alias resolution by leveraging the
    cached AllComponents instance, avoiding the 12.9ms creation cost.

    Args:
        param_name: Parameter name that might be an alias

    Returns:
        Canonical parameter name, or original name if not an alias
    """
    # Tempo2 uses ECCDOT; PINT canonical name is EDOT (not in AllComponents map).
    if param_name == "ECCDOT":
        return "EDOT"

    try:
        all_components = _get_all_components()
        canonical, _ = all_components.alias_to_pint_param(param_name)
        return canonical
    except Exception:
        # If alias resolution fails, return the original name
        return param_name


def pint_parameter_name(param_name: str) -> str | None:
    """Return the canonical PINT parameter name when ``param_name`` is recognized."""
    lookup = "EDOT" if param_name == "ECCDOT" else param_name
    try:
        canonical, _ = _get_all_components().alias_to_pint_param(lookup)
        return canonical
    except Exception:
        return None


def get_aliases_for_parameter(canonical_param: str) -> List[str]:
    """Get all aliases for a canonical parameter name.

    Args:
        canonical_param: The canonical parameter name

    Returns:
        List of all aliases for this parameter, including the canonical name itself
    """
    try:
        all_components = _get_all_components()
        aliases = [canonical_param]  # Start with canonical name

        # Search through the alias map to find all aliases that map to this canonical name
        alias_map = all_components._param_alias_map
        for alias, canonical in alias_map.items():
            if canonical == canonical_param and alias != canonical_param:
                aliases.append(alias)

        # Tempo2-style alias for eccentricity derivative (PINT canonical is EDOT).
        if canonical_param == "EDOT" and "ECCDOT" not in aliases:
            aliases.append("ECCDOT")

        return aliases
    except Exception:
        # If anything fails, just return the canonical name
        return [canonical_param]


def get_parameters_by_type_from_models(
    param_type: str, pint_models: Mapping[str, Any]
) -> List[str]:
    """Get parameters by type from PINT models, including dynamic derivatives and aliases.

    Args:
        param_type: Type of parameters to discover ('astrometry', 'spindown', etc.)
        pint_models: Dictionary mapping PTA names to PINT TimingModel instances

    Returns:
        List of parameter names discovered from actual models, including all aliases
    """
    from loguru import logger

    all_params = set()

    # Get category mapping
    category_mapping = get_category_mapping_from_pint()
    target_category = category_mapping[param_type]

    # Discover parameters from each PTA's actual model
    for pta_name, model in pint_models.items():
        try:
            # Extract parameters for the specific component
            for comp in model.components.values():
                if hasattr(comp, "category") and comp.category == target_category:
                    if hasattr(comp, "params"):
                        all_params.update(comp.params)  # Includes dynamic derivatives!

        except Exception as e:
            logger.warning(
                f"Failed to extract parameters from model for PTA {pta_name}: {e}"
            )
            continue

    # Build complete parameter list including all aliases
    all_params_with_aliases = set()
    for canonical_param in all_params:
        # Get all aliases for this canonical parameter
        aliases = get_aliases_for_parameter(canonical_param)
        all_params_with_aliases.update(aliases)

    # Include extra top-level params for this category if present on any model
    for extra in get_extra_top_level_params_for_category().get(param_type, []):
        # Add the extra only if at least one model has it set
        for tm in pint_models.values():
            if hasattr(tm, extra):
                try:
                    if getattr(tm, extra).value is not None:
                        all_params_with_aliases.add(extra)
                        break
                except Exception:
                    # Be robust to any attribute access issues
                    pass

    logger.debug(
        f"Component {param_type}: Found {len(all_params)} canonical parameters, {len(all_params_with_aliases)} total with aliases"
    )
    return list(all_params_with_aliases)


# ---------------------------------------------------------------------------
# Par-value unit boundary
# ---------------------------------------------------------------------------
#
# One place that knows what unit a par token or model value is in. Reading
# delegates alias resolution, ``unit_scale``/``scale_factor``/``scale_threshold``,
# ``long_double`` and Fortran ``D`` exponents to a PINT parameter object, so no
# consumer ever encodes a scale factor of its own. Emission picks the one
# spelling PINT and tempo2 read identically, and refuses when none exists.

#: Canonical unit per canonical PINT parameter name. ``A1`` is expressed in
#: ``lsec``, whose numeric value is the light-travel time in seconds. Epoch
#: parameters (``T0``, ``TASC``, ``PEPOCH``, ``START``, ``FINISH``) are
#: deliberately absent: ``MJDParameter.quantity`` is an astropy ``Time`` with
#: no ``.to()``, so epochs go through :func:`mjd_from_par` /
#: :func:`mjd_from_model` instead.
CANONICAL_SI: Dict[str, u.UnitBase] = {
    "A1": u.Unit("lsec"),
    "A1DOT": u.Unit("lsec") / u.s,
    "ECC": u.dimensionless_unscaled,
    "EDOT": u.s**-1,
    "EPS1": u.dimensionless_unscaled,
    "EPS1DOT": u.s**-1,
    "EPS2": u.dimensionless_unscaled,
    "EPS2DOT": u.s**-1,
    "OM": u.rad,
    "OMDOT": u.rad / u.s,
    "PB": u.s,
    "PBDOT": u.dimensionless_unscaled,
    "H3": u.s,
    "H4": u.s,
    "STIGMA": u.dimensionless_unscaled,
    "M2": u.Msun,
    "SINI": u.dimensionless_unscaled,
    "NE_SW": u.cm**-3,
}

#: Row-A parameters that tempo2 rescales heuristically (``|token| > 1e-7 ->
#: x1e-12``, ``readParfile.C:2134-2149``) while PINT applies its declared
#: ``1e-12/s`` unit unconditionally (``binary_ell1.py:140-158``). The only
#: tempo2-specific knowledge in this module; it exists because the two
#: packages genuinely differ. ``OMDOT`` is row A but not listed: tempo2 reads
#: it verbatim in ``deg/yr``, matching PINT's declared unit.
_TEMPO2_HEURISTIC_ONLY = frozenset({"EPS1DOT", "EPS2DOT"})

#: tempo2's shared magnitude threshold for the heuristic above.
_TEMPO2_SCALE_THRESHOLD = 1e-7


class ParUnitError(ValueError):
    """A par value cannot be read or written portably in SI."""


class _ParamDescriptor(NamedTuple):
    """Declared unit and Tempo scaling rule for one canonical parameter."""

    name: str
    units: u.UnitBase
    long_double: bool
    unit_scale: bool
    scale_factor: float
    scale_threshold: float
    is_epoch: bool


@lru_cache(maxsize=None)
def _descriptor(canonical: str) -> _ParamDescriptor:
    """Cache the declared unit and Tempo scaling rule for one parameter."""
    ac = _get_all_components()
    component = ac.param_component_map[canonical][0]
    proto = getattr(ac.components[component], canonical)
    return _ParamDescriptor(
        name=proto.name,
        units=proto.units,
        long_double=bool(getattr(proto, "long_double", False)),
        unit_scale=bool(getattr(proto, "unit_scale", False)),
        scale_factor=float(getattr(proto, "scale_factor", None) or 1.0),
        scale_threshold=float(getattr(proto, "scale_threshold", None) or math.inf),
        is_epoch=isinstance(proto, MJDParameter),
    )


def _prototype(canonical: str) -> floatParameter:
    """A fresh, unattached PINT parameter carrying the registry's convention.

    Constructed from the cached descriptor, never deep-copied: a registry
    parameter's ``_parent`` link drags in the whole component graph (~1.2 ms
    per deepcopy, measured 2026-08-05), while a bare ``floatParameter`` costs
    a few microseconds and reads identically.
    """
    d = _descriptor(canonical)
    return floatParameter(
        name=d.name,
        units=d.units,
        long_double=d.long_double,
        unit_scale=d.unit_scale,
        scale_factor=d.scale_factor if d.unit_scale else None,
        scale_threshold=d.scale_threshold if d.unit_scale else None,
        convert_tcb2tdb=False,  # audit-only object; never enters a TimingModel
    )


def has_canonical_unit(name: str) -> bool:
    """True when ``name`` (or its alias) has a declared canonical SI unit.

    The dispatch predicate for call sites that walk arbitrary par keys, such
    as the C5 passthrough audit, which must compare declared axes in SI and
    everything else as tokens.
    """
    canonical = pint_parameter_name(name)
    return canonical is not None and canonical in CANONICAL_SI


def _resolve_canonical(name: str) -> str:
    canonical = pint_parameter_name(name)
    if canonical is None:
        raise ParUnitError(f"unknown parameter name {name!r}")
    if canonical not in CANONICAL_SI:
        raise ParUnitError(
            f"no canonical SI unit declared for {canonical!r}; use mjd_from_par "
            "for epochs, or add an entry with a test"
        )
    return canonical


def si_quantity_from_token(name: str, token: Any) -> u.Quantity:
    """Interpret one par token exactly as PINT would, expressed in SI.

    Delegates alias resolution, ``unit_scale`` / ``scale_factor`` /
    ``scale_threshold``, ``long_double`` and Fortran ``D`` exponents to a PINT
    parameter object, so the caller never encodes a scale factor of its own.
    Raises :class:`ParUnitError` for a name with no declared canonical unit or
    an unparsable token.
    """
    canonical = _resolve_canonical(name)
    param = _prototype(canonical)
    try:
        # String, not float: PINT routes strings through fortran_float /
        # data2longdouble, preserving long double and Fortran D exponents.
        param.value = str(token).strip()
    except (TypeError, ValueError) as exc:
        raise ParUnitError(f"unparsable par token {token!r} for {canonical}") from exc
    return param.quantity.to(CANONICAL_SI[canonical])


def _find_par_token(par: Mapping[str, Any], *names: str) -> Optional[Tuple[str, str]]:
    """(spelling, value token) for the first present spelling, alias-aware."""
    wanted: Dict[str, None] = {}
    for name in names:
        wanted.setdefault(name.upper())
        canonical = pint_parameter_name(name)
        if canonical is not None:
            for alias in get_aliases_for_parameter(canonical):
                wanted.setdefault(alias.upper())
    for key in par:
        if key.upper() not in wanted:
            continue
        entries = par[key]
        first = entries[0] if isinstance(entries, (list, tuple)) else entries
        tokens = str(first).split()
        if not tokens:
            continue
        return key, tokens[0]
    return None


def si_from_par(
    par: Mapping[str, Any], *names: str, default: Optional[float] = None
) -> Optional[float]:
    """Alias-aware SI read of a parfile dict entry (first spelling present)."""
    found = _find_par_token(par, *names)
    if found is None:
        return default
    spelling, token = found
    return float(si_quantity_from_token(spelling, token).value)


def si_from_model(model: Any, name: str, *, default: float = 0.0) -> float:
    """SI read of a PINT ``TimingModel`` parameter via ``quantity.to_value``."""
    canonical = _resolve_canonical(name)
    if not hasattr(model, canonical):
        return default
    param = getattr(model, canonical)
    if param is None or getattr(param, "value", None) is None:
        return default
    return float(param.quantity.to_value(CANONICAL_SI[canonical]))


def token_from_si(name: str, value_si: float) -> str:
    """Inverse of :func:`si_quantity_from_token`, for par emission.

    Emits the spelling that PINT and tempo2 read identically
    and raises :class:`ParUnitError` when no
    such spelling exists. Both packages read a token through one of three
    rules — PINT's unconditional declared unit, the shared ``|token| > 1e-7 ->
    x1e-12`` heuristic, or verbatim — and the writer's job is to pick the one
    spelling both rules map back to the intended value. In particular a row-B
    value above ``scale_threshold`` must be emitted in the scaled spelling:
    written verbatim it is silently misread by *both* engines.
    """
    canonical = _resolve_canonical(name)
    if not math.isfinite(float(value_si)):
        raise ParUnitError(f"non-finite {canonical} value {value_si!r}")
    d = _descriptor(canonical)
    quantity = (float(value_si) * CANONICAL_SI[canonical]).to(d.units)
    v = float(quantity.value)

    if d.unit_scale and abs(v) > d.scale_threshold:
        emit = v / d.scale_factor  # both engines rescale it back
    elif (
        not d.unit_scale
        and canonical in _TEMPO2_HEURISTIC_ONLY
        and 0.0 < abs(v) <= _TEMPO2_SCALE_THRESHOLD
    ):
        raise ParUnitError(
            f"{canonical} {v:.6g} {d.units} has no portable spelling: PINT "
            f"applies its {d.units} unit unconditionally while tempo2 only "
            "rescales above 1e-7 (readParfile.C:2134-2149). Emit a coarser "
            "value or drop the parameter."
        )
    else:
        emit = v

    # The prototype is driven asymmetrically on purpose: reading assigns
    # ``param.value`` (a token, so the heuristic applies), emitting assigns
    # ``param.quantity`` (a Quantity, which PINT takes verbatim). The
    # round-trip pins the unit algebra to PINT; the token itself is the
    # shortest float64-exact spelling, since ``value_si`` is a float64 and
    # longer tokens would carry fake precision.
    param = _prototype(canonical)
    param.quantity = np.longdouble(emit) * d.units if d.long_double else emit * d.units
    return repr(float(param.quantity.to_value(d.units)))


def mjd_from_par(
    par: Mapping[str, Any], *names: str, default: Optional[float] = None
) -> Optional[np.longdouble]:
    """Read an epoch parameter as MJD days (row D).

    Separate from the SI accessors because ``MJDParameter.quantity`` is an
    astropy ``Time`` with no ``.to()``. There is no unit convention to resolve
    here: the token, ``param.value`` and the canonical form are all MJD days,
    in long double. Covers ``T0``, ``TASC``, ``PEPOCH``, ``START``, ``FINISH``.
    """
    found = _find_par_token(par, *names)
    if found is None:
        return None if default is None else np.longdouble(default)
    spelling, token = found
    try:
        return np.longdouble(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ParUnitError(f"unparsable epoch token {token!r} for {spelling}") from exc


def mjd_from_model(model: Any, name: str) -> Optional[np.longdouble]:
    """Model-side counterpart of :func:`mjd_from_par` (long-double MJD days)."""
    if not hasattr(model, name):
        return None
    param = getattr(model, name)
    if param is None or getattr(param, "value", None) is None:
        return None
    return np.longdouble(param.value)
