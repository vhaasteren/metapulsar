"""PINT parameter-name utilities used by the timing package.

Pure functions over PINT's alias tables and component registry. This module
must not import from the wider ``metapulsar`` package: the dependency
direction is ``metapulsar`` → ``metapulsar.timing``, never the reverse.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Mapping

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
