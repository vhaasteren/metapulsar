"""PINT helper functions for parameter discovery and model interaction.

This module provides pure functions that encapsulate PINT-specific logic
for parameter discovery, alias resolution, and model validation.
"""

from typing import Dict, List, Tuple, Any, Iterator
from functools import lru_cache
from pint.models import TimingModel
from pint.models.timing_model import AllComponents
from pint.models.parameter import Parameter
from contextlib import contextmanager
from pathlib import Path
import tempfile
import subprocess
from io import StringIO
import numpy as np
from pint.models.model_builder import parse_parfile


class PINTDiscoveryError(Exception):
    """Raised when PINT component discovery fails"""


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
    # Handle special case: EDOT -> ECCDOT alias that PINT doesn't have
    if param_name == "EDOT":
        return "ECCDOT"

    try:
        all_components = _get_all_components()
        canonical, _ = all_components.alias_to_pint_param(param_name)
        return canonical
    except Exception:
        # If alias resolution fails, return the original name
        return param_name


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

        # Handle special case: if canonical is ECCDOT, also include EDOT
        if canonical_param == "ECCDOT" and "EDOT" not in aliases:
            aliases.append("EDOT")

        return aliases
    except Exception:
        # If anything fails, just return the canonical name
        return [canonical_param]


def _parfile_alias_value_present(parfile_dict: Dict[str, Any], alias: str) -> bool:
    """Return True if alias is present in a parfile dict with a non-empty value."""
    if alias not in parfile_dict:
        return False
    value = parfile_dict[alias]
    if value is None:
        return False
    if isinstance(value, list):
        if not value:
            return False
        return bool(str(value[0]).strip())
    return bool(str(value).strip())


def has_parameter_alias(parfile_dict: Dict[str, Any], canonical_param: str) -> bool:
    """Return True if any PINT alias for canonical_param is present and non-empty."""
    return any(
        _parfile_alias_value_present(parfile_dict, alias)
        for alias in get_aliases_for_parameter(canonical_param)
    )


def has_equatorial_astrometry(parfile_dict: Dict[str, Any]) -> bool:
    """Return True if equatorial astrometry parameters are present (via PINT aliases)."""
    return has_parameter_alias(parfile_dict, "RAJ") and has_parameter_alias(
        parfile_dict, "DECJ"
    )


def has_ecliptic_astrometry(parfile_dict: Dict[str, Any]) -> bool:
    """Return True if ecliptic astrometry parameters are present (via PINT aliases)."""
    return has_parameter_alias(parfile_dict, "ELONG") and has_parameter_alias(
        parfile_dict, "ELAT"
    )


def detect_astrometry_style(parfile_dict: Dict[str, Any]) -> str:
    """Detect whether a parfile uses equatorial or ecliptic astrometry.

    Uses PINT's alias map rather than hard-coded parameter names.
    """
    has_equatorial = has_equatorial_astrometry(parfile_dict)
    has_ecliptic = has_ecliptic_astrometry(parfile_dict)

    if has_equatorial and has_ecliptic:
        raise ValueError(
            "Mixed astrometry detected (equatorial and ecliptic parameters present). "
            "Refuse to make ambiguous coordinate representation consistent."
        )
    if has_ecliptic:
        return "ecliptic"
    if has_equatorial:
        return "equatorial"
    raise ValueError(
        "Could not detect astrometry style. Expected either RAJ/DECJ or "
        "LAMBDA/BETA (or ELONG/ELAT)."
    )


def clear_all_components_cache():
    """Clear the AllComponents cache.

    This is useful for testing to ensure clean state between tests.
    """
    _get_all_components.cache_clear()


def check_component_available_in_model(model: TimingModel, component_type: str) -> bool:
    """Check if component type is available in a single PINT model.

    Args:
        model: PINT TimingModel instance
        component_type: Type of component to check ('astrometry', 'spindown', etc.)

    Returns:
        True if component is available in the model
    """
    from loguru import logger

    # Discover category mapping from PINT
    category_mapping = get_category_mapping_from_pint()

    if component_type not in category_mapping:
        logger.warning(f"Unknown component type: {component_type}")
        return False

    target_category = category_mapping[component_type]

    # Check if any component with the target category is available
    for component in model.components.values():
        if hasattr(component, "category") and component.category == target_category:
            logger.debug(f"Found component with category '{target_category}' in model")
            return True

    logger.debug(f"No component with category '{target_category}' found in model")
    return False


def get_parameter_identifiability_from_model(
    model: TimingModel, param_name: str
) -> bool:
    """Check if parameter is identifiable in a single PINT model.

    Args:
        model: PINT TimingModel instance
        param_name: Name of parameter to check

    Returns:
        True if parameter is fittable and free (identifiable)
    """
    from loguru import logger

    # Check if parameter is fittable (has derivatives implemented)
    if param_name not in model.fittable_params:
        logger.debug(f"Parameter '{param_name}' not fittable (no derivatives)")
        return False

    # Check if parameter is free (unfrozen)
    if param_name not in model.free_params:
        logger.debug(f"Parameter '{param_name}' not in free_params")
        return False

    logger.debug(f"Parameter '{param_name}' is identifiable (fittable and free)")
    return True


def get_parameters_by_type_from_models(
    param_type: str, pint_models: Dict[str, TimingModel]
) -> List[str]:
    """Get parameters by type from PINT models, including dynamic derivatives and aliases.

    Args:
        param_type: Type of parameters to discover ('astrometry', 'spindown', etc.)
        pint_models: Dictionary mapping PTA names to PINT TimingModel instances

    Returns:
        List of parameter names discovered from actual models, including all aliases

    Raises:
        PINTDiscoveryError: If parameter extraction fails
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


def get_parameters_by_type_from_parfiles(
    param_type: str, parfile_dicts: Dict[str, Dict]
) -> List[str]:
    """Get parameters by type from parfile dictionaries using PINT, including dynamic derivatives and aliases.

    Args:
        param_type: Type of parameters to discover ('astrometry', 'spindown', etc.)
        parfile_dicts: Dictionary mapping PTA names to parfile dictionaries

    Returns:
        List of parameter names discovered from actual parfiles, including all aliases

    Raises:
        PINTDiscoveryError: If PINT model creation fails
    """
    from loguru import logger

    # Create PINT models from parfile dictionaries
    pint_models = {}
    for pta_name, parfile_dict in parfile_dicts.items():
        try:
            pint_models[pta_name] = create_pint_model(parfile_dict)
        except Exception as e:
            logger.warning(f"Failed to create PINT model for PTA {pta_name}: {e}")
            continue

    # Delegate to the models-based function
    return get_parameters_by_type_from_models(param_type, pint_models)


def create_pint_model(parfile_data) -> TimingModel:
    """Create PINT model from parfile data (string or dict).

    Args:
        parfile_data: String content or dictionary representation of parfile

    Returns:
        PINT TimingModel instance

    Raises:
        PINTDiscoveryError: If model creation fails
    """
    from pint.models.model_builder import ModelBuilder
    from pint.exceptions import (
        TimingModelError,
        MissingParameter,
        UnknownParameter,
        UnknownBinaryModel,
        InvalidModelParameters,
        ComponentConflict,
    )
    from io import StringIO
    from loguru import logger

    try:
        builder = ModelBuilder()

        # Handle both string and dict inputs
        if isinstance(parfile_data, str):
            model = builder(StringIO(parfile_data), allow_tcb=True, allow_T2=True)
        else:  # dict
            model = builder(parfile_data, allow_tcb=True, allow_T2=True)

        return model
    except (
        TimingModelError,
        MissingParameter,
        UnknownParameter,
        UnknownBinaryModel,
        InvalidModelParameters,
        ComponentConflict,
    ) as e:
        logger.error(f"PINT model creation failed: {e}")
        raise  # Re-raise the original exception
    except Exception as e:
        logger.error(f"Unexpected error creating PINT model: {e}")
        raise PINTDiscoveryError(f"Unexpected error creating PINT model: {e}")


def dict_to_parfile_string(parfile_dict: Dict, format: str = "pint") -> str:
    """Convert parfile dictionary to string using PINT's exact formatting.

    Simple approach that preserves ALL parameters without complex categorization.

    Args:
        parfile_dict: Dictionary representation of parfile
        format: Output format ('pint', 'tempo', 'tempo2')

    Returns:
        Formatted parfile string using PINT's exact formatting
    """
    from datetime import datetime

    result = ""

    # Add format headers
    if format.lower() == "tempo2":
        result += "MODE 1\n"
    elif format.lower() == "pint":
        result += "# Created: " + datetime.now().isoformat() + "\n"
        result += "# Format:  PINT\n"
        result += "# By:      MetaPulsar\n"

    # Format ALL parameters using PINT's exact formatting
    for param_name, param_data in parfile_dict.items():
        if len(param_data) >= 1:
            # Handle multiple instances of the same parameter (e.g., multiple JUMP parameters)
            # Multiple instances are detected when we have multiple separate string values
            if isinstance(param_data[0], str) and len(param_data) > 1:
                # Multiple string values - iterate through all of them
                for value in param_data:
                    # Create Parameter object and use PINT's exact formatting
                    param = Parameter()
                    param.name = param_name
                    param.quantity = value
                    param.frozen = True  # Default to frozen for multiple instances

                    result += param.as_parfile_line(format=format)
            else:
                # Single value or list format
                value = param_data[0]
                # Handle different parfile dictionary formats
                if len(param_data) >= 2:
                    frozen = param_data[1] == "0"
                else:
                    frozen = True  # Default to frozen if not specified

                # Create Parameter object and use PINT's exact formatting
                param = Parameter()
                param.name = param_name
                param.quantity = value
                param.frozen = frozen
                # Note: uncertainty cannot be set directly on Parameter object
                # PINT handles uncertainty formatting in as_parfile_line()

                result += param.as_parfile_line(format=format)

    return result


def create_minimal_parfile_for_component(parfile_dict: Dict, component) -> str:
    """Create minimal parfile for component discovery using PINT component system.

    Args:
        parfile_dict: Parsed parfile dictionary
        component: String or list of strings specifying component(s) to include.
                  Spindown is always included as PINT requires it.
    """
    # Normalize component to list
    if isinstance(component, str):
        components = [component]
    else:
        components = list(component)

    # Always include spindown - PINT cannot process parfile without it
    if "spindown" not in components:
        components.append("spindown")

    # Create PINT model from parfile dictionary
    model = create_pint_model(parfile_dict)

    # Get category mapping from PINT
    category_mapping = get_category_mapping_from_pint()

    # Extract parameters from all requested components
    component_params = set()
    for comp_name in components:
        target_category = category_mapping.get(comp_name)
        if not target_category:
            continue

        for comp in model.components.values():
            if hasattr(comp, "category") and comp.category == target_category:
                if hasattr(comp, "params"):
                    component_params.update(comp.params)

    # Create minimal parfile content
    minimal_lines = []
    for param in component_params:
        if param in parfile_dict:
            value = parfile_dict[param]
            if isinstance(value, list):
                value_str = " ".join(str(v) for v in value)
            else:
                value_str = str(value)
            minimal_lines.append(f"{param} {value_str}")

    return "\n".join(minimal_lines)


def parse_parameter_using_pint(param_name: str, param_value) -> Tuple[Any, bool]:
    """Parse parameter value using PINT's parsing approach.

    This function elegantly handles parfile parameter parsing by extracting
    the parsing logic from PINT's Parameter.from_parfile_line() method.
    It handles the common parfile format of "value fit_status uncertainty" where:
    - value: the parameter value (float for numeric params, string for text params)
    - fit_status: 0=frozen, 1=free (int)
    - uncertainty: optional uncertainty value

    Args:
        param_name: Name of the parameter (e.g., "DM", "DMEPOCH", "UNITS")
        param_value: Parameter value from parfile dict (string or list)

    Returns:
        Tuple of (parsed_value, is_frozen)

    Raises:
        ValueError: If parameter cannot be parsed

    Examples:
        >>> parse_parameter_using_pint("DM", ["123.45 1 0.01"])
        (123.45, False)
        >>> parse_parameter_using_pint("DMEPOCH", ["55000 0"])
        (55000.0, True)
        >>> parse_parameter_using_pint("UNITS", ["TCB 0"])
        ("TCB", True)
    """
    # Handle list format from parse_parfile
    if isinstance(param_value, list):
        param_str = param_value[0]
    else:
        param_str = str(param_value)

    # Split the parameter string into components
    parts = param_str.split()

    if not parts:
        raise ValueError(f"Empty parameter value for {param_name}: {param_value}")

    # Parse value (first component)
    value = parts[0]

    # Try to convert to float for numeric parameters, keep as string for text parameters
    try:
        value = float(value)
    except ValueError:
        # Keep as string for non-numeric parameters like UNITS
        pass

    # Parse fit_status (second component, default to free if not specified)
    is_frozen = False  # Default to free
    if len(parts) > 1:
        try:
            fit_status = int(parts[1])
            is_frozen = fit_status == 0
        except ValueError:
            # If second part is not an integer, treat as uncertainty and assume free
            pass

    return value, is_frozen


# ----------------------- Pulse-number helper utilities ----------------------- #

from pint.toa import get_TOAs, TOAs  # noqa: E402 (import after top-level defs)


def ensure_pulse_numbers(toas: TOAs, model: TimingModel) -> TOAs:
    """Ensure TOAs has a complete pulse_number column.

    If a -pn flag was present in the .tim, PINT already parsed it into
    toas.table['pulse_number'] via phase_columns_from_flags(). If missing,
    compute from the model.
    """
    import numpy as np

    if "delta_pulse_number" not in toas.table.colnames:
        toas.table["delta_pulse_number"] = np.zeros(len(toas))
    if ("pulse_number" not in toas.table.colnames) or (
        toas.table["pulse_number"] != toas.table["pulse_number"]
    ).any():  # NaN check
        toas.compute_pulse_numbers(model)
    return toas


def write_pn_tim(toas: TOAs, out_path: Path) -> Path:
    """Write a Tempo2-format .tim with -pn flags from a TOAs table."""
    out_path = Path(out_path)
    toas.write_TOA_file(str(out_path), format="Tempo2", include_pn=True)
    return out_path


@contextmanager
def temporary_pn_tim_from_par_tim_pint(
    parfile_text: str, tim_path: Path
) -> Iterator[str]:
    """Yield a temporary pn-tagged .tim derived via PINT; file is deleted on exit."""
    tim_path = Path(tim_path)
    model = create_pint_model(parfile_text)
    toas = get_TOAs(str(tim_path), model=model, include_pn=True)
    ensure_pulse_numbers(toas, model)
    with tempfile.TemporaryDirectory(prefix="withpn_pint_") as td:
        out_path = Path(td) / "withpn.tim"
        write_pn_tim(toas, out_path)
        yield str(out_path)
    # temp dir auto-removed


@contextmanager
def temporary_pn_tim_from_par_tim_tempo2(
    parfile_text: str, tim_path: Path
) -> Iterator[str]:
    """Yield a temporary pn-tagged .tim via tempo2 output plugin; deleted on exit."""
    tim_path = Path(tim_path).resolve()

    # Preflight: check file exists and is readable
    if not tim_path.exists():
        raise FileNotFoundError(f"Tim file not found: {tim_path}")
    if not tim_path.is_file():
        raise ValueError(f"Tim path is not a file: {tim_path}")

    with tempfile.TemporaryDirectory(prefix="withpn_t2_") as td:
        td_path = Path(td)
        par_tmp = td_path / "orig.par"
        par_tmp.write_text(parfile_text, encoding="utf-8")
        cmd = [
            "tempo2",
            "-f",
            str(par_tmp),
            str(tim_path),
            "-output",
            "add_pulseNumber",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=str(td_path),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Sanitize error: extract relevant error lines, skip warranty banner
            # Combine stderr and stdout, prefer stderr for actual errors
            combined_output = (e.stderr or "").strip()
            if not combined_output or len(combined_output) < 20:
                # If stderr is empty/short, check stdout
                combined_output = (e.stdout or "").strip()

            error_lines = []
            warranty_keywords = [
                "warranty",
                "gpl",
                "free software",
                "absolutely no",
                "redistribute",
                "this program comes",
                "welcome to redistribute",
            ]

            for line in combined_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                line_lower = line.lower()
                # Skip warranty/license text more aggressively
                if any(keyword in line_lower for keyword in warranty_keywords):
                    continue
                # Keep ERROR lines, "Unable to open", and assertion failures
                if (
                    line.startswith("ERROR")
                    or "Unable to open" in line
                    or "Assertion" in line
                    or "failed" in line_lower
                    or line.startswith("tempo2:")
                ):
                    error_lines.append(line)

            # If still no errors found after filtering, try to find any non-warranty line
            if not error_lines:
                for line in combined_output.splitlines():
                    line = line.strip()
                    if line and not any(
                        keyword in line.lower() for keyword in warranty_keywords
                    ):
                        error_lines.append(line)
                        break

            error_msg = (
                "\n".join(error_lines) if error_lines else "tempo2 command failed"
            )
            raise RuntimeError(f"tempo2 add_pulseNumber failed: {error_msg}") from e
        src = td_path / "withpn.tim"
        if not src.exists():
            raise RuntimeError("tempo2 plugin did not produce withpn.tim")
        yield str(src)
    # temp dir auto-removed


def _write_pn_tim_libstempo(psr, out_path: Path) -> None:
    """Write a Tempo2-format .tim with -pn flags from a libstempo tempopulsar.

    Writes FORMAT 1, MODE 1, then one line per observation with name, freq, MJD,
    error, type 'g', all flags, and -pn <pulse_number>. Pulse numbers must already
    be filled (e.g. by calling psr.pulsenumbers()).
    """
    out_path = Path(out_path)
    # Compute pulse numbers (fills obsn[].pulseN and returns array)
    pn = psr.pulsenumbers(updatebats=True, formresiduals=True, removemean=True)
    names = psr.filename()
    freqs = psr.freqs
    stoas = psr.stoas
    errs = psr.toaerrs
    flag_names = psr.flags()
    lines = ["FORMAT 1\n", "MODE 1\n"]
    for i in range(psr.nobs):
        # name freq mjd error type
        # Keep TOA MJD in longdouble precision; do not downcast to float64.
        mjd = np.longdouble(stoas[i])
        mjd_str = np.format_float_positional(
            mjd,
            precision=30,
            unique=False,
            trim="k",
        )
        flag_parts = []
        for f in flag_names:
            val = psr.flagvals(f)[i]
            if val:
                flag_parts.append(f" -{f} {val}")
        flag_parts.append(f" -pn {int(pn[i])}")
        flag_str = "".join(flag_parts)
        name = str(names[i]).strip()
        freq = float(freqs[i])
        err = float(errs[i])
        line = f" {name} {freq:.5f} {mjd_str} {err:.5f} g{flag_str}\n"
        lines.append(line)
    out_path.write_text("".join(lines), encoding="utf-8")


@contextmanager
def temporary_pn_tim_from_par_tim_libstempo(
    parfile_text: str, tim_path: Path
) -> Iterator[str]:
    """Yield a temporary pn-tagged .tim via libstempo; deleted on exit.

    Uses libstempo to load par + tim, compute pulse numbers (same as tempo2),
    and write a .tim file with -pn flags. Equivalent in outcome to
    temporary_pn_tim_from_par_tim_tempo2 but without calling the tempo2 binary.

    Requires libstempo (and a tempo2 runtime) to be installed.
    """
    try:
        import libstempo as t2
    except ImportError as e:
        raise ImportError(
            "temporary_pn_tim_from_par_tim_libstempo requires libstempo. "
            "Install with: conda install -c conda-forge libstempo"
        ) from e

    tim_path = Path(tim_path).resolve()
    if not tim_path.exists():
        raise FileNotFoundError(f"Tim file not found: {tim_path}")
    if not tim_path.is_file():
        raise ValueError(f"Tim path is not a file: {tim_path}")

    with tempfile.TemporaryDirectory(prefix="withpn_libstempo_") as td:
        td_path = Path(td)
        par_tmp = td_path / "orig.par"
        par_tmp.write_text(parfile_text, encoding="utf-8")
        out_tim = td_path / "withpn.tim"
        psr = t2.tempopulsar(parfile=str(par_tmp), timfile=str(tim_path), dofit=False)
        _write_pn_tim_libstempo(psr, out_tim)
        yield str(out_tim)


@contextmanager
def temporary_par_with_track_minus_2(par_text: str) -> Iterator[str]:
    """Yield a temporary tempo2-formatted par file with TRACK -2; deleted on exit."""
    par_dict = parse_parfile(StringIO(par_text))
    par_dict["TRACK"] = ["-2"]
    par_out = dict_to_parfile_string(par_dict, format="tempo2")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".par", delete=False) as tf:
        tf.write(par_out)
        tf.flush()
        path = tf.name
    try:
        yield path
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
