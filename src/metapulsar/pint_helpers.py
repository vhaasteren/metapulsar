"""PINT helper functions for parameter discovery and model interaction.

This module provides pure functions that encapsulate PINT-specific logic
for parameter discovery, alias resolution, and model validation.
"""

from typing import (
    Dict,
    List,
    Tuple,
    Any,
    Iterator,
    Literal,
    Mapping,
    Optional,
    TYPE_CHECKING,
)
from pint.models import TimingModel
from pint.models.model_builder import parse_parfile
from pint.models.parameter import Parameter
from pint.exceptions import PrefixError
from pint.utils import split_prefixed_name
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import tempfile
import subprocess
from io import StringIO
import re
import numpy as np

# Parameter-name utilities live in the timing package (self-contained for the
# future ``nltiming`` split); re-exported here for the rest of MetaPulsar.
from nltiming.pint_compat import (
    CANONICAL_SI as CANONICAL_SI,
    KeyReturningDict as KeyReturningDict,
    ParUnitError as ParUnitError,
    _get_all_components as _get_all_components,
    get_aliases_for_parameter as get_aliases_for_parameter,
    get_category_mapping_from_pint as get_category_mapping_from_pint,
    get_extra_top_level_params_for_category as get_extra_top_level_params_for_category,
    get_parameters_by_type_from_models as get_parameters_by_type_from_models,
    has_canonical_unit as has_canonical_unit,
    mjd_from_model as mjd_from_model,
    mjd_from_par as mjd_from_par,
    pint_parameter_name as pint_parameter_name,
    resolve_parameter_alias as resolve_parameter_alias,
    si_from_model as si_from_model,
    si_from_par as si_from_par,
    si_quantity_from_token as si_quantity_from_token,
    token_from_si as token_from_si,
)

if TYPE_CHECKING:
    from .tim_file_analyzer import TimMetadata

#: Which orthometric Shapiro expression PINT evaluates for ELL1H/``T2`` binaries.
#: ``"full"`` is PINT's default (Freire & Wex 2010, Eq. 29); ``"absorbed"``
#: (Eq. 28) matches Tempo2's ELL1H/T2 mode 1.
Ell1hShapiroMode = Literal["full", "absorbed"]


class PINTDiscoveryError(Exception):
    """Raised when PINT component discovery fails"""


def _parfile_alias_value_present(parfile_dict: Mapping[str, Any], alias: str) -> bool:
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


def resolve_parfile_parameter_name(
    canonical_name: str,
    parfile_dict: Mapping[str, Any],
    *,
    fallback: str | None = None,
) -> str:
    """Return the parfile key spelling for a canonical parameter name.

    Preference order:
    1) first alias from get_aliases_for_parameter() present with non-empty value
    2) fallback (typically the PINT model param name)
    3) canonical_name
    """
    for alias in get_aliases_for_parameter(canonical_name):
        if _parfile_alias_value_present(parfile_dict, alias):
            return alias
    if fallback is not None:
        return fallback
    return canonical_name


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


def parameter_belongs_to_component_category(
    param_name: str, component_category: str
) -> bool:
    """Return whether PINT assigns a parameter to a component category.

    This uses PINT's cached component registry rather than constructing a full
    timing model. Indexed parameters are matched to the registered prefix
    template, so arbitrary indices are recognized.
    """
    canonical = pint_parameter_name(param_name)
    if canonical is None:
        return False

    all_components = _get_all_components()
    component_names = set(all_components.param_component_map.get(canonical, ()))

    try:
        canonical_prefix = split_prefixed_name(canonical)[0]
    except PrefixError:
        canonical_prefix = None

    if canonical_prefix is not None and not component_names:
        for (
            registered,
            registered_components,
        ) in all_components.param_component_map.items():
            try:
                registered_prefix = split_prefixed_name(registered)[0]
            except PrefixError:
                continue
            if registered_prefix == canonical_prefix:
                component_names.update(registered_components)

    return any(
        component.category == component_category
        for name in component_names
        if (component := all_components.components.get(name)) is not None
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


def get_parameters_by_type_from_parfiles(
    param_type: str,
    parfile_dicts: Dict[str, Dict],
    ell1h_shapiro: Ell1hShapiroMode = "full",
) -> List[str]:
    """Get parameters by type from parfile dictionaries using PINT, including dynamic derivatives and aliases.

    Args:
        param_type: Type of parameters to discover ('astrometry', 'spindown', etc.)
        parfile_dicts: Dictionary mapping PTA names to parfile dictionaries
        ell1h_shapiro: ELL1H orthometric Shapiro convention (see ``create_pint_model``)

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
            pint_models[pta_name] = create_pint_model(
                parfile_dict, ell1h_shapiro=ell1h_shapiro
            )
        except Exception as e:
            logger.warning(f"Failed to create PINT model for PTA {pta_name}: {e}")
            continue

    # Delegate to the models-based function
    return get_parameters_by_type_from_models(param_type, pint_models)


@lru_cache(maxsize=256)
def _guess_binary_model(par_keys: frozenset) -> tuple[str, ...]:
    """Cached ``guess_binary_model`` keyed on the par's parameter names.

    PINT's guess depends only on which parameters are present, so the values are
    irrelevant. Caching matters because each call constructs an ``AllComponents``
    registry.
    """
    from pint.models.model_builder import guess_binary_model

    return tuple(guess_binary_model({key: [] for key in par_keys}))


def resolve_binary_model(parfile_dict: Mapping[str, Any]) -> Optional[str]:
    """Return the binary component PINT will build for this par.

    Tempo2's ``BINARY T2`` is a wrapper, not a model: PINT resolves it to a
    concrete component from the parameters present
    (``ModelBuilder.choose_binary_model`` with ``allow_T2=True``, which is how
    MetaPulsar builds every model). So ``T2`` may mean ``ELL1``, ``ELL1H``,
    ``DD``, ``DDK``, ``DDH`` and so on, and only PINT's own
    ``guess_binary_model`` knows which. Any explicitly declared model is
    returned as written, because PINT overrides ``BINARY`` for ``T2`` only.

    Returns None when the par has no ``BINARY`` line, or when no single PINT
    component covers every binary parameter present, which is the case PINT
    itself refuses to build.
    """
    key = _find_parfile_key(parfile_dict, "BINARY")
    if key is None:
        return None
    entries = parfile_dict[key]
    raw = entries[0] if isinstance(entries, (list, tuple)) else entries
    tokens = str(raw).split()
    if not tokens:
        return None
    declared = tokens[0].upper()
    if declared != "T2":
        return declared
    guesses = _guess_binary_model(frozenset(k.upper() for k in parfile_dict))
    return guesses[0] if guesses else None


def _find_parfile_key(parfile_dict: Mapping[str, Any], name: str) -> Optional[str]:
    """Return the actual dict key matching ``name`` case-insensitively."""
    wanted = name.upper()
    return next((key for key in parfile_dict if key.upper() == wanted), None)


def create_pint_model(
    parfile_data: Any, ell1h_shapiro: Ell1hShapiroMode = "full"
) -> TimingModel:
    """Create PINT model from parfile data (string or dict).

    Args:
        parfile_data: String content or dictionary representation of parfile
        ell1h_shapiro: Which Freire & Wex (2010) orthometric Shapiro expression PINT
            evaluates for ELL1H/``T2`` models with ``H3``+``STIG``. ``"full"`` is
            PINT's default (Eq. 29); ``"absorbed"`` (Eq. 28) is what Tempo2
            ELL1H/T2 mode 1 evaluates and is required for cross-engine residual
            parity on mixed PINT+Tempo2 stacks.

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
    from loguru import logger

    try:
        builder = ModelBuilder()

        # Handle both string and dict inputs
        if isinstance(parfile_data, str):
            parfile_data = StringIO(parfile_data)
        elif isinstance(parfile_data, dict) and "C" in parfile_data:
            # PINT's *text* parser treats a leading "C " as a comment
            # (``parse_parfile``'s ``comments=("#", "C ")``), but the dict path
            # bypasses that and warns "Unrecognized parfile line". Drop comment
            # entries so a dict round-trips like its serialized form.
            parfile_data = {k: v for k, v in parfile_data.items() if k != "C"}
        model = builder(
            parfile_data,
            allow_tcb=True,
            allow_T2=True,
            ell1h_shapiro=ell1h_shapiro,
        )

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
    from .parfile_header import format_metapulsar_par_header

    result = ""

    # Add format headers
    if format.lower() == "tempo2":
        result += "MODE 1\n"
    elif format.lower() == "pint":
        result += format_metapulsar_par_header(format="PINT")

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


def parse_par_token(param_name: str, param_value) -> Tuple[Any, bool]:
    """Split a par line into (value-as-written, is_frozen).

    The value is the raw token, NOT resolved through any unit convention: for
    Tempo-convention parameters (EPS dots, A1DOT/XDOT, PBDOT, EDOT) the token
    is not the physical value. Use ``si_from_par`` / ``si_quantity_from_token``
    for physics, and this function only for row-C reads, strings and fit
    flags (see `feature_par_units.md` §2).

    Handles the common parfile format of "value fit_status uncertainty":
    - value: the token as written (float when numeric, string otherwise)
    - fit_status: 0=frozen, 1=free (int)
    - uncertainty: optional uncertainty value

    Args:
        param_name: Name of the parameter (e.g., "DM", "DMEPOCH", "UNITS")
        param_value: Parameter value from parfile dict (string or list)

    Returns:
        Tuple of (value_as_written, is_frozen)

    Raises:
        ValueError: If parameter cannot be parsed

    Examples:
        >>> parse_par_token("DM", ["123.45 1 0.01"])
        (123.45, False)
        >>> parse_par_token("DMEPOCH", ["55000 0"])
        (55000.0, True)
        >>> parse_par_token("UNITS", ["TCB 0"])
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


def _par_value_tokens_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Compare leading par-file value tokens, numerically when possible."""
    if a is None or b is None:
        return a == b
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def dedupe_nonrepeatable_par_lines(par_text: str) -> str:
    """Collapse duplicate lines for non-repeatable parameters in par-file text.

    Old tempo2 builds (before the NE_SW guard in textOutput.C, tempo2 commit
    bf00f36) write NE_SW twice when it is explicitly set: once in the parameter
    table and once (%.3f-formatted) in the conventions block. The IPTA DR2
    dataset's own ``working/`` par files carry this signature, and PINT's
    ModelBuilder rejects such content ("Parameter X is not a repeatable
    parameter. However, multiple line use it."), so tempo2-written par content
    is sanitized at the ingestion boundary.

    Per parameter name (resolved through PINT aliases):
    - unknown to PINT (tempo2 noise lines, control lines): left untouched
    - PINT-repeatable (JUMP, EFAC, ...): left untouched
    - duplicated non-repeatable with the same leading value (numeric compare
      when possible): keep the first occurrence (the parameter-table line,
      full precision), drop the rest with a warning
    - duplicated non-repeatable with conflicting values: raise ValueError
      rather than guess
    """
    all_components = _get_all_components()
    repeatable = all_components.repeatable_param

    def _canonical(name: str) -> Optional[str]:
        try:
            canonical, _ = all_components.alias_to_pint_param(name)
            return str(canonical)
        except Exception:
            return None

    first_values: Dict[str, Optional[str]] = {}
    out_lines: List[str] = []
    for line in par_text.splitlines():
        tokens = line.split()
        if not tokens or line.lstrip().startswith("#") or line.startswith("C "):
            out_lines.append(line)
            continue
        name = tokens[0]
        canonical = _canonical(name)
        if canonical is None or name in repeatable or canonical in repeatable:
            out_lines.append(line)
            continue
        value = tokens[1] if len(tokens) > 1 else None
        if canonical not in first_values:
            first_values[canonical] = value
            out_lines.append(line)
            continue
        if _par_value_tokens_equal(first_values[canonical], value):
            loguru_logger.warning(
                f"Dropping duplicate par line for non-repeatable parameter "
                f"{canonical}: {line.strip()!r} (keeping first occurrence)"
            )
            continue
        raise ValueError(
            f"Conflicting duplicate par lines for non-repeatable parameter "
            f"{canonical}: first value {first_values[canonical]!r} vs "
            f"duplicate line {line.strip()!r}"
        )
    return "\n".join(out_lines) + ("\n" if par_text.endswith("\n") else "")


# ----------------------- Pulse-number helper utilities ----------------------- #

from pint.toa import get_TOAs, TOAs  # noqa: E402 (import after top-level defs)
from loguru import logger as loguru_logger  # noqa: E402

PulseNumberMode = Literal["no", "yes", "reuse", "overwrite"]
PULSE_NUMBER_MODES: Tuple[str, ...] = ("no", "yes", "reuse", "overwrite")
TimPulseNumberStatus = Literal["complete", "mixed", "none"]


def validate_pulse_number_mode(value: object) -> PulseNumberMode:
    """Validate and normalize use_pulse_numbers mode string."""
    if not isinstance(value, str):
        raise ValueError(
            f"use_pulse_numbers must be one of {PULSE_NUMBER_MODES!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    mode = value.strip().lower()
    if mode not in PULSE_NUMBER_MODES:
        raise ValueError(
            f"use_pulse_numbers must be one of {PULSE_NUMBER_MODES!r}, got {value!r}"
        )
    return mode  # type: ignore[return-value]


def pulse_number_tracking_enabled(mode: PulseNumberMode) -> bool:
    return mode in ("yes", "reuse", "overwrite")


def sanitize_tempo2_tim_noise_directives(tim_text: str) -> str:
    """Remove Tempo2 white-noise directive lines (T2E*, TNE*) from .tim text."""
    kept: List[str] = []
    for line in tim_text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith("#") or stripped.upper().startswith("C "):
            kept.append(line)
            continue
        first_token = stripped.split()[0]
        if first_token.startswith("T2E") or first_token.startswith("TNE"):
            continue
        kept.append(line)
    return "".join(kept)


def _should_derive_pulse_numbers(
    mode: PulseNumberMode,
    status: TimPulseNumberStatus,
    tim_path: Path,
    n_with: int,
    n_without: int,
) -> bool:
    """Return True when pulse numbers must be re-derived for this mode/status."""
    if mode == "no":
        return False
    if mode == "overwrite":
        return True
    if status == "complete":
        return False
    if status == "mixed":
        loguru_logger.warning(
            "Mixed -pn flags in {}: {} TOAs with -pn, {} without; re-deriving pulse numbers",
            tim_path,
            n_with,
            n_without,
        )
        return True
    # none
    if mode == "reuse":
        loguru_logger.warning(
            "No complete -pn in {}; re-deriving pulse numbers (reuse mode)",
            tim_path,
        )
    return True


def ensure_pint_track_minus_2(model: TimingModel) -> None:
    """Set PINT TRACK to -2 for pulse-number tracking."""
    if "TRACK" in model.params:
        model.TRACK.value = "-2"
    else:
        loguru_logger.warning(
            "TRACK parameter not in PINT timing model; "
            "pulse-number tracking may be ineffective"
        )


def ensure_pulse_numbers(
    toas: TOAs, model: TimingModel, *, force_recompute: bool = False
) -> TOAs:
    """Ensure TOAs has a complete pulse_number column.

    If a -pn flag was present in the .tim, PINT already parsed it into
    toas.table['pulse_number'] via phase_columns_from_flags(). If missing,
    compute from the model.
    """
    import numpy as np

    if "delta_pulse_number" not in toas.table.colnames:
        toas.table["delta_pulse_number"] = np.zeros(len(toas))
    if force_recompute:
        toas.compute_pulse_numbers(model)
        return toas
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
    parfile_text: str,
    tim_path: Path,
    *,
    force_recompute: bool = False,
) -> Iterator[str]:
    """Yield a temporary pn-tagged .tim derived via PINT; file is deleted on exit."""
    tim_path = Path(tim_path)
    model = create_pint_model(parfile_text)
    toas = get_TOAs(str(tim_path), model=model, include_pn=True)
    ensure_pulse_numbers(toas, model, force_recompute=force_recompute)
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
            "-nofit",
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
        sanitized = sanitize_tempo2_tim_noise_directives(
            src.read_text(encoding="utf-8")
        )
        src.write_text(sanitized, encoding="utf-8")
        yield str(src)
    # temp dir auto-removed


@contextmanager
def resolved_tim_for_pulse_numbers(
    mode: PulseNumberMode,
    parfile_text: str,
    tim_path: Path,
    *,
    derive_backend: Literal["pint", "tempo2"],
    tim_metadata: Optional["TimMetadata"] = None,
) -> Iterator[str]:
    """Yield the .tim path to load for the given pulse-number mode."""
    from .tim_file_analyzer import TimFileAnalyzer

    tim_path = Path(tim_path)
    if mode == "no":
        yield str(tim_path)
        return

    if tim_metadata is None:
        tim_metadata = TimFileAnalyzer().get_tim_metadata(tim_path)

    status = tim_metadata.pn_status
    n_with = tim_metadata.pn_with_count
    n_without = tim_metadata.pn_without_count
    derive = _should_derive_pulse_numbers(mode, status, tim_path, n_with, n_without)

    if not derive:
        yield str(tim_path)
        return

    force_recompute = mode == "overwrite"
    if derive_backend == "pint":
        with temporary_pn_tim_from_par_tim_pint(
            parfile_text,
            tim_path,
            force_recompute=force_recompute,
        ) as pn_tim:
            yield pn_tim
    else:
        with temporary_pn_tim_from_par_tim_tempo2(parfile_text, tim_path) as pn_tim:
            yield pn_tim


def _write_pn_tim_libstempo(psr, out_path: Path) -> None:
    """Write a Tempo2-format .tim with -pn flags from a libstempo tempopulsar.

    Writes FORMAT 1, then one line per observation with name, freq, MJD,
    error, type 'g', all flags, and -pn <pulse_number>. Pulse numbers must already
    be filled (e.g. by calling psr.pulsenumbers()). MODE is omitted; fit-mode
    lives on the engine-facing .par after release-tim MODE transfer.
    """
    out_path = Path(out_path)
    # Compute pulse numbers (fills obsn[].pulseN and returns array)
    pn = psr.pulsenumbers(updatebats=True, formresiduals=True, removemean=True)
    names = psr.filename()
    freqs = psr.freqs
    stoas = psr.stoas
    errs = psr.toaerrs
    flag_names = psr.flags()
    lines = ["FORMAT 1\n"]
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


SECONDS_PER_DAY_LD = np.longdouble(86400)

# tempo2 binary models whose implementation actually reads ``param_fb``.
# Verified against tempo2 source: only ELL1model.C and BTXmodel.C reference
# param_fb. ELL1H, ELL1k, DD, DDGR, DDK, and T2 silently ignore every FBn.
TEMPO2_FB_CAPABLE_BINARY_MODELS = frozenset({"ELL1", "BTX"})

# Matches FB0..FBn and the bare ``FB`` alias (PINT alias for FB0; tempo2
# readParfile.C:2055-2063 reads ``FB`` as index 0). Deliberately does not match
# FBJ / TFBJ, which are orbital-frequency jumps, a different parameter.
_FB_NAME_RE = re.compile(r"^FB(\d*)$", re.IGNORECASE)


class OrbitalChartError(ValueError):
    """A par file cannot be aligned to the canonical FBX orbital chart."""


def format_longdouble_par_value(value) -> str:
    """Format a computed par value with an exact long-double round trip.

    NORMATIVE. ``unique=True`` asks numpy for the shortest decimal string that
    reparses to the identical long double. It is the only option measured to be
    exact for *computed* values; do not substitute any of these:

    ==========================================  ============================
    Alternative                                 Why it is rejected
    ==========================================  ============================
    ``format(x, '.20g')``, f-string, ``%g``     routes through float64;
                                                rel err 2.7e-17 on J2241 FB0,
                                                ~10 ns of orbital phase over
                                                the 4409-day PPTA span
    ``format_float_scientific(precision=19)``   rel err 1.2e-21 (FB0) and
                                                2.5e-21 (sigma_FB0); the
                                                round trip is NOT exact
    ``precision=np.finfo(longdouble).precision``  failed 8895 of 20000 random
                                                long doubles; also wrong on
                                                x86-64, where ``.precision``
                                                is 18 but 21 digits are needed
    ==========================================  ============================

    ``unique=True`` had 0 failures over the same 20000-sample stress test and
    adapts automatically to 80-bit (x86-64) and 128-bit (aarch64) long doubles,
    where any fixed digit count cannot.
    """
    value = np.longdouble(value)
    if not np.isfinite(value):
        raise OrbitalChartError(f"Non-finite computed par value: {value!r}")
    return np.format_float_scientific(value, unique=True)


def _require_wide_longdouble() -> None:
    """Refuse to compute par values where ``longdouble`` is only float64.

    On some platforms (notably arm64 macOS) ``np.longdouble`` aliases float64.
    Computing FB0 there would introduce the ~10 ns error described above with
    no visible symptom, so it is a hard error rather than a warning.
    """
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise OrbitalChartError(
            "np.longdouble is not wider than float64 on this platform; "
            "orbital chart alignment would lose ~10 ns of orbital phase. "
            "Run inside the project devcontainer."
        )


def _par_token_longdouble(token: str) -> np.longdouble:
    """Parse a par token as written, accepting Fortran ``D`` exponents.

    Token reader only (row C: PB/FB0 chart alignment, where token and value
    coincide); no unit convention is resolved. Physics reads go through
    ``si_from_par`` / ``si_quantity_from_token`` instead.
    """
    try:
        return np.longdouble(token.replace("D", "E").replace("d", "e"))
    except (ValueError, TypeError) as exc:
        raise OrbitalChartError(f"Unparsable par value token {token!r}") from exc


def align_orbital_chart(
    par_text: str,
    model,
    *,
    timing_package: str,
    pta_name: str = "?",
) -> tuple[str, bool]:
    """Rewrite a par so its orbital chart matches the one its PINT model reports.

    MetaPulsar names parameters from ``TimingModel.free_params`` and then
    resolves Enterprise design-matrix columns by that name
    (``metapulsar.py:561``), so the par and the model must agree on which
    parameter carries each degree of freedom. They disagree for a published
    hybrid ``PB + FB1..FBn`` par: PINT canonicalizes it to a complete FBX series
    with free ``FB0`` (``PulsarBinary._bridge_pb_to_fb0``, upstream #2023) while
    the par -- and hence a tempo2 engine reading it -- still says ``PB``.

    This does not reimplement that canonicalization. It asks the model which
    parameter is free and, when the par does not declare it, rewrites the one
    line that does. tempo2 evaluates ``pb = 1/FB0`` whenever ``FB0`` is set
    (``ELL1model.C:75-76``), so the edit is a coordinate relabel that leaves
    residuals unchanged.

    Applies to every PTA regardless of ``timing_package`` (feature doc S1.5): an
    unaligned hybrid par used as the merge reference would reintroduce ``PB`` as
    the shared binary chart even for PTAs that needed no alignment themselves.
    ``timing_package`` selects only the tempo2 FB-capability guard.

    Args:
        par_text: Par text to align. Not mutated; a new string is returned.
        model: PINT ``TimingModel`` built from **this same** par text.
        timing_package: ``"tempo2"`` or ``"pint"``. Callers that cannot declare
            an engine must pass ``"tempo2"`` (the strict default).
        pta_name: Diagnostic label only.

    Returns:
        ``(aligned_text, changed)``. ``changed`` is False when the par already
        agrees with its model, in which case ``aligned_text is par_text``.

    Raises:
        OrbitalChartError: when a tempo2-backed par sets FB terms on a binary
            model tempo2 does not evaluate them for; when the model reports free
            ``FB0`` but the par declares neither ``FB0`` nor ``PB``; on duplicate
            ``BINARY``/``PB``/``FB`` entries; on non-finite or non-positive
            ``PB``; on a platform where ``np.longdouble`` is only float64.

    Idempotent: re-running on the returned text yields ``changed is False``.
    """
    parfile_dict = parse_parfile(StringIO(par_text))

    def _single(name: str) -> str | None:
        """Return the sole entry for ``name``, or None; raise on duplicates."""
        entries = parfile_dict.get(name)
        if not entries:
            return None
        if len(entries) > 1:
            raise OrbitalChartError(
                f"PTA {pta_name!r}: duplicate {name} entries in par content: "
                f"{entries!r}"
            )
        return entries[0]

    fb_indices: dict[int, str] = {}
    for name in parfile_dict:
        match = _FB_NAME_RE.fullmatch(name)
        if match is None:
            continue
        index = int(match.group(1)) if match.group(1) else 0
        if index in fb_indices:
            raise OrbitalChartError(
                f"PTA {pta_name!r}: FB{index} declared twice, as "
                f"{fb_indices[index]!r} and {name!r}"
            )
        fb_indices[index] = name
        _single(name)  # rejects a repeated FBn line

    # Capability guard -- tempo2 only (invariant 3). Only ELL1 and BTX read
    # param_fb in tempo2; every other binary model silently ignores FB terms, so
    # PINT (which evaluates the full series for any binary model) and tempo2
    # would be solving different physics with no error from either.
    #
    # DELIBERATELY evaluated whenever the par carries FB terms, including when no
    # rewrite follows. That is intended, not an oversight to be "optimized" into
    # the rewrite-only path: the cross-engine mismatch exists whichever spelling
    # the constant term uses. It is a behaviour change for latent bad data that
    # currently loads with silently dropped FB terms; the corpus survey (S1.2)
    # finds zero such files, and failing loudly beats loading quietly.
    if fb_indices and str(timing_package).strip().lower() == "tempo2":
        binary_entry = _single("BINARY")
        if binary_entry is None:
            raise OrbitalChartError(
                f"PTA {pta_name!r}: par sets FB parameters but declares no "
                f"BINARY model; tempo2 cannot evaluate an orbital frequency "
                f"series without one"
            )
        binary_model = binary_entry.split()[0].upper()
        if binary_model not in TEMPO2_FB_CAPABLE_BINARY_MODELS:
            raise OrbitalChartError(
                f"PTA {pta_name!r}: BINARY {binary_model} does not evaluate FB "
                f"parameters in tempo2 (only "
                f"{sorted(TEMPO2_FB_CAPABLE_BINARY_MODELS)} read param_fb), but "
                f"the par sets FB{sorted(fb_indices)}. tempo2 would silently "
                f"drop these terms while PINT evaluates them. Refusing to build "
                f"a cross-engine model that is not the same physics."
            )

    # The model is the authority on which parameter is free (invariant 2).
    if "FB0" not in model.free_params or 0 in fb_indices:
        return par_text, False

    pb_entry = _single("PB")
    if pb_entry is None:
        raise OrbitalChartError(
            f"PTA {pta_name!r}: PINT reports free FB0 but the par declares "
            f"neither FB0 nor PB; the constant orbital frequency has no source"
        )

    _require_wide_longdouble()

    # Token layout: VALUE [FITFLAG] [UNCERTAINTY]. The fit flag is copied
    # verbatim -- tempo2 accepts 0/1 and Y/N, and re-encoding would corrupt it.
    pb_tokens = pb_entry.split()
    pb_days = _par_token_longdouble(pb_tokens[0])
    if not np.isfinite(pb_days) or pb_days <= 0:
        raise OrbitalChartError(
            f"PTA {pta_name!r}: PB must be finite and positive, got {pb_days!r}"
        )

    # Computed from THIS par's PB, never from model.FB0 -- feature doc S4.3.
    fb0 = 1 / (SECONDS_PER_DAY_LD * pb_days)
    new_tokens = [format_longdouble_par_value(fb0)]
    if len(pb_tokens) >= 2:
        new_tokens.append(pb_tokens[1])
    if len(pb_tokens) >= 3:
        # sigma_FB0 = |d(FB0)/d(PB)| * sigma_PB = sigma_PB / (86400 * PB**2)
        sigma_fb0 = abs(_par_token_longdouble(pb_tokens[2])) / (
            SECONDS_PER_DAY_LD * pb_days * pb_days
        )
        new_tokens.append(format_longdouble_par_value(sigma_fb0))

    # parse_parfile already proved there is exactly one active PB entry, so the
    # first non-comment line whose leading token is PB is unambiguous.
    out_lines = par_text.splitlines()
    for index, line in enumerate(out_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if tokens[0].upper() == "C":
            continue
        if tokens[0].upper() == "PB":
            prefix = line[: len(line) - len(line.lstrip())]
            out_lines[index] = f"{prefix}{'FB0':<15}{' '.join(new_tokens)}"
            break
    else:  # pragma: no cover - parse_parfile guarantees the line exists
        raise OrbitalChartError(
            f"PTA {pta_name!r}: PB present in the parfile dict but no active "
            f"PB line found in the text"
        )

    result = "\n".join(out_lines)
    if par_text.endswith("\n"):
        result += "\n"

    loguru_logger.info(
        f"PTA {pta_name!r}: aligned orbital chart to canonical FBX: "
        f"PB={pb_tokens[0]} -> FB0={new_tokens[0]} "
        f"(FB{sorted(fb_indices)} present, timing_package={timing_package!r})"
    )
    if "PBDOT" in parfile_dict and 1 in fb_indices:
        # Both engines resolve this the same way -- explicit FB1 wins and PBDOT
        # is ignored (tempo2 ELL1model.C:134-142; PINT #2023 likewise) -- so the
        # par is not rewritten further. Log it: a reader seeing a retained PBDOT
        # next to FB1 should know it is inert, not applied.
        loguru_logger.warning(
            f"PTA {pta_name!r}: par sets both PBDOT and FB1; explicit FB1 wins "
            f"in both tempo2 and PINT and the retained PBDOT is inert."
        )
    return result, True


def par_text_with_track_minus_2(par_text: str) -> str:
    """Return par text with TRACK set to -2 using line-based editing.

    Avoids PINT re-serialization, which can fail on Tempo2 fit flags (``N``/``Y``).
    """
    out_lines: List[str] = []
    replaced = False
    for line in par_text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.upper().startswith("C ")
        ):
            out_lines.append(line)
            continue
        first_token = stripped.split()[0]
        if first_token.upper() == "TRACK":
            prefix = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{prefix}TRACK -2")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append("TRACK -2")
    result = "\n".join(out_lines)
    if par_text.endswith("\n"):
        result += "\n"
    return result


@contextmanager
def temporary_par_with_track_minus_2(par_text: str) -> Iterator[str]:
    """Yield a temporary tempo2-formatted par file with TRACK -2; deleted on exit."""
    par_out = par_text_with_track_minus_2(par_text)
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
