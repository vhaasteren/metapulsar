"""Unified parameter and par file management for multi-PTA pulsar data.

This module consolidates all parameter management functionality:
- Making par files consistent across PTAs
- Building parameter mappings for MetaPulsar
- Resolving parameter aliases and availability
- Working with both PINT and Tempo2 PTAs
"""

from __future__ import annotations

import re
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from io import StringIO
from typing import Dict, Iterable, List, Any, Literal, Tuple, Optional, Set
import logging

from pint.models.model_builder import parse_parfile
from pint.models.timing_model import TimingModel

from .pint_helpers import (
    Ell1hShapiroMode,
    resolve_parameter_alias,
    get_aliases_for_parameter,
    create_pint_model,
    get_parameters_by_type_from_models,
    check_component_available_in_model,
    get_parameter_identifiability_from_model,
    dict_to_parfile_string,
    dedupe_nonrepeatable_par_lines,
    parse_parameter_using_pint,
    detect_astrometry_style,
)

logger = logging.getLogger(__name__)


# ===== ALIGNMENT POLICY =====

UnsupportedPolicy = Literal["strip", "error"]


@dataclass(frozen=True)
class AlignmentPolicy:
    """Policy for the multi-PTA ``consistent`` combination strategy.

    The consistent strategy rewrites every PTA's par file onto one common
    PINT/Tempo2 deterministic surface. Terms outside that surface are stripped
    by default; ``unsupported="error"`` turns them into a hard failure instead.
    Everything else in the profile (TDB, IERS2003, IAU2000B, FB90, and the
    boolean switches) is fixed, because only that combination is validated for
    cross-engine residual parity.

    Attributes:
        unsupported: ``"strip"`` (default) removes unsupported deterministic
            families with a warning; ``"error"`` raises listing every offender.
        ephem: Override the reference PTA's ``EPHEM``.
        clock: Override the reference PTA's ``CLOCK``/``CLK``.
        bipm_version: Year used to resolve a bare ``TT(BIPM)`` realization.
        ne_sw: Override the resolved constant solar-wind density in cm^-3.
    """

    unsupported: UnsupportedPolicy = "strip"
    ephem: Optional[str] = None
    clock: Optional[str] = None
    bipm_version: Optional[int] = None
    ne_sw: Optional[float] = None

    def __post_init__(self) -> None:
        if self.unsupported not in {"strip", "error"}:
            raise ValueError("unsupported must be 'strip' or 'error'")
        if self.ne_sw is not None and self.ne_sw < 0:
            raise ValueError("ne_sw must be non-negative")


def normalize_timing_package(package: Optional[str]) -> str:
    """Normalize a timing-package label ('libstempo' is spelled 'tempo2')."""
    normalized = (package or "pint").strip().lower()
    return "tempo2" if normalized == "libstempo" else normalized


def resolve_ell1h_shapiro_mode(
    timing_packages: Iterable[Optional[str]],
) -> Ell1hShapiroMode:
    """Return the ELL1H Shapiro convention PINT loads should use for a stack.

    Mixed PINT+Tempo2 stacks need PINT's ``"absorbed"`` evaluator (Freire & Wex
    2010, Eq. 28) to reproduce Tempo2's ELL1H/T2 mode-1 delay for the same
    printed ``(A1, EPS1, H3, STIG)``. Every other stack keeps PINT's default
    ``"full"`` (Eq. 29) so published PINT-only solutions are not changed.
    """
    packages = {normalize_timing_package(package) for package in timing_packages}
    return "absorbed" if {"pint", "tempo2"}.issubset(packages) else "full"


# ===== TEMPO1 AGGREGATE MODE =====

#: The six explicit states Tempo2's aggregate ``TEMPO1`` switch selects at once.
TEMPO1_DEFAULTS: Dict[str, List[str]] = {
    "UNITS": ["TDB"],
    "TIMEEPH": ["FB90"],
    "DILATEFREQ": ["N"],
    "PLANET_SHAPIRO": ["N"],
    "T2CMETHOD": ["TEMPO"],
    "CORRECT_TROPOSPHERE": ["N"],
}


def expand_tempo1(par: Dict[str, List[str]]) -> List[str]:
    """Replace an aggregate ``TEMPO1`` line with its six explicit states.

    Returns the names that were filled in (empty when ``TEMPO1`` is absent).
    Source-line ordering is irrelevant here because the common profile
    normalizes all six states afterwards.
    """
    tempo1_keys = [key for key in par if key.upper() == "TEMPO1"]
    if not tempo1_keys:
        return []
    for key in tempo1_keys:
        par.pop(key)
    present = {key.upper() for key in par}
    filled = []
    for key, value in TEMPO1_DEFAULTS.items():
        if key not in present:
            par[key] = list(value)
            filled.append(key)
    return filled


# ===== UNSUPPORTED DETERMINISTIC FAMILIES =====
#
# Matching is intentionally explicit: exact names, anchored prefixes, and
# anchored regular expressions. There is deliberately no generic "starts with
# DM/CM/TN" rule, because that would swallow noise hyperparameters (EFAC,
# EQUAD, ECORR, TNRedAmp, TNDMAmp, DMJUMP, ...) which are out of scope here.

#: Tempo2-only or PINT-unsafe surfaces, removed whenever PINT is in the stack.
_PINT_UNSAFE_EXACT = {
    "EPHEM_FILE",
    "EPH_FILE",
    "EOP_FILE",
    "CLK_CORR_CHAIN",
    "NE_SW_SIN",
    "NE_SW_IFUNC",
    "_NE_SW",
    "DMMODEL",
    "_DM",
    "_CM",
    "DMOFF",
    "SATJUMP",
    "PMRA2",
    "PMDEC2",
    "PMLAMBDA2",
    "PMBETA2",
    "PMELONG2",
    "PMELAT2",
    "PMRV",
    "DSHK",
    "D_AOP",
    "STEL_DX",
    "TELEPOCH",
}

#: PINT-only or Tempo2-unvalidated surfaces, removed whenever Tempo2 is in the stack.
_TEMPO2_UNSAFE_EXACT = {
    "SWP",
    "SWEPOCH",
    "VLBIAX",
    "VLBIAY",
    "VLBIAZ",
    "DMWXEPOCH",
}

#: Deterministic extensions outside the common mixed-engine surface.
_MIXED_UNSAFE_EXACT = {
    "WAVEEPOCH",
    "WAVE_OM",
    "WXEPOCH",
    "SIFUNC",
    "CM",
    "CMEPOCH",
    "CMWXEPOCH",
    "DMWXEPOCH",
    "EXPDIPEPS",
    "EXPDIPFREF",
    "CHROMGAUSS_FREF",
    "TNDMEVENT",
    "TNSHAPELETEVENT",
}

_MIXED_UNSAFE_PREFIXES = (
    "CMX_",
    "CMWXFREQ_",
    "CMWXSIN_",
    "CMWXCOS_",
    "CHROMX",
    "WXFREQ_",
    "WXSIN_",
    "WXCOS_",
    "EXPDIPEP_",
    "EXPDIPAMP_",
    "EXPDIPIDX_",
    "EXPDIPTAU_",
    "CHROMGAUSS_",
    "EXPEP_",
    "EXPPH_",
    "EXPTAU_",
    "EXPINDEX_",
    "GAUSEP_",
    "GAUSAMP_",
    "GAUSSIG_",
    "GAUSINDEX_",
    "PWSTART_",
    "PWSTOP_",
    "PWEP_",
    "PWPH_",
    "PWF0_",
    "PWF1_",
    "PWF2_",
)

_WAVE_RE = re.compile(r"^WAVE\d+$")
_IFUNC_RE = re.compile(r"^IFUNC\d+$")
_CM_DERIVATIVE_RE = re.compile(r"^CM\d+$")
_GLITCH_RE = re.compile(r"^(?:GLEP|GLPH|GLF0|GLF1|GLF2|GLF0D|GLTD|GLF0D2|GLTD2)_?\d+$")
_TELESCOPE_RE = re.compile(r"^(?:TELX|TELY|TELZ|TEL_DX)(?:_?\d+)?$")
_PLANET_PERTURBATION_RE = re.compile(r"^(?:DMASSPLANET|DPHASEPLANET)[1-9]$")

#: Tempo2 controls that stay in the written par but must not reach PINT's
#: ModelBuilder when a temporary model is built during alignment.
_TEMPO2_LOCAL_CONTROLS = {"IPM", "FDDC", "FDDI"}

#: Canonical fields copied back after a numeric ecliptic transformation.
#: Ordered so the rewritten par keeps a deterministic layout.
_ECL_COPY_KEYS = (
    "ELONG",
    "ELAT",
    "PMELONG",
    "PMELAT",
    "POSEPOCH",
    "ECL",
    "KOM",  # present for orientation-dependent DDK cases
)

#: Every input spelling replaced by the transformed astrometry (upper case).
_ECL_INPUT_ALIASES = frozenset(
    {
        "ELONG",
        "LAMBDA",
        "ELAT",
        "BETA",
        "PMELONG",
        "PMLAMBDA",
        "PMELAT",
        "PMBETA",
        "ECL",
    }
)

_BIPM_CLOCK_RE = re.compile(r"^TT\(BIPM(?P<year>\d{4})?\)$", re.IGNORECASE)


def _is_numbered(key: str, stem: str) -> bool:
    """True when ``key`` is ``stem`` followed by a bare integer index."""
    if not key.startswith(stem):
        return False
    return key[len(stem) :].isdigit()


def _is_pint_unsafe(key: str) -> bool:
    """True for surfaces PINT cannot represent (checked when PINT is present)."""
    return key in _PINT_UNSAFE_EXACT or _TELESCOPE_RE.fullmatch(key) is not None


def _is_tempo2_unsafe(key: str) -> bool:
    """True for surfaces Tempo2 cannot represent (checked when Tempo2 is present)."""
    return (
        key in _TEMPO2_UNSAFE_EXACT
        or _is_numbered(key, "NE_SW")
        or key.startswith(("SWX", "DMWX"))
    )


def _is_mixed_unsafe(key: str) -> bool:
    """True for deterministic terms outside the mixed-engine common surface."""
    return (
        key in _MIXED_UNSAFE_EXACT
        or key.startswith(_MIXED_UNSAFE_PREFIXES)
        or _WAVE_RE.fullmatch(key) is not None
        or _IFUNC_RE.fullmatch(key) is not None
        or _CM_DERIVATIVE_RE.fullmatch(key) is not None
        or _GLITCH_RE.fullmatch(key) is not None
        or _PLANET_PERTURBATION_RE.fullmatch(key) is not None
    )


def _first_token(value: Optional[List[str]]) -> Optional[str]:
    """Return the leading whitespace-separated token of a parsed par value."""
    if not value:
        return None
    parts = str(value[0]).split()
    return parts[0] if parts else None


def dmmodel_constraint_values(par: Dict[str, List[str]]) -> List[str]:
    """Return ``CONSTRAIN`` entries that constrain a DMMODEL grid."""
    return [
        value
        for value in par.get("CONSTRAIN", [])
        if value.split() and value.split()[0].upper().startswith("DMMODEL")
    ]


def strip_dmmodel_constraints(par: Dict[str, List[str]]) -> List[str]:
    """Drop DMMODEL ``CONSTRAIN`` entries, keeping unrelated constraints."""
    old = par.get("CONSTRAIN", [])
    removed = dmmodel_constraint_values(par)
    keep = [value for value in old if value not in removed]
    if keep:
        par["CONSTRAIN"] = keep
    else:
        par.pop("CONSTRAIN", None)
    return removed


def _set_boolean(par: Dict[str, List[str]], key: str, value: bool) -> None:
    par[key] = ["Y" if value else "N"]


class ParameterManager:
    """Unified parameter and par file management for multi-PTA pulsar data.

    This class consolidates all parameter management functionality:
    - Making par files consistent across PTAs
    - Building parameter mappings for MetaPulsar
    - Resolving parameter aliases and availability
    - Working with both PINT and Tempo2 PTAs
    """

    _EQUATORIAL_WARNING = (
        "Equatorial astrometry detected. PINT/tempo2 parity after T2CMETHOD "
        "modification is typically a few ns (about 6 ns on NG5 J1600), "
        "slightly larger than ecliptic pars with full convention alignment "
        "(about 1 ns on NG11 J1600). Reason: no explicit ECL obliquity "
        "convention to align; residual differences from ecliptic-frame "
        "geometry entering delay terms may remain. For best parity, prefer "
        "ecliptic coordinates in new timing solutions."
    )

    def __init__(
        self,
        file_data: Dict[str, Dict[str, Any]],  # pta_name -> file data
        combine_components: List[str] = [
            "astrometry",
            "spindown",
            "binary",
            "dispersion",
        ],
        add_dm_derivatives: bool = True,
        output_dir: Path = None,
        pulsar_name: str = None,
        exclude_from_consistent: List[str] | tuple[str, ...] = ("DM",),
        alignment_policy: Optional[AlignmentPolicy] = None,
    ):
        """Initialize with file data and configuration.

        Args:
            file_data: File data from FileDiscoveryService
            combine_components: List of components to make consistent
            add_dm_derivatives: Whether to add DM1, DM2 parameters
            output_dir: Directory for output files
            pulsar_name: Name of the pulsar (used for output filename generation)
            exclude_from_consistent: Canonical timing-model parameter names to keep
                PTA-specific even when their component is in combine_components.
                Defaults to ("DM",) so each PTA keeps its own reference DM while
                consistent dispersion still shares DM1/DM2. Pass an empty list to
                merge all parameters in selected components.
            alignment_policy: Policy for the multi-PTA common profile. ``None``
                means ``AlignmentPolicy()`` (strip unsupported families, take
                ``EPHEM``/clock/``NE_SW`` from the reference PTA).
        """
        self.file_data = file_data
        self.combine_components = combine_components
        self.add_dm_derivatives = add_dm_derivatives
        self.output_dir = output_dir
        self.pulsar_name = pulsar_name
        self.exclude_from_consistent = self._normalize_excluded_consistent_parameters(
            exclude_from_consistent
        )
        self.alignment_policy = alignment_policy or AlignmentPolicy()

        # Use first dictionary key as reference (consistent with MetaPulsarFactory)
        self.reference_pta = next(iter(file_data.keys()))

        self.logger = logger

        # Cache for PINT models
        self._pint_models_cache: Optional[Dict[str, TimingModel]] = None

    @property
    def ell1h_shapiro(self) -> Ell1hShapiroMode:
        """ELL1H orthometric Shapiro convention used for every PINT model build.

        ``"absorbed"`` on mixed PINT+Tempo2 stacks so temporary models here match
        the factory's materialization; ``"full"`` (PINT's default) otherwise.
        """
        return resolve_ell1h_shapiro_mode(
            self._get_timing_package(pta_name) for pta_name in self.file_data
        )

    def _create_model(self, parfile_data: Any) -> TimingModel:
        """Build a PINT model with this stack's ELL1H Shapiro convention."""
        return create_pint_model(parfile_data, ell1h_shapiro=self.ell1h_shapiro)

    def _create_model_from_dict(self, par: Dict[str, List[str]]) -> TimingModel:
        """Build a PINT model from a par dict, minus tempo2-only controls.

        ``IPM``/``FDDC``/``FDDI`` stay in the written par but mean nothing to
        PINT's ModelBuilder, which would only warn about them.
        """
        return self._create_model(
            {
                key: value
                for key, value in par.items()
                if key.upper() not in _TEMPO2_LOCAL_CONTROLS
            }
        )

    @property
    def pint_models(self) -> Dict[str, TimingModel]:
        """Get cached PINT models, creating them if needed.

        Returns:
            Dictionary mapping PTA names to PINT TimingModel instances
        """
        if self._pint_models_cache is None:
            self._pint_models_cache = {}
            for pta_name in self.file_data.keys():
                parfile_content = self._get_parfile_content(pta_name)
                self._pint_models_cache[pta_name] = self._create_model(parfile_content)
        return self._pint_models_cache

    def _clear_pint_models_cache(self):
        """Clear the PINT models cache."""
        self._pint_models_cache = None

    def _set_pint_models_from_dicts(
        self, parfile_dicts: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Rebuild the model cache from the current, transformed dictionaries.

        Component discovery must observe the post-strip / post-transform state,
        not the original ``file_data`` par content.
        """
        self._pint_models_cache = {
            pta_name: self._create_model_from_dict(par)
            for pta_name, par in parfile_dicts.items()
        }

    def _normalize_excluded_consistent_parameters(
        self, exclude_from_consistent: List[str] | tuple[str, ...]
    ) -> set[str]:
        """Return canonical PINT names excluded from consistent component merging."""
        return {
            resolve_parameter_alias(param).upper()
            for param in tuple(exclude_from_consistent)
        }

    def _is_excluded_from_consistent(self, param_name: str) -> bool:
        return (
            resolve_parameter_alias(param_name).upper() in self.exclude_from_consistent
        )

    # ===== MAIN PUBLIC METHODS =====

    def make_parfiles_consistent(self) -> Dict[str, Path]:
        """Make par files consistent across PTAs so that the certain model
        components (astrometry, spindown, binary, dispersion) are have
        consistent values between PTAs.

        Ordering matters and is fixed:

        1. parse with PINT;
        2. expand aggregate ``TEMPO1``;
        3. strip (or reject) unsupported deterministic families so later PINT
           model creation is safe;
        4. convert every TCB input to TDB;
        5. numerically transform ecliptic astrometry;
        6. apply the explicit convention profile (engine-gated);
        7. copy the selected reference components;
        8. apply the existing dispersion cleanup;
        9. serialize.

        Args:
            None

        Returns:
            Dictionary of consistent parfile contents for each PTA
        """
        self.logger.info("Making par files consistent across PTAs")

        # Clear cache at start of new consistency run
        self._clear_pint_models_cache()

        # 1. Parse par files into dictionaries
        parfile_dicts = self._parse_parfiles()

        # 2-3. Expand TEMPO1 and remove unsupported deterministic families
        self._prepare_common_surface(parfile_dicts)

        # 4. Convert units if needed (every TCB input, not only mixed collections)
        parfile_contents = {
            pta_name: dict_to_parfile_string(par, format="pint")
            for pta_name, par in parfile_dicts.items()
        }
        converted_parfiles = self._convert_units_if_needed(parfile_contents)

        # 5-8. Transform astrometry, align conventions, merge components, DM cleanup
        consistent_parfiles = self._make_parameters_consistent(converted_parfiles)

        # 9. Write consistent par files to output directory
        output_files = self._write_consistent_parfiles(consistent_parfiles)

        self.logger.info(
            f"Successfully created {len(output_files)} consistent par files"
        )
        return output_files

    def build_parameter_mappings(self) -> "ParameterMapping":
        """Build parameter mappings for MetaPulsar.

        The parameter mappings map the meta pulsar parameter names to the
        parameter names of the underlying PTA pulsars. The composite parameters
        will get an additional PTA name suffix to make them unique.

        """
        self.logger.info("Building parameter mappings for MetaPulsar")

        # 1. Discover parameters for components that should be merged
        mergeable_params = self._discover_mergeable_parameters()

        # 2. Process parameters from all PTAs
        fitparameters, setparameters = self._process_all_pta_parameters(
            mergeable_params
        )

        # 3. Validate consistency
        self._validate_parameter_consistency(fitparameters, setparameters)

        # 4. Build result
        return self._build_parameter_mapping_result(fitparameters, setparameters)

    # ===== COMMON-SURFACE PREPARATION =====

    def _prepare_common_surface(
        self, parfile_dicts: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Reject ambiguous models, then prepare the shared deterministic surface.

        Only multi-PTA combinations are rewritten: a single-PTA pulsar has no
        second engine or reference to align against, so its native deterministic
        model is preserved. Invalid orthometric parameter combinations are still
        rejected early for every ``consistent`` invocation.
        """
        for pta_name, par in parfile_dicts.items():
            self._reject_orthometric_conflict(pta_name, par)

        if len(parfile_dicts) < 2:
            self.logger.info("Common-surface preparation skipped (reason=single_pta)")
            return

        normalized_packages = self._normalized_timing_packages()
        has_pint = "pint" in normalized_packages
        has_tempo2 = "tempo2" in normalized_packages
        mixed = self._is_cross_engine_mix(normalized_packages)

        for pta_name, par in parfile_dicts.items():
            filled = expand_tempo1(par)
            if filled:
                self.logger.info(
                    f"PTA {pta_name}: expanded aggregate TEMPO1 into explicit "
                    f"{', '.join(filled)}"
                )
            self._strip_or_error_unsupported(
                pta_name,
                par,
                has_pint=has_pint,
                has_tempo2=has_tempo2,
                mixed=mixed,
            )

    def _reject_orthometric_conflict(
        self, pta_name: str, par: Dict[str, List[str]]
    ) -> None:
        """Reject mutually exclusive H4 and STIG orthometric parameterizations."""
        present = {key.upper() for key in par}
        stigma_names = present & {"STIG", "STIGMA", "VARSIGMA"}
        if "H4" not in present or not stigma_names:
            return

        stigma = sorted(stigma_names)[0]
        raise ValueError(
            f"PTA {pta_name}: invalid orthometric Shapiro model specifies both "
            f"H4 and {stigma}. Tempo2 would ignore {stigma} and use the "
            "approximate H4 model, while PINT rejects the combination. Remove "
            "one parameter explicitly; consistent alignment will not choose a "
            "physical model on the user's behalf."
        )

    def _collect_unsupported_keys(
        self,
        par: Dict[str, List[str]],
        *,
        has_pint: bool,
        has_tempo2: bool,
        mixed: bool,
    ) -> Set[str]:
        """Return the par keys outside the common deterministic surface."""
        keys: Set[str] = set()
        for key in par:
            upper = key.upper()
            if has_pint and _is_pint_unsafe(upper):
                keys.add(key)
            elif has_tempo2 and _is_tempo2_unsafe(upper):
                keys.add(key)
            elif mixed and _is_mixed_unsafe(upper):
                keys.add(key)
        keys |= self._unsupported_solar_geometry_keys(
            par, has_pint=has_pint, has_tempo2=has_tempo2
        )
        return keys

    def _unsupported_solar_geometry_keys(
        self, par: Dict[str, List[str]], *, has_pint: bool, has_tempo2: bool
    ) -> Set[str]:
        """Return value-dependent solar-wind geometry violations.

        ``IPM 0`` disables Tempo2's interplanetary medium entirely (PINT has no
        equivalent switch) and ``SWM 1`` selects PINT's non-constant solar-wind
        model (Tempo2 has no equivalent). Both are normalized on output rather
        than merely dropped: see ``_apply_mixed_engine_switches``.
        """
        violations: Set[str] = set()
        for key, value in par.items():
            upper = key.upper()
            if has_pint and upper == "IPM" and _first_token(value) == "0":
                violations.add(key)
            if has_tempo2 and upper == "SWM" and _first_token(value) == "1":
                violations.add(key)
        return violations

    def _strip_or_error_unsupported(
        self,
        pta_name: str,
        par: Dict[str, List[str]],
        *,
        has_pint: bool,
        has_tempo2: bool,
        mixed: bool,
    ) -> None:
        """Apply the policy for every unsupported family found in one par."""
        keys = self._collect_unsupported_keys(
            par, has_pint=has_pint, has_tempo2=has_tempo2, mixed=mixed
        )
        constraints = dmmodel_constraint_values(par) if has_pint else []
        if not keys and not constraints:
            return

        details = sorted(keys) + [f"CONSTRAIN {value}" for value in constraints]
        if self.alignment_policy.unsupported == "error":
            raise ValueError(
                f"PTA {pta_name}: unsupported timing parameters for the consistent "
                f"common profile: {', '.join(details)}"
            )

        for key in keys:
            par.pop(key, None)
        if constraints:
            strip_dmmodel_constraints(par)
        self.logger.warning(
            f"PTA {pta_name}: stripped unsupported deterministic parameters: "
            f"{', '.join(details)}"
        )

    # ===== PARFILE CONSISTENCY METHODS =====

    def _convert_units_if_needed(
        self, parfile_contents: Dict[str, str]
    ) -> Dict[str, str]:
        """Normalize timescales only where combination requires it.

        A mixed PINT/Tempo2 stack is always materialized as explicit TDB because
        PINT otherwise converts TCB models internally. Single-engine stacks keep
        a homogeneous native timescale; genuinely mixed TCB/TDB collections are
        normalized to TDB. A single PTA is never rewritten.
        """
        self.logger.info("Checking if unit conversion is needed")

        if len(parfile_contents) < 2:
            self.logger.info("Unit normalization skipped (reason=single_pta)")
            return parfile_contents

        file_units = self._determine_parfile_units(parfile_contents)

        unique_units = set(file_units.values())
        if unique_units == {"TDB"}:
            self.logger.info("All par files have TDB units. No conversion needed.")
            return parfile_contents

        mixed_engines = self._is_cross_engine_mix(self._normalized_timing_packages())
        if len(unique_units) == 1 and not mixed_engines:
            self.logger.info(
                "Unit normalization skipped "
                f"(reason=homogeneous_single_engine, units={next(iter(unique_units))})"
            )
            return parfile_contents

        self.logger.info(
            f"Non-TDB units detected: {sorted(unique_units)}. "
            "Converting TCB inputs to TDB."
        )
        return self._convert_tcb_inputs(file_units, parfile_contents)

    def _determine_parfile_units(
        self, parfile_contents: Dict[str, str]
    ) -> Dict[str, str]:
        """Determine the units of all par files for this pulsar."""
        self.logger.info("Determining units for all par files")

        file_units = {}

        for pta_name, parfile_content in parfile_contents.items():
            try:
                # Parse to check current units
                parfile_dict = parse_parfile(StringIO(parfile_content))
                units_value = parfile_dict.get(
                    "UNITS", [self._get_default_time_units(pta_name)]
                )
                current_units, _ = parse_parameter_using_pint("UNITS", units_value)
                file_units[pta_name] = str(current_units).upper()

            except Exception as e:
                self.logger.error(f"Error reading par file for PTA {pta_name}: {e}")
                raise RuntimeError(f"Failed to read par file for PTA {pta_name}") from e

        return file_units

    def _get_default_time_units(self, pta_name: str) -> str:
        """Get the default time units for a PTA based on its timing package.

        Args:
            pta_name: Name of the PTA

        Returns:
            Default time units: "TDB" for PINT, "TCB" for Tempo2
        """
        timing_package = self.file_data[pta_name].get("timing_package", "pint")
        return "TDB" if timing_package == "pint" else "TCB"

    def _convert_tcb_inputs(
        self, file_units: Dict[str, str], parfile_contents: Dict[str, str]
    ) -> Dict[str, str]:
        """Convert every non-TDB par file to TDB using its own timing package.

        The collection need not be mixed; an all-TCB collection is converted too.
        """
        converted_parfiles = {}

        for pta_name, parfile_content in parfile_contents.items():
            current_units = file_units[pta_name]

            if current_units == "TDB":
                # Already in TDB, no conversion needed
                converted_parfiles[pta_name] = parfile_content
            else:
                # Get timing package for this PTA
                timing_package = self._get_timing_package(pta_name)

                if timing_package == "pint":
                    # Use PINT conversion for PINT PTAs
                    try:
                        converted_content = self._convert_pint_to_tdb(parfile_content)
                        converted_parfiles[pta_name] = converted_content
                        self.logger.debug(f"Converted PTA {pta_name} using PINT")
                    except Exception as e:
                        self.logger.error(
                            f"PINT conversion failed for PTA {pta_name}: {e}"
                        )
                        raise RuntimeError(
                            f"PINT unit conversion failed for PTA {pta_name}"
                        ) from e
                else:
                    # Use Tempo2 conversion for Tempo2 PTAs, or fallback
                    try:
                        converted_content = self._convert_tempo2_to_tdb(parfile_content)
                        converted_parfiles[pta_name] = converted_content
                        self.logger.debug(f"Converted PTA {pta_name} using Tempo2")
                    except Exception as e:
                        self.logger.error(
                            f"Tempo2 conversion failed for PTA {pta_name}: {e}"
                        )
                        raise RuntimeError(
                            f"Tempo2 unit conversion failed for PTA {pta_name}"
                        ) from e

                self._assert_explicit_tdb(pta_name, converted_parfiles[pta_name])

        return converted_parfiles

    def _assert_explicit_tdb(self, pta_name: str, parfile_content: str) -> None:
        """Re-parse converted content and require an explicit ``UNITS TDB``."""
        parsed = parse_parfile(StringIO(parfile_content))
        units_value = parsed.get("UNITS")
        current_units = (
            str(parse_parameter_using_pint("UNITS", units_value)[0]).upper()
            if units_value
            else None
        )
        if current_units != "TDB":
            raise RuntimeError(
                f"Unit conversion for PTA {pta_name} did not produce an explicit "
                f"UNITS TDB line (found {current_units!r})"
            )

    def _convert_pint_to_tdb(self, parfile_content: str) -> str:
        """Convert par file from TCB to TDB using PINT ModelBuilder."""
        try:
            # Create PINT model and parse par file
            model = self._create_model(parfile_content)

            # Write par file with TDB units
            new_file = StringIO()
            model.write_parfile(new_file)

            return new_file.getvalue()
        except Exception as e:
            raise RuntimeError(f"PINT conversion failed: {e}") from e

    def _align_parameter(
        self,
        parfile_dict: Dict[str, List[str]],
        reference_dict: Dict[str, List[str]],
        aliases,
        required: bool = False,
    ) -> bool:
        alias_list = [aliases] if isinstance(aliases, str) else list(aliases)

        # Grab the reference value from whichever alias the reference PTA used
        ref_key = None
        for alias in alias_list:
            if alias in reference_dict:
                ref_key = alias
                break
        if ref_key is None:
            msg = f"No alias from {alias_list} found in reference PTA {self.reference_pta}"
            if required:
                raise ValueError(msg)
            self.logger.error(msg)
            return False
        ref_value = reference_dict[ref_key]

        # Keep only the first alias that already exists in this PTA; drop the rest
        target_key = None
        for alias in alias_list:
            if alias in parfile_dict:
                if target_key is None:
                    target_key = alias
                else:
                    parfile_dict.pop(alias)
                    self.logger.error(
                        f"Dropping duplicate {alias} found in PTA (not {self.reference_pta})"
                    )

        if target_key is None:
            # Inject the reference key/value if PTA lacked any alias
            target_key = ref_key

        # Update the value in-place using the existing key
        parfile_dict[target_key] = ref_value
        return True

    def _is_t2cmethod_tempo(self, value: List[str]) -> bool:
        if not value:
            return False
        first = value[0].split()[0].upper() if value[0].split() else ""
        return first == "TEMPO"

    def _normalize_timing_package(self, package: str) -> str:
        return normalize_timing_package(package)

    def _normalized_timing_packages(self) -> Set[str]:
        return {
            self._normalize_timing_package(self._get_timing_package(pta_name))
            for pta_name in self.file_data
        }

    def _is_cross_engine_mix(self, normalized_packages: Set[str]) -> bool:
        return {"pint", "tempo2"}.issubset(normalized_packages)

    def _parse_ecl_value(self, parfile_dict: Dict[str, List[str]]) -> Optional[str]:
        raw_value = parfile_dict.get("ECL")
        if not raw_value:
            return None
        parts = raw_value[0].split() if raw_value[0] else []
        if not parts:
            return None
        return parts[0].upper()

    def _parse_t2cmethod_value(
        self, parfile_dict: Dict[str, List[str]]
    ) -> Optional[str]:
        raw_value = parfile_dict.get("T2CMETHOD")
        if not raw_value:
            return None
        parts = raw_value[0].split() if raw_value[0] else []
        if not parts:
            return None
        return parts[0].upper()

    def _parse_ne_sw_value(self, parfile_dict: Dict[str, List[str]]) -> Optional[float]:
        """Return explicit NE_SW (cm^-3) from a parfile dict, or None if absent.

        Alias-aware: NANOGrav-style par files spell it SOLARN0 (PINT aliases:
        NE1AU, SOLARN0), which must count as an explicit value.
        """
        for alias in get_aliases_for_parameter("NE_SW"):
            raw = parfile_dict.get(alias)
            if raw:
                value, _frozen = parse_parameter_using_pint(alias, raw)
                return float(value)
        return None

    def _resolve_consistent_ne_sw(
        self,
        reference_dict: Dict[str, List[str]],
        normalized_packages: Set[str],
    ) -> Optional[float]:
        """Resolve the consistent NE_SW density (cm^-3) for this pulsar stack.

        Precedence: ``AlignmentPolicy.ne_sw``, then the reference PTA's explicit
        ``NE_SW``/``NE1AU``/``SOLARN0``, then tempo2's implicit 4 cm^-3 when
        tempo2 is in the stack.
        """
        if self.alignment_policy.ne_sw is not None:
            return float(self.alignment_policy.ne_sw)
        explicit = self._parse_ne_sw_value(reference_dict)
        if explicit is not None:
            return explicit
        if "tempo2" in normalized_packages:
            return 4.0
        return None

    def _align_ne_sw_convention(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
    ) -> None:
        """Align explicit NE_SW across PTAs to resolve tempo2/PINT default mismatch."""
        normalized = self._normalized_timing_packages()
        consistent_ne_sw = self._resolve_consistent_ne_sw(reference_dict, normalized)
        if consistent_ne_sw is None:
            self.logger.info("NE_SW alignment skipped (reason=pint_only_implicit_zero)")
            return

        line = [f"{consistent_ne_sw:g} 0"]
        for pta_name, parfile_dict in parfile_dicts.items():
            old = self._parse_ne_sw_value(parfile_dict)
            # Drop every alias spelling before writing the canonical line, so a
            # SOLARN0 line can never coexist with the injected NE_SW (PINT maps
            # both to NE_SW and rejects the pair as a repeated parameter).
            for alias in get_aliases_for_parameter("NE_SW"):
                if alias != "NE_SW" and alias in parfile_dict:
                    parfile_dict.pop(alias)
                    self.logger.info(
                        f"PTA {pta_name}: replaced {alias} with canonical NE_SW"
                    )
            if old is not None and abs(old - consistent_ne_sw) > 1e-9:
                self.logger.warning(
                    f"PTA {pta_name}: overwriting NE_SW {old:g} with consistent "
                    f"value {consistent_ne_sw:g} from reference/resolution policy"
                )
            elif old is None:
                self.logger.info(
                    f"PTA {pta_name}: NE_SW aligned to {consistent_ne_sw:g} (was absent)"
                )
            parfile_dict["NE_SW"] = line

    def _collect_convention_states(
        self, parfile_dicts: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, Optional[str]]]:
        states: Dict[str, Dict[str, Optional[str]]] = {}
        for pta_name, parfile_dict in parfile_dicts.items():
            states[pta_name] = {
                "style": detect_astrometry_style(parfile_dict),
                "ecl": self._parse_ecl_value(parfile_dict),
                "t2cmethod": self._parse_t2cmethod_value(parfile_dict),
                "package": self._normalize_timing_package(
                    self._get_timing_package(pta_name)
                ),
            }
        return states

    def _resolve_reference_conventions(
        self, reference_dict: Dict[str, List[str]]
    ) -> Tuple[str, str]:
        """Resolve the EPHEM and clock realization every PTA must adopt.

        Order of precedence: ``AlignmentPolicy`` value, then the reference PTA's
        value. A bare ``TT(BIPM)`` is ambiguous across environments and must be
        pinned with ``AlignmentPolicy.bipm_version``.
        """
        policy = self.alignment_policy

        ephem = policy.ephem or _first_token(reference_dict.get("EPHEM"))
        if ephem is None:
            raise ValueError(
                "Consistent alignment requires EPHEM: no alias from ['EPHEM'] "
                f"found in reference PTA {self.reference_pta} and "
                "AlignmentPolicy.ephem is not set"
            )

        clock = (
            policy.clock
            or _first_token(reference_dict.get("CLOCK"))
            or _first_token(reference_dict.get("CLK"))
        )
        if clock is None:
            raise ValueError(
                "Consistent alignment requires a clock realization: no alias "
                "from ['CLOCK', 'CLK'] found in reference PTA "
                f"{self.reference_pta} and AlignmentPolicy.clock is not set"
            )

        return ephem, self._resolve_bipm_clock(clock)

    def _resolve_bipm_clock(self, clock: str) -> str:
        """Require a dated BIPM realization, resolving a bare ``TT(BIPM)``."""
        match = _BIPM_CLOCK_RE.match(clock.strip())
        if match is None:
            return clock

        version = self.alignment_policy.bipm_version
        year = match.group("year")
        if year is None:
            if version is None:
                raise ValueError(
                    f"Bare TT(BIPM) is ambiguous; set AlignmentPolicy.bipm_version "
                    f"(reference PTA {self.reference_pta})"
                )
            return f"TT(BIPM{version})"

        if version is not None and int(year) != int(version):
            raise ValueError(
                f"Clock realization {clock!r} disagrees with "
                f"AlignmentPolicy.bipm_version={version}"
            )
        return clock

    def _set_aliased_value(
        self,
        parfile_dict: Dict[str, List[str]],
        aliases: List[str],
        value: str,
    ) -> None:
        """Write ``value`` under whichever alias this PTA already uses."""
        target_key = None
        for alias in aliases:
            if alias in parfile_dict:
                if target_key is None:
                    target_key = alias
                else:
                    parfile_dict.pop(alias)
                    self.logger.error(
                        f"Dropping duplicate {alias} found in PTA (not {self.reference_pta})"
                    )
        if target_key is None:
            target_key = aliases[0]
        parfile_dict[target_key] = [value]

    def _set_engine_clock_value(
        self,
        pta_name: str,
        parfile_dict: Dict[str, List[str]],
        value: str,
    ) -> None:
        """Write the resolved clock with the target engine's native keyword."""
        package = self._normalize_timing_package(self._get_timing_package(pta_name))
        target_key = "CLK" if package == "tempo2" else "CLOCK"
        for alias in ("CLOCK", "CLK"):
            parfile_dict.pop(alias, None)
        parfile_dict[target_key] = [value]

    def _align_reference_conventions(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        ephem: str,
        clock: str,
    ) -> None:
        """Apply the resolved EPHEM and clock convention to every PTA."""
        for pta_name, parfile_dict in parfile_dicts.items():
            self._set_aliased_value(parfile_dict, ["EPHEM"], ephem)
            self._set_engine_clock_value(pta_name, parfile_dict, clock)

    def _apply_cross_engine_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        convention_states: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """Apply convention rules needed when PINT and tempo2 pulsars are mixed.

        Ecliptic frames are handled numerically by
        ``_transform_ecliptic_for_all``; what remains here is the equatorial
        case, which has no active obliquity convention to align.
        """
        for pta_name, parfile_dict in parfile_dicts.items():
            style = convention_states[pta_name]["style"]
            if style == "ecliptic":
                continue

            actions: List[str] = []
            if "ECL" in parfile_dict:
                old_ecl = parfile_dict.pop("ECL")
                actions.append(f"removed ECL ({old_ecl[0]})")
            self.logger.warning(f"[{pta_name}] {self._EQUATORIAL_WARNING}")
            actions.append("emitted equatorial warning")

            self.logger.info(
                f"PTA {pta_name}: consistent convention rules applied "
                f"(reason=cross_engine_parity, style={style}); "
                f"actions: {', '.join(actions)}"
            )

    def _apply_tempo2_only_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
        convention_states: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """Align heterogeneous T2CMETHOD conventions for tempo2-only stacks."""
        t2_values = {state["t2cmethod"] for state in convention_states.values()}
        if len(t2_values) > 1:
            if "T2CMETHOD" not in reference_dict:
                self.logger.info(
                    "T2CMETHOD alignment skipped (reason=reference_missing_t2cmethod)"
                )
            else:
                for pta_name, parfile_dict in parfile_dicts.items():
                    self._align_parameter(
                        parfile_dict, reference_dict, "T2CMETHOD", required=False
                    )
                    self.logger.info(
                        f"PTA {pta_name}: consistent convention rules applied "
                        "(reason=tempo2_only_t2cmethod_heterogeneous); "
                        f"set T2CMETHOD={parfile_dict['T2CMETHOD'][0]}"
                    )
        else:
            self.logger.info(
                "T2CMETHOD alignment skipped "
                "(reason=homogeneous_t2cmethod_single_engine_tempo2)"
            )

    def _apply_consistent_convention_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
    ) -> None:
        """Apply gated convention rules across consistent parfile dictionaries."""
        ephem, clock = self._resolve_reference_conventions(reference_dict)
        convention_states = self._collect_convention_states(parfile_dicts)

        if len(parfile_dicts) == 1:
            self.logger.info("Consistent convention rules skipped (reason=single_pta)")
            return

        # Multi-PTA combinations share the reference ephemeris and clock.
        self._align_reference_conventions(parfile_dicts, ephem, clock)

        normalized_packages = self._normalized_timing_packages()

        if self._is_cross_engine_mix(normalized_packages):
            self._apply_cross_engine_rules(parfile_dicts, convention_states)
        elif normalized_packages == {"pint"}:
            self.logger.info(
                "Engine-specific convention rules skipped (reason=single_engine_pint)"
            )
        elif normalized_packages == {"tempo2"}:
            self._apply_tempo2_only_rules(
                parfile_dicts, reference_dict, convention_states
            )
        else:
            self.logger.info(
                "Engine-specific convention rules skipped "
                f"(reason=unsupported_packages:{sorted(normalized_packages)})"
            )

        self._apply_explicit_conventions(parfile_dicts)

    # ===== NUMERIC ECLIPTIC ALIGNMENT =====

    def _resolve_target_ecl(
        self,
        reference_dict: Dict[str, List[str]],
        convention_states: Dict[str, Dict[str, Optional[str]]],
    ) -> Optional[str]:
        """Return the obliquity convention every ecliptic PTA must adopt.

        ``None`` means no transformation is needed: single-engine stacks that
        already share one convention are left alone.
        """
        normalized_packages = self._normalized_timing_packages()
        ecl_values = {
            state["ecl"]
            for state in convention_states.values()
            if state["style"] == "ecliptic"
        }
        if not ecl_values:
            return None

        if self._is_cross_engine_mix(normalized_packages):
            # PINT and tempo2 agree best for ecliptic pars under IERS2003.
            return "IERS2003"
        if len(ecl_values) <= 1:
            return None
        if normalized_packages == {"pint"}:
            return self._parse_ecl_value(reference_dict) or "IERS2010"
        if normalized_packages == {"tempo2"}:
            return "IERS2003"
        return None

    def _transform_ecliptic_for_all(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
    ) -> None:
        """Numerically move every ecliptic PTA onto one obliquity convention.

        This is a coordinate transformation, never a relabel: the position and
        proper motion are rotated so the physical direction is preserved.
        """
        if len(parfile_dicts) == 1:
            self.logger.info("Ecliptic alignment skipped (reason=single_pta)")
            return

        convention_states = self._collect_convention_states(parfile_dicts)
        target_ecl = self._resolve_target_ecl(reference_dict, convention_states)
        if target_ecl is None:
            self.logger.info(
                "Ecliptic alignment skipped (reason=no_transform_required)"
            )
            return

        for pta_name, state in convention_states.items():
            if state["style"] != "ecliptic":
                continue
            self._transform_ecliptic_astrometry(
                pta_name, parfile_dicts[pta_name], target_ecl
            )

    def _transform_ecliptic_astrometry(
        self,
        pta_name: str,
        par: Dict[str, List[str]],
        target_ecl: str,
    ) -> None:
        """Rotate one PTA's ecliptic astrometry onto ``target_ecl`` using PINT."""
        source_ecl = self._parse_ecl_value(par)

        try:
            model = self._create_model_from_dict(par)
            epoch = model.POSEPOCH.quantity if "POSEPOCH" in model.params else None
            converted = model.as_ICRS(epoch=epoch).as_ECL(epoch=epoch, ecl=target_ecl)
            written = StringIO()
            converted.write_parfile(written)
            transformed = parse_parfile(StringIO(written.getvalue()))
        except Exception as e:
            raise ValueError(
                f"PTA {pta_name}: ecliptic astrometry conversion "
                f"{source_ecl or '(missing)'} -> {target_ecl} failed: {e}"
            ) from e

        for key in list(par):
            if key.upper() in _ECL_INPUT_ALIASES:
                par.pop(key)
        for key in _ECL_COPY_KEYS:
            value = self._first_alias_value(transformed, key)
            if value is None:
                par.pop(key, None)
            else:
                par[key] = value

        self.logger.info(
            f"PTA {pta_name}: transformed ecliptic astrometry "
            f"{source_ecl or '(missing)'} -> {target_ecl} "
            "(numeric coordinate conversion, not a relabel)"
        )

    @staticmethod
    def _first_alias_value(
        parfile_dict: Dict[str, List[str]], canonical_param: str
    ) -> Optional[List[str]]:
        """Return the value stored under any PINT alias of ``canonical_param``."""
        for alias in get_aliases_for_parameter(canonical_param):
            if alias in parfile_dict:
                return parfile_dict[alias]
        return None

    # ===== EXPLICIT CONVENTION PROFILE =====

    def _apply_explicit_conventions(
        self, parfile_dicts: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Write the explicit common profile onto every PTA dictionary."""
        mixed = self._is_cross_engine_mix(self._normalized_timing_packages())

        for pta_name, par in parfile_dicts.items():
            if mixed:
                par["UNITS"] = ["TDB"]
                self._apply_mixed_engine_switches(pta_name, par)

    def _apply_mixed_engine_switches(
        self, pta_name: str, par: Dict[str, List[str]]
    ) -> None:
        """Force the validated PINT+Tempo2 residual-parity switches.

        Applied only to mixed stacks: a PINT-only combination must not have its
        troposphere or planetary-Shapiro settings flipped to ``N``.
        """
        par["T2CMETHOD"] = ["IAU2000B"]
        par["TIMEEPH"] = ["FB90"]
        _set_boolean(par, "DILATEFREQ", False)
        _set_boolean(par, "CORRECT_TROPOSPHERE", False)
        _set_boolean(par, "PLANET_SHAPIRO", False)
        # Whole-Solar-System Shapiro stays enabled; it is independent of the
        # planetary term above.
        removed_no_ss = par.pop("NO_SS_SHAPIRO", None)
        par["SWM"] = ["0"]
        if self._normalize_timing_package(self._get_timing_package(pta_name)) == (
            "tempo2"
        ):
            par["IPM"] = ["1"]
        else:
            # IPM is a tempo2 control; PINT output carries no such switch.
            par.pop("IPM", None)

        if removed_no_ss is not None:
            self.logger.info(
                f"PTA {pta_name}: removed NO_SS_SHAPIRO ({removed_no_ss[0]}); "
                "whole-Solar-System Shapiro enabled"
            )

    @staticmethod
    def _declared_nharms(par: Dict[str, List[str]]) -> Optional[int]:
        """Return the harmonic count declared under either spelling, if any."""
        present = {key.upper(): key for key in par}
        for name in ("NHARMS", "NHARM"):
            key = present.get(name)
            if key and par.get(key):
                try:
                    return int(str(par[key][0]).split()[0])
                except (TypeError, ValueError, IndexError):
                    continue
        return None

    def _align_ell1h_nharms_for_all(
        self, parfile_dicts: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Give every PTA the same ELL1H harmonic count.

        ``NHARM`` is unknown to PINT and therefore never copied by the binary
        component merge, so the count has to be resolved once for the stack:
        the largest value any input declared, floored at 7. A finer truncation
        is always safe for both engines.
        """
        declared = [
            value
            for value in (self._declared_nharms(par) for par in parfile_dicts.values())
            if value is not None
        ]
        nharms = max([*declared, 7])
        for par in parfile_dicts.values():
            self._align_ell1h_nharms(par, nharms)

    def _align_ell1h_nharms(
        self, par: Dict[str, List[str]], nharms: Optional[int] = None
    ) -> None:
        """Ensure NHARM (tempo2) and NHARMS (PINT) >= 7 for H3+H4 ELL1H/T2 pars.

        PINT floors ``NHARMS`` at 7 whenever ``H4`` is set, while tempo2 defaults
        to ``nharm=4`` when the keyword is absent. The two spellings are not
        aliases of each other, so a shared par has to carry both.

        Not applied for the ``H3``+``STIG`` orthometric path: both engines ignore
        the harmonic count there (see ``ell1h_shapiro``). A count left over from
        the input is dropped in that case, because component merging can replace
        an ``H3``+``H4`` model with an ``H3``+``STIG`` one.

        Must run *after* the component merge, which is what decides the final
        binary model of each written par.
        """
        present = {key.upper(): key for key in par}
        has_stig = "STIG" in present or "STIGMA" in present
        if has_stig:
            for name in ("NHARM", "NHARMS"):
                if name in present:
                    par.pop(present[name])
            return
        if "H4" not in present:
            return
        if "H3" not in present:
            return

        if nharms is None:
            declared = self._declared_nharms(par)
            nharms = 7 if declared is None else max(declared, 7)

        # Dual emit: tempo2 reads NHARM; PINT reads NHARMS (no cross-alias).
        par[present.get("NHARM", "NHARM")] = [str(nharms)]
        par[present.get("NHARMS", "NHARMS")] = [str(nharms)]

    def _make_parameters_consistent(
        self, parfile_data: Dict[str, str]
    ) -> Dict[str, str]:
        """Make parameters consistent using reference PTA values.

        This function really is the workhorse of the MetaPulsar procedure to
        make par models consistent across PTAs. Method:

        - Start with parfiles that have been unit-converted (done)
        - Numerically transform ecliptic astrometry onto one obliquity
          convention (a coordinate rotation, never a relabel)
        - Align explicit NE_SW when required for tempo2/PINT parity
        - Always align CLOCK and EPHEM parameters, and write the explicit
          convention profile (engine-gated; see ``_apply_explicit_conventions``)
        - Rebuild the PINT model cache from these transformed dictionaries
        - Determine which model 'components' (astrometry, spindown, etc.) are
          being made consistent, and find all parameters in the models
        - For each component, replace the parameters with the values of the
          reference PTA
        - For dispersion, remove DMX parameters
        - Optionally, add DM1 and DM2 parameters
        - Convert back to par file strings
        - Write consistent par files to output directory

        This method is deterministic, so we do not have to save the new parfiles
        (but we can, as an option)

        Args:
            parfile_data: Dictionary of parfile contents for each PTA

        Returns:
            Dictionary of consistent parfile contents for each PTA
        """
        self.logger.info(
            f"Making parameters consistent using reference PTA: {self.reference_pta}"
        )

        # Parse all par files
        parfile_dicts = {}
        for pta_name, parfile_content in parfile_data.items():
            try:
                parfile_dict = parse_parfile(StringIO(parfile_content))
                parfile_dicts[pta_name] = parfile_dict
            except Exception as e:
                self.logger.error(f"Error parsing par file for PTA {pta_name}: {e}")
                raise RuntimeError(
                    f"Failed to parse par file for PTA {pta_name}"
                ) from e

        # Get reference PTA parameters (mutated in place by the steps below)
        reference_dict = parfile_dicts[self.reference_pta]

        # Numerically transform ecliptic astrometry onto one obliquity convention.
        try:
            self._transform_ecliptic_for_all(parfile_dicts, reference_dict)
        except ValueError as e:
            self.logger.error(f"Ecliptic astrometry alignment failed: {e}")
            raise RuntimeError(f"Ecliptic astrometry alignment failed: {e}") from e

        # Align the solar-wind amplitude and the explicit convention profile.
        self._align_ne_sw_convention(parfile_dicts, reference_dict)
        try:
            self._apply_consistent_convention_rules(parfile_dicts, reference_dict)
        except ValueError as e:
            self.logger.error(f"Consistent convention rules failed: {e}")
            raise RuntimeError(f"Consistent convention rules failed: {e}") from e

        # Component discovery must observe the transformed dictionaries, not the
        # original file_data cache.
        self._set_pint_models_from_dicts(parfile_dicts)

        # Pre-compute component parameters for ALL components
        component_params_map = {}
        pint_models = self.pint_models  # Use cached models
        for component in self.combine_components:
            component_params_map[component] = get_parameters_by_type_from_models(
                component, pint_models
            )

        # Pre-compute DMX parameters for ALL PTAs
        dmx_params_map = {}
        for pta_name, parfile_dict in parfile_dicts.items():
            dmx_params_map[pta_name] = self._get_dmx_parameters_from_parfile(
                parfile_dict
            )

        # Process each component
        for component in self.combine_components:
            self.logger.info(f"Making {component} parameters consistent")

            # Always call standard component consistency logic first
            self._make_component_parameters_consistent(
                parfile_dicts,
                reference_dict,
                self.reference_pta,
                component,
                component_params_map[component],
            )

            # For dispersion, also apply special DM logic
            if component == "dispersion":
                self._handle_dm_special_cases(
                    parfile_dicts,
                    reference_dict,
                    self.add_dm_derivatives,
                    dmx_params_map,
                )

        # The binary component merge decides each par's final orthometric model,
        # so the harmonic count is normalized last.
        self._align_ell1h_nharms_for_all(parfile_dicts)

        # Convert back to par file strings
        consistent_parfiles = {}
        for pta_name, parfile_dict in parfile_dicts.items():
            try:
                consistent_content = dict_to_parfile_string(parfile_dict, format="pint")
                consistent_parfiles[pta_name] = consistent_content
                self.logger.debug(f"Converted PTA {pta_name} par file back to string")
            except Exception as e:
                self.logger.error(f"Error converting par file for PTA {pta_name}: {e}")
                raise RuntimeError(
                    f"Failed to convert par file for PTA {pta_name}"
                ) from e

        return consistent_parfiles

    def _make_component_parameters_consistent(
        self,
        parfile_dicts: Dict[str, Dict],
        reference_dict: Dict,
        reference_pta: str,
        component: str,
        component_params: List[str],
    ) -> None:
        """Make parameters for a specific component consistent."""
        consistent_params = [
            param
            for param in component_params
            if not self._is_excluded_from_consistent(param)
        ]

        if not consistent_params:
            self.logger.debug(
                f"No non-excluded parameters found for component {component}, skipping"
            )
            return

        reference_values = {}
        for param in consistent_params:
            if param in reference_dict:
                reference_values[param] = reference_dict[param]

        for pta_name, parfile_dict in parfile_dicts.items():
            if pta_name == reference_pta:
                continue

            for param in consistent_params:
                if param in parfile_dict:
                    parfile_dict.pop(param)

            for param, value in reference_values.items():
                parfile_dict[param] = value

    def _handle_dm_special_cases(
        self,
        parfile_dicts: Dict[str, Dict],
        reference_dict: Dict,
        add_dm_derivatives: bool,
        dmx_params_map: Dict[str, List[str]],
    ) -> None:
        """Handle DM-specific special cases: DMX removal, DMEPOCH, DM1/DM2 derivatives."""

        dmepoch_value = reference_dict.get("DMEPOCH", ["55000"])
        reference_dmepoch, _ = parse_parameter_using_pint("DMEPOCH", dmepoch_value)
        self.logger.debug(f"Reference DMEPOCH: {reference_dmepoch}")

        for pta_name, parfile_dict in parfile_dicts.items():
            dmx_params = dmx_params_map[pta_name]
            for dmx_param in dmx_params:
                old_value = parfile_dict[dmx_param]
                parfile_dict.pop(dmx_param)
                self.logger.debug(f"PTA {pta_name}: Removed {dmx_param} = {old_value}")

            dm_value = parfile_dict.get("DM")
            if dm_value is None:
                raise ValueError(
                    f"DM parameter is missing from parfile for PTA {pta_name}. "
                    "DM is required for consistent dispersion cleanup because it "
                    "is kept as that PTA's local reference DM."
                )

            local_dm, dm_is_frozen = parse_parameter_using_pint("DM", dm_value)
            if dm_is_frozen:
                self.logger.warning(
                    f"DM parameter in parfile for PTA {pta_name} is not free. "
                    "Setting to free."
                )

            parfile_dict["DM"] = [f"{local_dm} 1"]
            self.logger.debug(f"PTA {pta_name}: Preserved local DM = {local_dm} (free)")

            parfile_dict["DMEPOCH"] = [f"{reference_dmepoch} 0"]
            self.logger.debug(
                f"PTA {pta_name}: Set DMEPOCH = {reference_dmepoch} (frozen)"
            )

            if add_dm_derivatives:
                parfile_dict["DM1"] = ["0.0 1"]
                parfile_dict["DM2"] = ["0.0 1"]
                self.logger.info(f"PTA {pta_name}: Set DM1 = 0.0, DM2 = 0.0")

    def _get_dmx_parameters_from_parfile(self, parfile_dict: Dict) -> List[str]:
        """Get DMX parameters from a parfile using PINT component discovery."""
        # Create PINT model directly from dictionary
        model = self._create_model_from_dict(parfile_dict)

        # Find DMX parameters from dispersion_dmx component
        dmx_params = []
        for comp in model.components.values():
            if hasattr(comp, "category") and comp.category == "dispersion_dmx":
                if hasattr(comp, "params"):
                    dmx_params.extend(comp.params)

        return dmx_params

    def _write_consistent_parfiles(
        self, consistent_parfiles: Dict[str, str]
    ) -> Dict[str, Path]:
        """Write consistent par files to output directory."""
        if self.output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="consistent_parfiles_"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_files = {}

        for pta_name, parfile_content in consistent_parfiles.items():
            output_filename = self._get_output_filename(pta_name)
            output_path = self.output_dir / output_filename

            with open(output_path, "w") as f:
                f.write(parfile_content)

            output_files[pta_name] = output_path
            self.logger.debug(f"Written consistent par file: {output_path}")

        return output_files

    def _get_output_filename(self, pta_name: str) -> str:
        """Generate output filename for consistent par file."""
        if self.pulsar_name:
            return f"{self.pulsar_name}_consistent_{pta_name}.par"
        else:
            return f"consistent_{pta_name}.par"

    # ===== PARAMETER MAPPING METHODS =====

    def _discover_mergeable_parameters(self) -> List[str]:
        """Discover parameters that can be merged based on component types."""
        mergeable_params = []
        for component_type in self.combine_components:
            # Convert file data to parfile_dicts for pint_helpers
            parfile_dicts = {}
            for pta_name in self.file_data.keys():
                parfile_content = self._get_parfile_content(pta_name)
                parfile_dicts[pta_name] = parse_parfile(StringIO(parfile_content))

            pint_models = self.pint_models  # Use cached models
            params = get_parameters_by_type_from_models(component_type, pint_models)
            mergeable_params.extend(
                param
                for param in params
                if not self._is_excluded_from_consistent(param)
            )
        return mergeable_params

    def _process_all_pta_parameters(
        self, mergeable_params: List[str]
    ) -> Tuple[Dict, Dict]:
        """Process parameters from all PTAs."""
        fitparameters = {}
        setparameters = {}

        # Create PINT models from file data
        pint_models = {}
        for pta_name in self.file_data.keys():
            parfile_content = self._get_parfile_content(pta_name)
            pint_models[pta_name] = self._create_model(parfile_content)

        for pta_name, model in pint_models.items():
            self._process_pta_parameters(
                pta_name, model, mergeable_params, fitparameters, "free"
            )
            self._process_pta_parameters(
                pta_name, model, mergeable_params, setparameters, "all"
            )

            # Make sure Offset is added if PHOFF is not present
            # Neither Enterprise nor PINT report that parameter that is
            # typically sneakily fit for
            if "PHOFF" not in model.params:
                self._add_pta_specific_parameter(
                    "PHOFF", pta_name, "Offset", fitparameters
                )
                self._add_pta_specific_parameter(
                    "PHOFF", pta_name, "Offset", setparameters
                )

        return fitparameters, setparameters

    def _process_pta_parameters(
        self,
        pta_name: str,
        model: TimingModel,
        mergeable_params: List[str],
        target_dict: Dict,
        parameter_type: str = "all",
    ) -> None:
        """Process parameters for a single PINT model.

        Args:
            pta_name: Name of the PTA
            model: PINT TimingModel instance
            mergeable_params: List of parameters that should be merged
            target_dict: Dictionary to update with parameters
            parameter_type: Type of parameters to process ("free" or "all")
        """
        if parameter_type == "free":
            param_list = model.free_params  # Only free (unfrozen) parameters
            self.logger.debug(
                f"Processing PTA '{pta_name}' with {len(param_list)} free parameters"
            )
        else:
            param_list = model.params  # ALL parameters present in model
            self.logger.debug(
                f"Processing PTA '{pta_name}' with {len(param_list)} total parameters"
            )

        for param_name in param_list:
            meta_parname = self.resolve_parameter_aliases(param_name)

            # Check if this parameter should be merged
            if param_name in mergeable_params:
                # Add as merged parameter - will fail if not available across PTAs
                self._add_merged_parameter(
                    meta_parname, pta_name, param_name, target_dict
                )
            else:
                # Parameter not mergeable (detector-specific), make it PTA-specific
                self._add_pta_specific_parameter(
                    meta_parname, pta_name, param_name, target_dict
                )

    def _add_merged_parameter(
        self, meta_parname: str, pta_name: str, param_name: str, target_dict: Dict
    ) -> None:
        """Add a merged parameter to target dictionary."""
        if meta_parname not in target_dict:
            target_dict[meta_parname] = {}
        target_dict[meta_parname][pta_name] = param_name

    def _add_pta_specific_parameter(
        self, meta_parname: str, pta_name: str, param_name: str, target_dict: Dict
    ) -> None:
        """Add a PTA-specific parameter to target dictionary."""
        # For PTA-specific parameters, use the original parameter name
        full_parname = f"{param_name}_{pta_name}"
        target_dict[full_parname] = {pta_name: param_name}

    def _validate_parameter_consistency(
        self, fitparameters: Dict, setparameters: Dict
    ) -> None:
        """Validate parameter consistency."""
        # Check that all fit parameters are also in set parameters
        fit_param_names = set(fitparameters.keys())
        set_param_names = set(setparameters.keys())

        missing_from_set = fit_param_names - set_param_names
        if missing_from_set:
            raise ParameterInconsistencyError(
                f"Fit parameters not found in set parameters: {missing_from_set}"
            )

    def _build_parameter_mapping_result(
        self, fitparameters: Dict, setparameters: Dict
    ) -> "ParameterMapping":
        """Build the final ParameterMapping result."""
        merged_parameters = [
            name for name in fitparameters.keys() if len(fitparameters[name]) > 1
        ]
        pta_specific_parameters = [
            name for name in fitparameters.keys() if len(fitparameters[name]) == 1
        ]

        return ParameterMapping(
            fitparameters=fitparameters,
            setparameters=setparameters,
            merged_parameters=merged_parameters,
            pta_specific_parameters=pta_specific_parameters,
        )

    # ===== PARAMETER RESOLUTION METHODS =====

    def resolve_parameter_aliases(self, param_name: str) -> str:
        """Resolve parameter aliases to canonical names."""
        canonical = resolve_parameter_alias(param_name)
        if canonical != param_name:
            self.logger.debug(
                f"Resolved parameter alias '{param_name}' -> '{canonical}'"
            )
        return canonical

    def check_component_available_across_ptas(self, component_type: str) -> bool:
        """Check if component type is available across all PINT models."""
        for pta_name in self.file_data.keys():
            parfile_content = self._get_parfile_content(pta_name)
            model = self._create_model(parfile_content)

            if not check_component_available_in_model(model, component_type):
                return False
        return True

    def check_parameter_identifiable(self, pta_name: str, param_name: str) -> bool:
        """Check if parameter is identifiable in specific PINT model."""
        if pta_name not in self.file_data:
            return False

        parfile_content = self._get_parfile_content(pta_name)
        model = self._create_model(parfile_content)
        return get_parameter_identifiability_from_model(model, param_name)

    def _parse_parfiles(self) -> Dict[str, Dict]:
        """Parse parfile content strings into dictionaries using PINT's parse_parfile."""
        return {
            pta_name: parse_parfile(StringIO(self._get_parfile_content(pta_name)))
            for pta_name in self.file_data.keys()
        }

    def _get_parfile_content(self, pta_name: str) -> str:
        """Get parfile content for a specific PTA from file data."""
        return self.file_data[pta_name]["par_content"]

    def _get_timing_package(self, pta_name: str) -> str:
        """Get timing package for a specific PTA from file data.

        Defaults to ``"pint"``: parameter-mapping callers build ``file_data``
        from already-materialized models and do not carry the key.
        """
        return self.file_data[pta_name].get("timing_package", "pint")

    def _convert_tempo2_to_tdb(self, parfile_content: str) -> str:
        """Convert par file from TCB to TDB using tempo2 subprocess."""
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".par", delete=False
        ) as input_file:
            input_file.write(parfile_content)
            input_file.flush()

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".par", delete=False
            ) as output_file:
                try:
                    # Run tempo2 transform command
                    subprocess.run(
                        [
                            "tempo2",
                            "-gr",
                            "transform",
                            input_file.name,
                            output_file.name,
                            "tdb",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    # Read converted content
                    output_file.seek(0)
                    converted_content = output_file.read()

                    # Old tempo2 builds (pre-bf00f36) write NE_SW twice in
                    # transform output; PINT rejects duplicated non-repeatable
                    # parameters, so sanitize at this ingestion boundary.
                    return dedupe_nonrepeatable_par_lines(converted_content)

                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Tempo2 conversion failed: {e.stderr}") from e
                finally:
                    # Clean up temporary files
                    input_file.close()
                    output_file.close()
                    Path(input_file.name).unlink(missing_ok=True)
                    Path(output_file.name).unlink(missing_ok=True)

    def _is_parameter_for_component(
        self, param_name: str, component_params: List[str]
    ) -> bool:
        """Check if parameter belongs to a specific component."""
        return param_name in component_params

    def _get_parfile_dicts(self) -> Dict[str, Dict]:
        """Get parfile dictionaries for all PTAs."""
        return self._parse_parfiles()


class ParameterMapping:
    """Data class for parameter mapping results."""

    def __init__(
        self,
        fitparameters: Dict,
        setparameters: Dict,
        merged_parameters: List[str],
        pta_specific_parameters: List[str],
    ):
        self.fitparameters = fitparameters  # Only FREE parameters (unfrozen)
        self.setparameters = setparameters  # ALL parameters present in model
        self.merged_parameters = merged_parameters
        self.pta_specific_parameters = pta_specific_parameters


class ParameterInconsistencyError(Exception):
    """Raised when parameters are inconsistent across PTAs"""

    pass
