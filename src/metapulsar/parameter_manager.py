"""Unified parameter and par file management for multi-PTA pulsar data.

This module consolidates all parameter management functionality:
- Making par files consistent across PTAs
- Building parameter mappings for MetaPulsar
- Resolving parameter aliases and availability
- Working with both PINT and Tempo2 PTAs
"""

from __future__ import annotations

import re
import math
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from io import StringIO
from typing import Dict, Iterable, List, Any, Literal, Tuple, Optional, Set
import logging

from pint.models.model_builder import parse_parfile
from pint.models.timing_model import TimingModel
from pint.models.binary_ell1 import BinaryELL1H

from .pint_helpers import (
    Ell1hShapiroMode,
    resolve_parameter_alias,
    resolve_parfile_parameter_name,
    get_aliases_for_parameter,
    create_pint_model,
    get_parameters_by_type_from_models,
    check_component_available_in_model,
    get_parameter_identifiability_from_model,
    dict_to_parfile_string,
    dedupe_nonrepeatable_par_lines,
    parse_par_token,
    si_from_par,
    detect_astrometry_style,
)

logger = logging.getLogger(__name__)


# ===== ALIGNMENT POLICY =====

UnsupportedPolicy = Literal["strip", "error"]
BinaryConversionMode = Literal["auto", "off", "always"]
UnsupportedBinaryPolicy = Literal["error", "keep"]
H3OnlyPolicy = Literal["error", "sample_stigma"]
MixedOrthometricSextetPolicy = Literal["warn_unfreeze", "error"]
ConventionProfile = Literal["auto", "always"]


@dataclass(frozen=True)
class AlignmentPolicy:
    """Policy for the multi-PTA ``shared`` combination strategy.

    The shared strategy rewrites every PTA's par file onto one common
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
        convention_profile: ``"auto"`` (default) keeps today's gated alignment.
            ``"always"`` forces the validated mixed-engine deterministic surface
            and TDB normalization even when only one PTA or one timing package
            is present, so a single-PTA product can be PINT↔Tempo2 round-tripped
            without a donor leg.
        binary_conversion: Gated ELL1-family → DD/DDH conversion mode.
        binary_conversion_threshold_s: Scale-gate threshold in seconds.
        unsupported_binary: ``error``/``keep`` when the gate fires on an
            unsupported family.
        binary_fidelity_floor_s: Absolute floor of the delay-fidelity tolerance.
        binary_fidelity_tolerance_factor: Multiplier applied to every delay-fidelity
            tolerance (Roemer, Shapiro, and total) after the derived budget and
            floor are computed. Default ``1.0`` preserves the published §7.5
            budget; values ``> 1`` loosen the check (e.g. ``1.05`` for a 5%
            margin), values in ``(0, 1)`` tighten it. Does not change the
            scale gate or the conversion map.
        h3_only: ELL1H H3-only handling (``error`` or ``sample_stigma``).
        stigma_central: Prior-central ς for ``h3_only='sample_stigma'``.
        stigma_provenance: Provenance string for ``stigma_central``.
        mixed_orthometric_sextet: When an ELL1H par mixes free/frozen flags on
            ``{A1, EPS1, EPS2, TASC, H3, ς}``, ``"warn_unfreeze"`` (default)
            frees every present sextet member so the all-free conversion path
            applies, and warns that the free subspace grew relative to the
            release; ``"error"`` leaves the mixed flags and lets the §7.2 gate
            refuse with ``unsupported_fit_pattern``.
    """

    unsupported: UnsupportedPolicy = "strip"
    ephem: Optional[str] = None
    clock: Optional[str] = None
    bipm_version: Optional[int] = None
    ne_sw: Optional[float] = None
    convention_profile: ConventionProfile = "auto"

    # --- gated binary-family conversion ---
    binary_conversion: BinaryConversionMode = "auto"
    binary_conversion_threshold_s: float = 1e-9
    unsupported_binary: UnsupportedBinaryPolicy = "error"
    binary_fidelity_floor_s: float = 1e-10
    binary_fidelity_tolerance_factor: float = 1.0
    h3_only: H3OnlyPolicy = "error"
    stigma_central: Optional[float] = None
    stigma_provenance: Optional[str] = None
    mixed_orthometric_sextet: MixedOrthometricSextetPolicy = "warn_unfreeze"

    def __post_init__(self) -> None:
        if self.unsupported not in {"strip", "error"}:
            raise ValueError("unsupported must be 'strip' or 'error'")
        if self.convention_profile not in {"auto", "always"}:
            raise ValueError("convention_profile must be 'auto' or 'always'")
        if self.bipm_version is not None and (
            isinstance(self.bipm_version, bool)
            or not isinstance(self.bipm_version, int)
            or not 1000 <= self.bipm_version <= 9999
        ):
            raise ValueError("bipm_version must be a four-digit integer year")
        if self.ne_sw is not None:
            if isinstance(self.ne_sw, bool):
                raise ValueError("ne_sw must be a finite non-negative number")
            try:
                ne_sw = float(self.ne_sw)
            except (TypeError, ValueError) as exc:
                raise ValueError("ne_sw must be a finite non-negative number") from exc
            if not math.isfinite(ne_sw) or ne_sw < 0:
                raise ValueError("ne_sw must be a finite non-negative number")
        if self.binary_conversion not in {"auto", "off", "always"}:
            raise ValueError("binary_conversion must be 'auto', 'off', or 'always'")
        for name in ("binary_conversion_threshold_s", "binary_fidelity_floor_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not float(value) > 0
            ):
                raise ValueError(f"{name} must be a finite number > 0")
        factor = self.binary_fidelity_tolerance_factor
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor))
            or not float(factor) > 0
        ):
            raise ValueError(
                "binary_fidelity_tolerance_factor must be a finite number > 0"
            )
        if self.unsupported_binary not in {"error", "keep"}:
            raise ValueError("unsupported_binary must be 'error' or 'keep'")
        if self.h3_only not in {"error", "sample_stigma"}:
            raise ValueError("h3_only must be 'error' or 'sample_stigma'")
        if self.h3_only == "sample_stigma":
            sc = self.stigma_central
            if (
                sc is None
                or isinstance(sc, bool)
                or not isinstance(sc, (int, float))
                or not 0.0 < float(sc) <= 1.0
            ):
                raise ValueError(
                    "h3_only='sample_stigma' requires stigma_central in (0, 1]"
                )
            if not self.stigma_provenance:
                raise ValueError("h3_only='sample_stigma' requires stigma_provenance")
        elif self.stigma_central is not None or self.stigma_provenance is not None:
            raise ValueError(
                "stigma_central/stigma_provenance require h3_only='sample_stigma'"
            )
        if self.mixed_orthometric_sextet not in {"warn_unfreeze", "error"}:
            raise ValueError(
                "mixed_orthometric_sextet must be 'warn_unfreeze' or 'error'"
            )


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


#: Flag-selected white-noise keywords whose bare spelling is not portable.
#: Tempo2 declares ``EFAC`` and ``EQUAD`` as global scalars, so a PINT-flavoured
#: ``EFAC -f <flagval> <value>`` line makes it read the flag name as the value
#: and every TOA uncertainty becomes NaN, which silently NaNs the whole residual
#: series. PINT declares the tempo2 spellings as aliases of its own parameters,
#: so those are the portable pin -- the same argument that picks ``CLK`` over
#: ``CLOCK``.
PORTABLE_NOISE_KEYWORDS: Dict[str, str] = {
    "EFAC": "T2EFAC",
    "EQUAD": "T2EQUAD",
    "ECORR": "TNECORR",
}


def _is_flag_selector(entry: str) -> bool:
    """True when a par entry starts with a Tempo2 flag (``-f``), not a number."""
    token = entry.split()[0] if entry.split() else ""
    return len(token) > 1 and token[0] == "-" and not token[1].isdigit()


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
        "Equatorial astrometry detected. PINT/tempo2 agreement after T2CMETHOD "
        "modification is typically a few ns (about 6 ns on NG5 J1600), "
        "slightly larger than ecliptic pars with full convention alignment "
        "(about 1 ns on NG11 J1600). Reason: no explicit ECL obliquity "
        "convention to align; residual differences from ecliptic-frame "
        "geometry entering delay terms may remain. For best agreement, prefer "
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
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        alignment_policy: Optional[AlignmentPolicy] = None,
    ):
        """Initialize with file data and configuration.

        Args:
            file_data: File data from FileDiscovery
            combine_components: List of components to make consistent
            add_dm_derivatives: Whether to add DM1, DM2 parameters
            output_dir: Directory for output files
            pulsar_name: Name of the pulsar (used for output filename generation)
            exclude_from_shared: Canonical timing-model parameter names to keep
                PTA-specific even when their component is in combine_components.
                Defaults to ("DM",) so each PTA keeps its own reference DM while
                shared dispersion still shares DM1/DM2. Pass an empty list to
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
        self.exclude_from_shared = self._normalize_excluded_shared_parameters(
            exclude_from_shared
        )
        self.alignment_policy = alignment_policy or AlignmentPolicy()

        # Use first dictionary key as reference (consistent with MetaPulsarFactory)
        self.reference_pta = next(iter(file_data.keys()))

        self.logger = logger

        # Cache for PINT models
        self._pint_models_cache: Optional[Dict[str, TimingModel]] = None
        self.last_binary_conversion_report = None

    def _force_full_convention_profile(self) -> bool:
        return self.alignment_policy.convention_profile == "always"

    @property
    def ell1h_shapiro(self) -> Ell1hShapiroMode:
        """ELL1H orthometric Shapiro convention used for every PINT model build.

        ``"absorbed"`` on mixed PINT+Tempo2 stacks (and when
        ``convention_profile="always"``) so temporary models here match the
        factory's materialization; ``"full"`` (PINT's default) otherwise.
        """
        if self._force_full_convention_profile():
            return "absorbed"
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

    def _normalize_excluded_shared_parameters(
        self, exclude_from_shared: List[str] | tuple[str, ...]
    ) -> set[str]:
        """Return canonical PINT names excluded from shared component merging."""
        return {
            resolve_parameter_alias(param).upper()
            for param in tuple(exclude_from_shared)
        }

    def _is_excluded_from_shared(self, param_name: str) -> bool:
        return resolve_parameter_alias(param_name).upper() in self.exclude_from_shared

    # ===== MAIN PUBLIC METHODS =====

    def make_parfiles_shared(self) -> Dict[str, Path]:
        """Align charts, normalize units, then share model components across PTAs.

        Chart alignment runs FIRST and unconditionally (feature doc S1.5): the
        component merge copies values keyed by names present in the reference
        PTA's parfile dict, so a hybrid reference would key the orbit as ``PB``
        and reintroduce it as the shared binary chart. Aligning first makes the
        merge chart-safe whatever the PTA ordering.

        The remaining ordering is fixed:

        1. align the orbital chart and parse with PINT;
        2. expand aggregate ``TEMPO1``;
        3. strip (or reject) unsupported deterministic families so later PINT
           model creation is safe;
        4. normalize every par to explicit ``UNITS TDB``;
        5. numerically transform ecliptic astrometry;
        6. apply the explicit convention profile (engine-gated);
        7. copy the selected reference components;
        8. apply the existing dispersion cleanup;
        9. serialize.

        Returns:
            Dictionary of written par file paths for each PTA
        """
        self.logger.info("Making par files consistent across PTAs")
        # Existing pattern: start a consistency run with a clean model cache.
        # file_data is never mutated (invariant 1), so a rebuild still yields
        # *release* models -- do not "clear because alignment rewrote content".
        self._clear_pint_models_cache()
        # A1: never let a stale conversion report survive manager reuse.
        self.last_binary_conversion_report = None

        if self._force_full_convention_profile():
            packages = sorted(self._normalized_timing_packages())
            self.logger.info(
                "Convention profile forced (convention_profile=always); applying "
                f"mixed-engine common surface to {len(self.file_data)} PTA(s), "
                f"packages={packages}"
            )

        # 1. Align the orbital chart, then parse into dictionaries
        parfile_dicts = {
            pta_name: parse_parfile(StringIO(content))
            for pta_name, (content, _) in self._aligned_parfile_contents().items()
        }

        # 2-3. Expand TEMPO1 and remove unsupported deterministic families
        self._prepare_common_surface(parfile_dicts)

        # 4. Normalize every par to explicit UNITS TDB
        parfile_contents = {
            pta_name: dict_to_parfile_string(par, format="pint")
            for pta_name, par in parfile_dicts.items()
        }
        converted_parfiles = self._convert_units_if_needed(parfile_contents)

        # 5-8. Transform astrometry, align conventions, merge components, DM cleanup
        shared_parfiles = self._make_parameters_shared(converted_parfiles)

        # 9. Write shared par files to output directory
        output_files = self._write_shared_parfiles(shared_parfiles)

        self.logger.info(
            f"Successfully created {len(output_files)} consistent par files"
        )
        return output_files

    def engine_parfiles(self) -> Dict[str, Path]:
        """Par files each engine should consume, with no cross-PTA harmonization.

        Used by the ``per_pta`` strategy: the orbital chart is aligned to PINT's,
        and nothing else happens. No unit conversion, no parameter sharing.

        Returns the caller's own path for every PTA whose par already agrees with
        its model (invariant 5), so an unaffected pulsar writes no file at all
        and its engine consumes -- and ``_retain_pta_files`` captures -- the data
        release's file byte for byte.

        The no-write path requires ``file_data[pta]["par"]`` to be a real path.
        When ``par`` is ``None`` (test doubles), content is always written even
        if ``changed`` is False so the caller still receives a usable path.
        Factory discovery always supplies paths in production.
        """
        paths: Dict[str, Path] = {}
        for pta_name, (content, changed) in self._aligned_parfile_contents().items():
            source = self.file_data[pta_name].get("par")
            if not changed and source is not None:
                paths[pta_name] = Path(source)
                continue
            paths[pta_name] = self._write_parfile(pta_name, content, tag="aligned")
        return paths

    def _aligned_parfile_contents(self) -> Dict[str, Tuple[str, bool]]:
        """Per-PTA par text with the orbital chart aligned to its PINT model.

        Pipeline-local: ``self.file_data`` is NOT modified (feature doc
        invariant 1), so ``par_content`` keeps the data release's own bytes and
        ``pint_models`` stays a model of release content -- which is what makes
        it the correct trigger authority.

        Applies to every PTA regardless of ``timing_package``. See feature doc
        S1.5: an unaligned hybrid par used as the merge reference would
        reintroduce ``PB`` as the shared binary chart.
        """
        from .pint_helpers import align_orbital_chart

        models = self.pint_models  # built from release content, cached
        aligned: Dict[str, Tuple[str, bool]] = {}
        for pta_name in self.file_data:
            aligned[pta_name] = align_orbital_chart(
                self._get_parfile_content(pta_name),
                models[pta_name],
                timing_package=self._get_timing_package(pta_name),
                pta_name=pta_name,
            )
        changed = sorted(p for p, (_, c) in aligned.items() if c)
        if changed:
            self.logger.info(f"Orbital chart aligned for PTAs: {changed}")
        return aligned

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

        Only multi-PTA combinations are rewritten under ``convention_profile="auto"``:
        a single-PTA pulsar has no second engine or reference to align against, so
        its native deterministic model is preserved. With
        ``convention_profile="always"``, TEMPO1 expand and unsupported stripping
        still run for a single PTA (treating the stack as mixed for
        ``_is_mixed_unsafe`` keys). Invalid orthometric parameter combinations
        are always rejected early.
        """
        for pta_name, par in parfile_dicts.items():
            self._reject_orthometric_conflict(pta_name, par)

        force = self._force_full_convention_profile()
        if len(parfile_dicts) < 2 and not force:
            self.logger.info("Common-surface preparation skipped (reason=single_pta)")
            return

        normalized_packages = self._normalized_timing_packages()
        has_pint = "pint" in normalized_packages
        has_tempo2 = "tempo2" in normalized_packages
        mixed = force or self._is_cross_engine_mix(normalized_packages)

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
        stigma_names = present & set(get_aliases_for_parameter("STIGMA"))
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

    @staticmethod
    def _iter_active_par_lines(content: str):
        """Yield non-comment active parfile lines (tempo2 ``C `` / ``#`` skipped)."""
        for line in content.splitlines():
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.upper().startswith("C ") or stripped.upper() == "C":
                continue
            yield line

    @classmethod
    def _active_units_lines(cls, content: str) -> list[str]:
        """Return active lines whose first token is UNITS (case-insensitive)."""
        units_lines: list[str] = []
        for line in cls._iter_active_par_lines(content):
            tokens = line.split()
            if tokens and tokens[0].upper() == "UNITS":
                units_lines.append(line)
        return units_lines

    @classmethod
    def _assert_single_units_tdb(cls, content: str, *, pta_name: str) -> None:
        """Require exactly one active ``UNITS TDB`` line after conversion."""
        units_lines = cls._active_units_lines(content)
        if len(units_lines) != 1:
            raise ValueError(
                f"PTA {pta_name!r}: expected exactly one active UNITS line after "
                f"normalization, found {len(units_lines)}"
            )
        tokens = units_lines[0].split()
        if len(tokens) < 2 or tokens[1].upper() != "TDB":
            raise ValueError(
                f"PTA {pta_name!r}: expected UNITS TDB after normalization, "
                f"got {units_lines[0]!r}"
            )

    @staticmethod
    def _stamp_units_tdb(content: str) -> str:
        """Append an explicit ``UNITS TDB`` line without reformatting other bytes."""
        stamped = content
        if stamped and not stamped.endswith("\n"):
            stamped += "\n"
        stamped += "UNITS TDB\n"
        return stamped

    def _effective_units_for_content(self, pta_name: str, content: str) -> str:
        """Resolve effective time units from raw active UNITS lines.

        Replaces the previous PINT-parse path: package-aware ``SI`` handling and
        targeted ``ValueError``s require a raw-line scan.
        """
        units_lines = self._active_units_lines(content)
        if len(units_lines) > 1:
            raise ValueError(
                f"PTA {pta_name!r}: duplicate active UNITS lines "
                f"({len(units_lines)}); refuse silent first/last-line-wins"
            )

        timing_package = self._get_timing_package(pta_name)
        if not units_lines:
            return "TCB" if timing_package == "tempo2" else "TDB"

        tokens = units_lines[0].split()
        if len(tokens) < 2:
            raise ValueError(
                f"PTA {pta_name!r}: UNITS line has no value: {units_lines[0]!r}"
            )
        value = tokens[1].upper()
        if value == "TDB":
            return "TDB"
        if value == "TCB":
            return "TCB"
        if value == "SI":
            if timing_package == "tempo2":
                # tempo2 treats SI as a synonym for TCB (SI_UNITS).
                return "TCB"
            raise ValueError(
                f"PTA {pta_name!r}: UNITS SI is tempo2 syntax; PINT accepts "
                "only TDB/TCB"
            )
        raise ValueError(
            f"PTA {pta_name!r}: unknown UNITS value {tokens[1]!r} "
            "(expected TDB, TCB, or tempo2-owned SI)"
        )

    def _normalize_parfile_to_tdb(self, pta_name: str, content: str) -> str:
        """Normalize one par to explicit UNITS TDB (convert or stamp as needed)."""
        effective = self._effective_units_for_content(pta_name, content)
        if effective == "TCB":
            timing_package = self._get_timing_package(pta_name)
            if timing_package == "pint":
                converted = self._convert_pint_to_tdb(content)
            else:
                converted = self._convert_tempo2_to_tdb(content)
            self._assert_single_units_tdb(converted, pta_name=pta_name)
            return converted

        if effective != "TDB":
            raise ValueError(
                f"PTA {pta_name!r}: unsupported effective units {effective!r}"
            )

        units_lines = self._active_units_lines(content)
        if not units_lines:
            stamped = self._stamp_units_tdb(content)
            self._assert_single_units_tdb(stamped, pta_name=pta_name)
            return stamped

        # Already carries an explicit UNITS TDB line — leave byte-identical.
        self._assert_single_units_tdb(content, pta_name=pta_name)
        return content

    def _convert_units_if_needed(
        self, parfile_contents: Dict[str, str]
    ) -> Dict[str, str]:
        """Normalize to TDB only when the stack requires a common timescale.

        Operates on the text it is given. It previously discarded its argument
        and re-read ``self.file_data``, which would undo chart alignment
        (feature doc invariant 1).

        Mixed PINT/Tempo2 stacks always use explicit TDB. Homogeneous
        single-engine stacks preserve a common native timescale, while
        genuinely mixed TCB/TDB inputs are normalized to TDB. A single PTA
        likewise preserves its native timescale under ``convention_profile="auto"``.
        With ``convention_profile="always"``, every PTA is normalized to
        explicit ``UNITS TDB``.
        """
        force = self._force_full_convention_profile()
        file_units = {
            pta_name: self._effective_units_for_content(pta_name, content)
            for pta_name, content in parfile_contents.items()
        }
        if not force and len(parfile_contents) <= 1:
            self.logger.info("Unit normalization skipped (reason=single_pta)")
            return parfile_contents

        unique_units = set(file_units.values())
        mixed_engines = self._is_cross_engine_mix(self._normalized_timing_packages())
        if not force and len(unique_units) == 1 and not mixed_engines:
            self.logger.info(
                "Unit normalization skipped "
                "(reason=homogeneous_single_engine_timescale)"
            )
            return parfile_contents

        self.logger.info("Normalizing all par files to explicit UNITS TDB")
        return {
            pta_name: self._normalize_parfile_to_tdb(pta_name, content)
            for pta_name, content in parfile_contents.items()
        }

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

        Retained for callers/tests that still pass precomputed unit maps; the
        primary path is :meth:`_convert_units_if_needed`. The collection need
        not be mixed; an all-TCB collection is converted too.
        """
        converted_parfiles = {}

        for pta_name, parfile_content in parfile_contents.items():
            current_units = file_units[pta_name]

            if current_units == "TDB":
                converted_parfiles[pta_name] = self._normalize_parfile_to_tdb(
                    pta_name, parfile_content
                )
            else:
                timing_package = self._get_timing_package(pta_name)

                if timing_package == "pint":
                    try:
                        converted_content = self._convert_pint_to_tdb(parfile_content)
                        self._assert_single_units_tdb(
                            converted_content, pta_name=pta_name
                        )
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
                    try:
                        converted_content = self._convert_tempo2_to_tdb(parfile_content)
                        self._assert_single_units_tdb(
                            converted_content, pta_name=pta_name
                        )
                        converted_parfiles[pta_name] = converted_content
                        self.logger.debug(f"Converted PTA {pta_name} using Tempo2")
                    except Exception as e:
                        self.logger.error(
                            f"Tempo2 conversion failed for PTA {pta_name}: {e}"
                        )
                        raise RuntimeError(
                            f"Tempo2 unit conversion failed for PTA {pta_name}"
                        ) from e

        return converted_parfiles

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

        Alias-aware via the SI boundary: NANOGrav-style par files spell it
        SOLARN0 (PINT aliases: NE1AU, SOLARN0), which must count as an
        explicit value.
        """
        value = si_from_par(parfile_dict, "NE_SW", default=None)
        return None if value is None else float(value)

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

        clock_aliases = get_aliases_for_parameter("CLOCK")
        clock = policy.clock or _first_token(
            self._first_alias_value(reference_dict, "CLOCK")
        )
        if clock is None:
            raise ValueError(
                "Consistent alignment requires a clock realization: no alias "
                f"from {clock_aliases} found in reference PTA "
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
        parfile_dict: Dict[str, List[str]],
        value: str,
    ) -> None:
        """Write the resolved clock as ``CLK``, which both PINT and Tempo2 honour.

        Tempo2 only reads ``CLK`` (it silently ignores ``CLOCK`` and falls back
        to its default realization). PINT declares ``CLOCK`` with
        ``aliases=["CLK"]``, so the Tempo2 spelling is the portable pin.
        Never emit both keywords — PINT rejects that as a non-repeatable
        parameter. See ``bug_clock_keyword_portability.md``.
        """
        for alias in get_aliases_for_parameter("CLOCK"):
            parfile_dict.pop(alias, None)
        parfile_dict["CLK"] = [value]

    def _set_portable_noise_keywords(
        self,
        pta_name: str,
        parfile_dict: Dict[str, List[str]],
    ) -> None:
        """Rewrite flag-selected white-noise keywords to their portable spelling.

        Only entries that carry a flag selector are moved: a bare ``EFAC 1.5``
        is Tempo2's global scalar and stays where it is. See
        ``PORTABLE_NOISE_KEYWORDS``.
        """
        for source, target in PORTABLE_NOISE_KEYWORDS.items():
            entries = parfile_dict.get(source)
            if not entries:
                continue
            moved = [entry for entry in entries if _is_flag_selector(entry)]
            if not moved:
                continue
            kept = [entry for entry in entries if not _is_flag_selector(entry)]
            if kept:
                parfile_dict[source] = kept
            else:
                parfile_dict.pop(source)
            parfile_dict[target] = parfile_dict.get(target, []) + moved
            self.logger.info(
                f"PTA {pta_name}: rewrote {len(moved)} {source} line(s) as "
                f"{target} (portable spelling)"
            )

    def _align_reference_conventions(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        ephem: str,
        clock: str,
    ) -> None:
        """Apply the resolved EPHEM and clock convention to every PTA."""
        for pta_name, parfile_dict in parfile_dicts.items():
            self._set_aliased_value(parfile_dict, ["EPHEM"], ephem)
            self._set_engine_clock_value(parfile_dict, clock)
            self._set_portable_noise_keywords(pta_name, parfile_dict)

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
                f"(reason=cross_engine_agreement, style={style}); "
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
            reference_t2cmethod = (
                self._parse_t2cmethod_value(reference_dict) or "IAU2000B"
            )
            for pta_name, parfile_dict in parfile_dicts.items():
                if reference_t2cmethod == "IAU2000B":
                    parfile_dict["T2CMETHOD"] = ["IAU2000B"]
                else:
                    self._align_parameter(
                        parfile_dict, reference_dict, "T2CMETHOD", required=False
                    )
                self.logger.info(
                    f"PTA {pta_name}: shared convention rules applied "
                    "(reason=tempo2_only_t2cmethod_heterogeneous); "
                    f"set T2CMETHOD={parfile_dict['T2CMETHOD'][0]}"
                )
        else:
            self.logger.info(
                "T2CMETHOD alignment skipped "
                "(reason=homogeneous_t2cmethod_single_engine_tempo2)"
            )

    def _apply_shared_convention_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
    ) -> None:
        """Apply gated convention rules across shared parfile dictionaries."""
        # Astrometry validation is local to each model and must still run for a
        # single PTA, even though cross-PTA convention alignment is skipped under
        # convention_profile="auto".
        convention_states = self._collect_convention_states(parfile_dicts)
        force = self._force_full_convention_profile()
        if len(parfile_dicts) == 1 and not force:
            self.logger.info("Shared convention rules skipped (reason=single_pta)")
            return

        ephem, clock = self._resolve_reference_conventions(reference_dict)

        # Multi-PTA combinations (and forced full-profile stacks) share the
        # reference ephemeris and clock.
        self._align_reference_conventions(parfile_dicts, ephem, clock)

        normalized_packages = self._normalized_timing_packages()

        if force or self._is_cross_engine_mix(normalized_packages):
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

        if self._force_full_convention_profile() or self._is_cross_engine_mix(
            normalized_packages
        ):
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
        if len(parfile_dicts) == 1 and not self._force_full_convention_profile():
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
            astrometry = next(
                component
                for component in converted.components.values()
                if component.category == "astrometry"
            )
            # ECL is an explicit MetaPulsar frame-policy field, and KOM is
            # retained for orientation-dependent DDK models. Current PINT also
            # includes ECL in AstrometryEcliptic.params.
            canonical_copy_names = list(
                dict.fromkeys([*astrometry.params, "ECL", "KOM"])
            )
            input_names = {
                alias
                for canonical in canonical_copy_names
                for alias in get_aliases_for_parameter(canonical)
            }
            written = StringIO()
            converted.write_parfile(written)
            transformed = parse_parfile(StringIO(written.getvalue()))
        except Exception as e:
            raise ValueError(
                f"PTA {pta_name}: ecliptic astrometry conversion "
                f"{source_ecl or '(missing)'} -> {target_ecl} failed: {e}"
            ) from e

        for key in list(par):
            if key.upper() in input_names:
                par.pop(key)
        for key in canonical_copy_names:
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
        mixed = self._force_full_convention_profile() or self._is_cross_engine_mix(
            self._normalized_timing_packages()
        )

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
        declared = []
        for name in ("NHARMS", "NHARM"):
            key = present.get(name)
            if key and par.get(key):
                try:
                    declared.append(int(str(par[key][0]).split()[0]))
                except (TypeError, ValueError, IndexError):
                    continue
        return max(declared) if declared else None

    @staticmethod
    def _ell1h_state(model: TimingModel) -> Tuple[Optional[str], Optional[int]]:
        """Return PINT's selected ELL1H branch and effective harmonic count."""
        ell1h = next(
            (
                component
                for component in model.components.values()
                if isinstance(component, BinaryELL1H)
            ),
            None,
        )
        if ell1h is None:
            return None, None
        if model.STIGMA.quantity is not None:
            return "stigma", None
        if model.H3.quantity is not None and model.H4.quantity is not None:
            return "h4", int(model.NHARMS.value)
        return None, None

    def _align_ell1h_nharms_for_all(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        pint_models: Dict[str, TimingModel],
    ) -> None:
        """Give mixed-engine ELL1H models the same effective harmonic count.

        ``NHARM`` is unknown to PINT and therefore never copied by the binary
        component merge, so the count has to be resolved once for the stack:
        the largest value any input declared, floored at 7. A finer truncation
        is always safe for both engines.
        """
        if not (
            self._force_full_convention_profile()
            or self._is_cross_engine_mix(self._normalized_timing_packages())
        ):
            return

        binary_is_shared = "binary" in self.combine_components
        states = {}
        effective_nharms = []
        for pta_name in parfile_dicts:
            model_name = self.reference_pta if binary_is_shared else pta_name
            state, effective = self._ell1h_state(pint_models[model_name])
            states[pta_name] = state
            if effective is not None:
                effective_nharms.append(effective)

        declared = [
            value
            for value in (self._declared_nharms(par) for par in parfile_dicts.values())
            if value is not None
        ]
        nharms = max([*declared, *effective_nharms, 7])
        for pta_name, par in parfile_dicts.items():
            self._align_ell1h_nharms(par, states[pta_name], nharms)

    def _align_ell1h_nharms(
        self,
        par: Dict[str, List[str]],
        state: Optional[str],
        nharms: Optional[int] = None,
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
        if state == "stigma":
            for name in ("NHARM", "NHARMS"):
                if name in present:
                    par.pop(present[name])
            return
        if state != "h4":
            return

        if nharms is None:
            declared = self._declared_nharms(par)
            nharms = 7 if declared is None else max(declared, 7)

        # Dual emit: tempo2 reads NHARM; PINT reads NHARMS (no cross-alias).
        par[present.get("NHARM", "NHARM")] = [str(nharms)]
        par[present.get("NHARMS", "NHARMS")] = [str(nharms)]

    def _make_parameters_shared(self, parfile_data: Dict[str, str]) -> Dict[str, str]:
        """Make parameters shared using reference PTA values.

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
          being made shared, and find all parameters in the models
        - For each component, replace the parameters with the values of the
          reference PTA
        - For dispersion, remove DMX parameters
        - Optionally, add DM1 and DM2 parameters
        - Convert back to par file strings
        - Write shared par files to output directory

        This method is deterministic, so we do not have to save the new parfiles
        (but we can, as an option)

        Args:
            parfile_data: Dictionary of parfile contents for each PTA

        Returns:
            Dictionary of shared parfile contents for each PTA
        """
        self.logger.info(
            f"Making parameters shared using reference PTA: {self.reference_pta}"
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
            self._apply_shared_convention_rules(parfile_dicts, reference_dict)
        except ValueError as e:
            self.logger.error(f"Shared convention rules failed: {e}")
            raise RuntimeError(f"Shared convention rules failed: {e}") from e

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
            self._make_component_parameters_shared(
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
        self._align_ell1h_nharms_for_all(parfile_dicts, pint_models)

        # Gated ELL1-family → DD/DDH conversion (mixed-engine shared path).
        self._maybe_convert_shared_binary(parfile_dicts)

        # Convert back to par file strings
        shared_parfiles = {}
        for pta_name, parfile_dict in parfile_dicts.items():
            try:
                consistent_content = dict_to_parfile_string(parfile_dict, format="pint")
                shared_parfiles[pta_name] = consistent_content
                self.logger.debug(f"Converted PTA {pta_name} par file back to string")
            except Exception as e:
                self.logger.error(f"Error converting par file for PTA {pta_name}: {e}")
                raise RuntimeError(
                    f"Failed to convert par file for PTA {pta_name}"
                ) from e

        return shared_parfiles

    def _maybe_convert_shared_binary(
        self, parfile_dicts: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Orchestrate Contract 1–2 binary conversion (§8.2)."""
        from .binary_family_convert import (
            BinaryConversionError,
            BinaryConversionReport,
            apply_binary_patch,
            assert_postconditions,
            convert_shared_binary,
            decide_binary_conversion,
            prepare_mixed_orthometric_sextet,
            remediation_message,
            _is_mixed_orthometric_sextet_refusal,
            _nonbinary_snapshot,
            _unsupported_message,
        )

        timing_packages = {pta: self._get_timing_package(pta) for pta in parfile_dicts}
        force_mixed_engine = self._force_full_convention_profile()
        decision = decide_binary_conversion(
            parfile_dicts,
            reference_pta=self.reference_pta,
            timing_packages=timing_packages,
            combine_components=self.combine_components,
            policy=self.alignment_policy,
            span_mjd=self._tim_span_mjd(),
            force_mixed_engine=force_mixed_engine,
        )

        detail = decision.warnings[0] if decision.warnings else ""
        if _is_mixed_orthometric_sextet_refusal(decision):
            if self.alignment_policy.mixed_orthometric_sextet == "error":
                message = _unsupported_message(
                    decision.reason,
                    decision.scale,
                    detail=detail,
                    par=parfile_dicts[self.reference_pta],
                    policy=self.alignment_policy,
                )
                # This policy is an explicit hard refusal; unlike the generic
                # unsupported family policy, unsupported_binary="keep" must
                # not downgrade it.
                raise BinaryConversionError(message)

            unfrozen = prepare_mixed_orthometric_sextet(
                parfile_dicts,
                policy=self.alignment_policy,
                decision=decision,
            )
            if not unfrozen:  # pragma: no cover - decision/helper postcondition
                raise BinaryConversionError(
                    "mixed_orthometric_sextet preparation changed no parameters"
                )
            decision = decide_binary_conversion(
                parfile_dicts,
                reference_pta=self.reference_pta,
                timing_packages=timing_packages,
                combine_components=self.combine_components,
                policy=self.alignment_policy,
                span_mjd=self._tim_span_mjd(),
                force_mixed_engine=force_mixed_engine,
            )
            if decision.outcome != "convert":
                raise BinaryConversionError(
                    "mixed_orthometric_sextet postcondition failed: prepared "
                    f"model resolved to {decision.outcome}/{decision.reason}"
                )
            self.logger.warning(
                "Pulsar %s, reference PTA %s: mixed orthometric sextet; "
                "unfreezing %s so ELL1H→DDH conversion can proceed. This "
                "expands the free subspace relative to the release. Set "
                "AlignmentPolicy(mixed_orthometric_sextet='error') to refuse "
                "instead.",
                self.pulsar_name or "(unknown)",
                self.reference_pta,
                ", ".join(unfrozen),
            )

        if decision.outcome == "skip":
            self.last_binary_conversion_report = BinaryConversionReport(
                decision=decision, record=None
            )
            self.logger.info(
                "Binary conversion skipped: reason=%s pint_binary_model=%s",
                decision.reason,
                decision.resolved_binary_model,
            )
            return

        if decision.outcome == "unsupported":
            detail = decision.warnings[0] if decision.warnings else ""
            message = _unsupported_message(
                decision.reason,
                decision.scale,
                detail=detail,
                par=parfile_dicts[self.reference_pta],
                policy=self.alignment_policy,
            )
            if self.alignment_policy.unsupported_binary == "error":
                raise BinaryConversionError(message)
            self.logger.warning(message)
            self.last_binary_conversion_report = BinaryConversionReport(
                decision=decision, record=None
            )
            return

        # outcome == "convert"
        pre_nonbinary = {
            pta: _nonbinary_snapshot(par) for pta, par in parfile_dicts.items()
        }
        try:
            patch, record = convert_shared_binary(
                parfile_dicts[self.reference_pta],
                decision,
                pta_names=tuple(parfile_dicts),
                policy=self.alignment_policy,
                ell1h_shapiro=self.ell1h_shapiro,
            )
        except BinaryConversionError:
            raise
        except Exception as exc:
            raise BinaryConversionError(
                f"conversion_failed: {exc}\n{remediation_message()}"
            ) from exc

        for pta_name, par in parfile_dicts.items():
            # Engine-native spellings (STIGMA vs tempo2's STIG); postcondition 2
            # is alias-resolved. Clock pins always use portable ``CLK``.
            apply_binary_patch(par, patch, timing_package=timing_packages[pta_name])
        self._set_pint_models_from_dicts(parfile_dicts)
        assert_postconditions(
            parfile_dicts,
            target_family=decision.target_family or patch.binary_value,
            pre_nonbinary=pre_nonbinary,
        )
        self.last_binary_conversion_report = BinaryConversionReport(
            decision=decision, record=record
        )
        scale = decision.scale
        fid = record.fidelity
        self.logger.info(
            "Binary conversion applied: %s (PINT %s) → %s reason=%s "
            "scale_s=%s total_max_abs_s=%s",
            decision.source_family,
            decision.resolved_binary_model,
            decision.target_family,
            decision.reason,
            None if scale is None else f"{scale.scale_s:.6e}",
            f"{fid.total_max_abs_s:.6e}",
        )

    def _make_component_parameters_shared(
        self,
        parfile_dicts: Dict[str, Dict],
        reference_dict: Dict,
        reference_pta: str,
        component: str,
        component_params: List[str],
    ) -> None:
        """Make parameters for a specific component shared across PTAs."""
        shared_params = [
            param
            for param in component_params
            if not self._is_excluded_from_shared(param)
        ]

        if not shared_params:
            self.logger.debug(
                f"No non-excluded parameters found for component {component}, skipping"
            )
            return

        reference_values = {}
        for param in shared_params:
            if param in reference_dict:
                reference_values[param] = reference_dict[param]

        for pta_name, parfile_dict in parfile_dicts.items():
            if pta_name == reference_pta:
                continue

            for param in shared_params:
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
        reference_dmepoch, _ = parse_par_token("DMEPOCH", dmepoch_value)
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

            local_dm, dm_is_frozen = parse_par_token("DM", dm_value)
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
        """Get DMX parameters to strip from a parfile for shared dispersion.

        Uses PINT's ``DispersionDMX`` component params as the primary source of
        truth, then also includes raw-dict keys whose prefixes are in PINT's
        ``ignore_prefix`` and start with ``DMX`` (currently ``DMXEP_``,
        ``DMXF1_``, ``DMXF2_``). Those auxiliaries are ignored by PINT on load
        (and unknown to this Tempo2 checkout), so they never appear on the
        component but remain in the ``parse_parfile`` dict unless removed here.
        """
        from pint.exceptions import PrefixError
        from pint.models.timing_model import ignore_prefix
        from pint.utils import split_prefixed_name

        model = self._create_model_from_dict(parfile_dict)

        dmx_params: List[str] = []
        for comp in model.components.values():
            if getattr(comp, "category", None) == "dispersion_dmx":
                dmx_params.extend(getattr(comp, "params", []))

        ignored_dmx_prefixes = {
            prefix for prefix in ignore_prefix if prefix.startswith("DMX")
        }
        for key in parfile_dict:
            try:
                prefix, _, _ = split_prefixed_name(key)
            except PrefixError:
                continue
            if prefix in ignored_dmx_prefixes:
                dmx_params.append(key)

        # Model discovery can invent defaults (e.g. bare ``DMX``) that were
        # never present in the par dict; only return keys we can actually pop.
        return [name for name in dict.fromkeys(dmx_params) if name in parfile_dict]

    def _write_parfile(self, pta_name: str, content: str, *, tag: str) -> Path:
        """Write one par file into the output directory and return its path."""
        if self.output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="metapulsar_parfiles_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self._get_output_filename(pta_name, tag)
        output_path.write_text(content, encoding="utf-8")
        self.logger.debug(f"Written {tag} par file: {output_path}")
        return output_path

    def _write_shared_parfiles(
        self, shared_parfiles: Dict[str, str]
    ) -> Dict[str, Path]:
        """Write consistent par files to output directory."""
        return {
            pta_name: self._write_parfile(pta_name, content, tag="shared")
            for pta_name, content in shared_parfiles.items()
        }

    def _get_output_filename(self, pta_name: str, tag: str = "shared") -> str:
        """Generate output filename for a written par file."""
        if self.pulsar_name:
            return f"{self.pulsar_name}_{tag}_{pta_name}.par"
        return f"{tag}_{pta_name}.par"

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
                param for param in params if not self._is_excluded_from_shared(param)
            )
        return mergeable_params

    def _process_all_pta_parameters(
        self, mergeable_params: List[str]
    ) -> Tuple[Dict, Dict]:
        """Process parameters from all PTAs."""
        fitparameters = {}
        setparameters = {}
        parfile_dicts = self._parse_parfiles()

        # Create PINT models from file data
        pint_models = {}
        for pta_name in self.file_data.keys():
            parfile_content = self._get_parfile_content(pta_name)
            pint_models[pta_name] = self._create_model(parfile_content)

        for pta_name, model in pint_models.items():
            parfile_dict = parfile_dicts[pta_name]
            self._process_pta_parameters(
                pta_name, model, mergeable_params, fitparameters, "free", parfile_dict
            )
            self._process_pta_parameters(
                pta_name, model, mergeable_params, setparameters, "all", parfile_dict
            )

            # Make sure Offset is added if PHOFF is not present
            # Neither Enterprise nor PINT report that parameter that is
            # typically sneakily fit for
            if "PHOFF" not in model.params:
                self._add_pta_specific_parameter(
                    "PHOFF", pta_name, "Offset", "Offset", fitparameters
                )
                self._add_pta_specific_parameter(
                    "PHOFF", pta_name, "Offset", "Offset", setparameters
                )

        return fitparameters, setparameters

    def _process_pta_parameters(
        self,
        pta_name: str,
        model: TimingModel,
        mergeable_params: List[str],
        target_dict: Dict,
        parameter_type: str = "all",
        parfile_dict: Dict[str, Any] | None = None,
    ) -> None:
        """Process parameters for a single PINT model.

        Args:
            pta_name: Name of the PTA
            model: PINT TimingModel instance
            mergeable_params: List of parameters that should be merged
            target_dict: Dictionary to update with parameters
            parameter_type: Type of parameters to process ("free" or "all")
            parfile_dict: Parsed parfile dictionary for backend-native name lookup
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

        if parfile_dict is None:
            parfile_dict = {}

        for param_name in param_list:
            canonical_name = self.resolve_parameter_aliases(param_name)
            mapped_name = resolve_parfile_parameter_name(
                canonical_name,
                parfile_dict,
                fallback=param_name,
            )

            # Check if this parameter should be merged
            if param_name in mergeable_params:
                # Add as merged parameter - will fail if not available across PTAs
                self._add_merged_parameter(
                    canonical_name, pta_name, mapped_name, target_dict
                )
            else:
                # Parameter not mergeable (detector-specific), make it PTA-specific
                self._add_pta_specific_parameter(
                    canonical_name, pta_name, param_name, mapped_name, target_dict
                )

    def _add_merged_parameter(
        self, meta_parname: str, pta_name: str, param_name: str, target_dict: Dict
    ) -> None:
        """Add a merged parameter to target dictionary."""
        if meta_parname not in target_dict:
            target_dict[meta_parname] = {}
        target_dict[meta_parname][pta_name] = param_name

    def _add_pta_specific_parameter(
        self,
        meta_parname: str,
        pta_name: str,
        meta_param_name: str,
        mapped_param_name: str,
        target_dict: Dict,
    ) -> None:
        """Add a PTA-specific parameter to target dictionary."""
        full_parname = f"{meta_param_name}_{pta_name}"
        target_dict[full_parname] = {pta_name: mapped_param_name}

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

    def _tim_span_mjd(self) -> Optional[Tuple[float, float]]:
        """Union ``(mjd_min, mjd_max)`` from PTA ``tim_metadata``, if available.

        Used by the ELL1-family scale gate when the reference par lacks
        ``START``/``FINISH``. Returns ``None`` when no leg carries both ends
        (par-only / ParameterManager mapping callers).
        """
        mins: List[float] = []
        maxs: List[float] = []
        for info in self.file_data.values():
            meta = info.get("tim_metadata")
            if meta is None:
                continue
            mjd_min = getattr(meta, "mjd_min", None)
            mjd_max = getattr(meta, "mjd_max", None)
            if mjd_min is None or mjd_max is None:
                continue
            mins.append(float(mjd_min))
            maxs.append(float(mjd_max))
        if not mins:
            return None
        return (min(mins), max(maxs))

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
