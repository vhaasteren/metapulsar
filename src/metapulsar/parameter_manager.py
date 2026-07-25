"""Unified parameter and par file management for multi-PTA pulsar data.

This module consolidates all parameter management functionality:
- Making par files consistent across PTAs
- Building parameter mappings for MetaPulsar
- Resolving parameter aliases and availability
- Working with both PINT and Tempo2 PTAs
"""

from __future__ import annotations

import tempfile
import subprocess
from pathlib import Path
from io import StringIO
from typing import Dict, List, Any, Tuple, Optional, Set
import logging

from pint.models.model_builder import parse_parfile
from pint.models.timing_model import TimingModel

from .pint_helpers import (
    resolve_parameter_alias,
    resolve_parfile_parameter_name,
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
        """
        self.file_data = file_data
        self.combine_components = combine_components
        self.add_dm_derivatives = add_dm_derivatives
        self.output_dir = output_dir
        self.pulsar_name = pulsar_name
        self.exclude_from_shared = self._normalize_excluded_shared_parameters(
            exclude_from_shared
        )

        # Use first dictionary key as reference (consistent with MetaPulsarFactory)
        self.reference_pta = next(iter(file_data.keys()))

        self.logger = logger

        # Cache for PINT models
        self._pint_models_cache = None

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
                self._pint_models_cache[pta_name] = create_pint_model(parfile_content)
        return self._pint_models_cache

    def _clear_pint_models_cache(self):
        """Clear the PINT models cache."""
        self._pint_models_cache = None

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
        """Make par files consistent across PTAs so that the certain model
        components (astrometry, spindown, binary, dispersion) are have
        consistent values between PTAs.

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

        # 2. Convert units if needed
        converted_parfiles = self._convert_units_if_needed(parfile_dicts)

        # 3. Make parameters consistent
        shared_parfiles = self._make_parameters_shared(converted_parfiles)

        # 4. Write consistent par files to output directory
        output_files = self._write_shared_parfiles(shared_parfiles)

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

    def _determine_parfile_units(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Determine effective units for all par files via raw active-line scan."""
        self.logger.info("Determining units for all par files")

        file_units: Dict[str, str] = {}
        parfile_contents: Dict[str, str] = {}
        for pta_name in self.file_data.keys():
            parfile_content = self._get_parfile_content(pta_name)
            file_units[pta_name] = self._effective_units_for_content(
                pta_name, parfile_content
            )
            parfile_contents[pta_name] = parfile_content
        return file_units, parfile_contents

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
        self, parfile_dicts: Dict[str, Dict]
    ) -> Dict[str, str]:
        """Normalize every retained par to explicit UNITS TDB before sharing.

        Unlike the previous early-return when all files already shared one unit
        system, every TCB (including tempo2-owned SI and no-UNITS tempo2
        defaults) is converted, and every TDB par is stamped if needed.
        """
        del parfile_dicts  # contents are re-read from source files
        self.logger.info("Normalizing all par files to explicit UNITS TDB")
        _, parfile_contents = self._determine_parfile_units()
        normalized: Dict[str, str] = {}
        for pta_name, content in parfile_contents.items():
            normalized[pta_name] = self._normalize_parfile_to_tdb(pta_name, content)
        return normalized

    def _get_default_time_units(self, pta_name: str) -> str:
        """Get the default time units for a PTA based on its timing package.

        Args:
            pta_name: Name of the PTA

        Returns:
            Default time units: "TDB" for PINT, "TCB" for Tempo2
        """
        timing_package = self.file_data[pta_name].get("timing_package", "pint")
        return "TDB" if timing_package == "pint" else "TCB"

    def _convert_mixed_units(
        self, file_units: Dict[str, str], parfile_contents: Dict[str, str]
    ) -> Dict[str, str]:
        """Convert TCB pars to TDB using the owning timing package.

        Retained for callers/tests that still pass precomputed unit maps; the
        primary path is :meth:`_convert_units_if_needed`.
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
            model = create_pint_model(parfile_content)

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
        normalized = (package or "pint").strip().lower()
        if normalized == "libstempo":
            return "tempo2"
        return normalized

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
        """Resolve the consistent NE_SW density (cm^-3) for this pulsar stack."""
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

    def _require_reference_conventions(
        self, reference_dict: Dict[str, List[str]]
    ) -> None:
        """Require reference conventions needed for multi-PTA parfile alignment."""
        if "EPHEM" not in reference_dict:
            raise ValueError(
                f"No alias from ['EPHEM'] found in reference PTA {self.reference_pta}"
            )
        if "CLOCK" not in reference_dict and "CLK" not in reference_dict:
            raise ValueError(
                "No alias from ['CLOCK', 'CLK'] found in reference PTA "
                f"{self.reference_pta}"
            )

    def _align_reference_conventions(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
    ) -> None:
        """Apply the reference EPHEM and clock convention to every PTA."""
        for parfile_dict in parfile_dicts.values():
            self._align_parameter(parfile_dict, reference_dict, "EPHEM", required=True)
            self._align_parameter(
                parfile_dict, reference_dict, ["CLOCK", "CLK"], required=True
            )

    def _apply_cross_engine_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        convention_states: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """Apply convention rules needed when PINT and tempo2 pulsars are mixed."""
        for pta_name, parfile_dict in parfile_dicts.items():
            actions: List[str] = []
            style = convention_states[pta_name]["style"]

            if style == "ecliptic":
                # PINT and tempo2 agree best for ecliptic pars under IERS2003.
                old_ecl = parfile_dict.get("ECL", ["(missing)"])[0]
                parfile_dict["ECL"] = ["IERS2003"]
                actions.append(f"set ECL=IERS2003 (was {old_ecl})")
            else:
                # Equatorial pars have no active ECL obliquity convention to align.
                if "ECL" in parfile_dict:
                    old_ecl = parfile_dict.pop("ECL")
                    actions.append(f"removed ECL ({old_ecl[0]})")
                self.logger.warning(f"[{pta_name}] {self._EQUATORIAL_WARNING}")
                actions.append("emitted equatorial warning")

            if "T2CMETHOD" in parfile_dict and self._is_t2cmethod_tempo(
                parfile_dict["T2CMETHOD"]
            ):
                # Active tempo2 TEMPO mode is a cross-engine convention mismatch.
                old_t2c = parfile_dict.pop("T2CMETHOD")
                actions.append(f"removed T2CMETHOD ({old_t2c[0]})")

            self.logger.info(
                f"PTA {pta_name}: consistent convention rules applied "
                f"(reason=cross_engine_agreement, style={style}); "
                f"actions: {', '.join(actions) if actions else 'none'}"
            )

    def _apply_pint_only_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
        convention_states: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """Align heterogeneous ecliptic conventions for multi-PTA PINT stacks."""
        ecliptic_ptas = [
            pta_name
            for pta_name, state in convention_states.items()
            if state["style"] == "ecliptic"
        ]
        ecliptic_ecl_values = {
            convention_states[pta_name]["ecl"] for pta_name in ecliptic_ptas
        }

        if len(ecliptic_ecl_values) <= 1:
            self.logger.info(
                "Consistent convention rules skipped "
                "(reason=homogeneous_ecl_single_engine_pint)"
            )
            return

        reference_ecl = self._parse_ecl_value(reference_dict)
        target_ecl = reference_ecl or "IERS2010"
        for pta_name in ecliptic_ptas:
            old_ecl = parfile_dicts[pta_name].get("ECL", ["(missing)"])[0]
            parfile_dicts[pta_name]["ECL"] = [target_ecl]
            self.logger.info(
                f"PTA {pta_name}: consistent convention rules applied "
                f"(reason=pint_only_ecl_heterogeneous); set ECL={target_ecl} (was {old_ecl})"
            )

    def _apply_tempo2_only_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
        convention_states: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """Align heterogeneous ecliptic and T2CMETHOD conventions for tempo2 stacks."""
        ecliptic_ptas = [
            pta_name
            for pta_name, state in convention_states.items()
            if state["style"] == "ecliptic"
        ]
        ecliptic_ecl_values = {
            convention_states[pta_name]["ecl"] for pta_name in ecliptic_ptas
        }
        if len(ecliptic_ecl_values) > 1:
            for pta_name in ecliptic_ptas:
                old_ecl = parfile_dicts[pta_name].get("ECL", ["(missing)"])[0]
                parfile_dicts[pta_name]["ECL"] = ["IERS2003"]
                self.logger.info(
                    f"PTA {pta_name}: consistent convention rules applied "
                    f"(reason=tempo2_only_ecl_heterogeneous); "
                    f"set ECL=IERS2003 (was {old_ecl})"
                )
        else:
            self.logger.info(
                "Consistent convention rules skipped "
                "(reason=homogeneous_ecl_single_engine_tempo2)"
            )

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

    def _apply_shared_convention_rules(
        self,
        parfile_dicts: Dict[str, Dict[str, List[str]]],
        reference_dict: Dict[str, List[str]],
    ) -> None:
        """Apply gated convention rules across consistent parfile dictionaries."""
        self._require_reference_conventions(reference_dict)
        convention_states = self._collect_convention_states(parfile_dicts)

        if len(parfile_dicts) == 1:
            self.logger.info("Consistent convention rules skipped (reason=single_pta)")
            return

        # Multi-PTA combinations share the reference ephemeris and clock.
        self._align_reference_conventions(parfile_dicts, reference_dict)

        normalized_packages = self._normalized_timing_packages()

        if self._is_cross_engine_mix(normalized_packages):
            self._apply_cross_engine_rules(parfile_dicts, convention_states)
            return

        if normalized_packages == {"pint"}:
            self._apply_pint_only_rules(
                parfile_dicts, reference_dict, convention_states
            )
            return

        if normalized_packages == {"tempo2"}:
            self._apply_tempo2_only_rules(
                parfile_dicts, reference_dict, convention_states
            )
            return

        self.logger.info(
            "Consistent convention rules skipped "
            f"(reason=unsupported_single_engine_packages:{sorted(normalized_packages)})"
        )

    def _make_parameters_shared(self, parfile_data: Dict[str, str]) -> Dict[str, str]:
        """Make parameters shared using reference PTA values.

        This function really is the workhorse of the MetaPulsar procedure to
        make par models consistent across PTAs. Method:

        - Start with parfiles that have been unit-converted (done)
        - Get all parameters from the reference PTA
        - Determine which model 'components' (astrometry, spindown, etc.) are
          being made shared, and find all parameters in the models
        - For each component, replace the parameters with the values of the
          reference PTA
        - For dispersion, remove DMX parameters
        - Optionally, add DM1 and DM2 parameters
        - Align explicit NE_SW when required for tempo2/PINT parity
        - Always align CLOCK and EPHEM parameters
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

        # Get reference PTA parameters
        reference_dict = parfile_dicts[self.reference_pta]

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

        self._align_ne_sw_convention(parfile_dicts, reference_dict)

        # Apply reference and engine-specific conventions after component updates.
        try:
            self._apply_shared_convention_rules(parfile_dicts, reference_dict)
        except ValueError as e:
            self.logger.error(f"Shared convention rules failed: {e}")
            raise RuntimeError(f"Shared convention rules failed: {e}") from e

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
        model = create_pint_model(parfile_dict)

        # Find DMX parameters from dispersion_dmx component
        dmx_params = []
        for comp in model.components.values():
            if hasattr(comp, "category") and comp.category == "dispersion_dmx":
                if hasattr(comp, "params"):
                    dmx_params.extend(comp.params)

        return dmx_params

    def _write_shared_parfiles(
        self, shared_parfiles: Dict[str, str]
    ) -> Dict[str, Path]:
        """Write consistent par files to output directory."""
        if self.output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="shared_parfiles_"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_files = {}

        for pta_name, parfile_content in shared_parfiles.items():
            output_filename = self._get_output_filename(pta_name)
            output_path = self.output_dir / output_filename

            with open(output_path, "w") as f:
                f.write(parfile_content)

            output_files[pta_name] = output_path
            self.logger.debug(f"Written consistent par file: {output_path}")

        return output_files

    def _get_output_filename(self, pta_name: str) -> str:
        """Generate output filename for shared par file."""
        if self.pulsar_name:
            return f"{self.pulsar_name}_shared_{pta_name}.par"
        else:
            return f"shared_{pta_name}.par"

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
            pint_models[pta_name] = create_pint_model(parfile_content)

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
            model = create_pint_model(parfile_content)

            if not check_component_available_in_model(model, component_type):
                return False
        return True

    def check_parameter_identifiable(self, pta_name: str, param_name: str) -> bool:
        """Check if parameter is identifiable in specific PINT model."""
        if pta_name not in self.file_data:
            return False

        parfile_content = self._get_parfile_content(pta_name)
        model = create_pint_model(parfile_content)
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
        """Get timing package for a specific PTA from file data."""
        return self.file_data[pta_name]["timing_package"]

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
