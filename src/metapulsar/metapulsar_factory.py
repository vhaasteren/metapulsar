"""Meta-Pulsar Factory for creating MetaPulsars by orchestrating PTA timing objects.

This module provides a factory class that creates MetaPulsars by discovering files,
building per-PTA timing objects, and wrapping them with metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from contextlib import nullcontext
from pathlib import Path
import shutil
import tempfile
import warnings
from urllib.parse import quote
from loguru import logger

# Import MetaPulsar and ParameterManager
from .metapulsar import MetaPulsar, normalize_combination_strategy
from .parameter_manager import AlignmentPolicy, ParameterManager
from .position_helpers import discover_pulsars_by_position

# Import PINT for model creation
try:
    from pint.models import get_model_and_toas
except ImportError:
    get_model_and_toas = None

# Import sandbox for robust libstempo usage
from .sandbox_tempo2 import tempopulsar
from .tim_file_analyzer import TimFileAnalyzer, TimMetadata
from .tim_canonical import (
    convert_jump_mjd_par_text,
    discover_effective_tim_mode,
    ensure_par_mode,
    inject_pulse_numbers,
    parse_jump_mjd_windows,
    write_canonical_tim,
)
from .pint_helpers import (
    Ell1hShapiroMode,
    PulseNumberMode,
    ensure_pint_track_minus_2,
    pulse_number_tracking_enabled,
    resolved_tim_for_pulse_numbers,
    temporary_par_with_track_minus_2,
    validate_pulse_number_mode,
    parameter_belongs_to_component_category,
)


def _par_content_has_dmx(par_content: str) -> bool:
    """Return True if a tempo2/PINT par string declares a DMX model."""
    from io import StringIO
    from pint.models.model_builder import parse_parfile

    return any(
        parameter_belongs_to_component_category(key, "dispersion_dmx")
        for key in parse_parfile(StringIO(par_content))
    )


_SINGLE_PTA_SHARED_DMX_WARNING = (
    "combination_strategy='shared' on a single-PTA MetaPulsar strips DMX from "
    "the timing model (dispersion sharing replaces DMX with DM/DM1/DM2). "
    "Single-PTA psrs that require DMX: use combination_strategy='per_pta'."
)


def _safe_pta_filename(pta_name: str) -> str:
    """Return an injective filesystem-safe stem for a PTA key."""
    return quote(pta_name, safe="._-")


# Default components for the shared combination strategy
DEFAULT_COMBINE_COMPONENTS: List[str] = [
    "astrometry",
    "spindown",
    "binary",
    "dispersion",
]


class MetaPulsarFactory:
    """Factory for creating MetaPulsars by orchestrating PTA timing-object creation.

    This class provides methods to discover files, build per-PTA timing objects,
    and wrap them in MetaPulsar objects with appropriate metadata.

    """

    def __init__(self):
        """Initialize the MetaPulsar factory.

        Note: File discovery should be handled separately using FileDiscovery.
        This factory only handles object creation from provided file paths.
        """
        self.logger = logger
        self._tim_analyzer = TimFileAnalyzer()
        # ParameterManager will be instantiated as needed in methods

    def _ensure_parfile_content(
        self, file_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Ensure parfile content is present in file data.

        Args:
            file_data: File data structure (may be missing par_content)

        Returns:
            Updated file data with par_content for all PTAs

        Raises:
            ValueError: If par file path is missing or file cannot be read
        """
        validated_file_data = {}

        for pta_name, files in file_data.items():
            validated_files = []

            for file_info in files:
                # Create a copy to avoid modifying original
                validated_file_info = file_info.copy()

                # Check if par_content is missing
                if "par_content" not in validated_file_info:
                    # Ensure par file path exists
                    if "par" not in validated_file_info:
                        raise ValueError(f"Missing 'par' file path for PTA {pta_name}")

                    par_path = validated_file_info["par"]
                    if isinstance(par_path, str):
                        par_path = Path(par_path)

                    # Read parfile content
                    try:
                        par_content = par_path.read_text(encoding="utf-8")
                        validated_file_info["par_content"] = par_content
                        self.logger.debug(
                            f"Read parfile content for {pta_name} from {par_path}"
                        )
                    except FileNotFoundError:
                        raise ValueError(f"Parfile not found: {par_path}")
                    except Exception as e:
                        raise ValueError(f"Failed to read parfile {par_path}: {e}")

                validated_files.append(validated_file_info)

            validated_file_data[pta_name] = validated_files

        return validated_file_data

    def _warn_single_pta_shared_dmx_strip(
        self,
        single_file_data: Dict[str, Dict[str, Any]],
        combine_components: List[str],
    ) -> None:
        """Warn when shared strategy will strip DMX from a single-PTA pulsar."""
        if len(single_file_data) != 1:
            return
        if "dispersion" not in combine_components:
            return
        file_info = next(iter(single_file_data.values()))
        par_content = file_info.get("par_content")
        if not par_content:
            par_path = file_info.get("par")
            if par_path is None:
                return
            try:
                par_content = Path(par_path).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except OSError:
                return
        if not _par_content_has_dmx(par_content):
            return
        warnings.warn(_SINGLE_PTA_SHARED_DMX_WARNING, UserWarning, stacklevel=3)
        self.logger.warning(_SINGLE_PTA_SHARED_DMX_WARNING)

    def _ensure_tim_metadata(
        self, file_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Ensure each file dict has tim_metadata (populate at factory boundary)."""
        enriched: Dict[str, List[Dict[str, Any]]] = {}
        for pta_name, files in file_data.items():
            enriched_files = []
            for file_info in files:
                updated = file_info.copy()
                if "tim_metadata" not in updated:
                    tim_path = updated.get("tim")
                    if tim_path is not None:
                        if isinstance(tim_path, str):
                            tim_path = Path(tim_path)
                        updated["tim_metadata"] = self._tim_analyzer.get_tim_metadata(
                            tim_path
                        )
                enriched_files.append(updated)
            enriched[pta_name] = enriched_files
        return enriched

    def create_metapulsar(
        self,
        file_data: Dict[str, List[Dict[str, Any]]],
        combination_strategy: str = "shared",
        reference_pta: str = None,
        combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
        add_dm_derivatives: bool = True,
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        parfile_output_dir: Path = None,
        timfile_output_dir: Path = None,
        use_pulse_numbers: str = "yes",
        clock_dir: Path | str | None = None,
        alignment_policy: AlignmentPolicy | None = None,
        convert_jump_mjd: bool = False,
        canonicalize_tim: bool = True,
    ) -> MetaPulsar:
        """Create MetaPulsar using specified combination strategy.

        Args:
            file_data: File data from FileDiscovery (should contain data for single pulsar only)
            combination_strategy: Strategy for combining PTAs:
                - "shared": shared timing-model params across PTAs (modifies par
                  files for consistency; default; ex-"consistent")
                - "per_pta": per-PTA timing-model params preserved (ex-"composite").
                The legacy "consistent"/"composite" spellings are accepted as
                deprecated aliases.
            reference_pta: PTA to use as reference (for the shared strategy). If None, uses first PTA in file_data.
            combine_components: List of components to share (for the shared strategy).
                Defaults to all components: ["astrometry", "spindown", "binary", "dispersion"]
            add_dm_derivatives: Whether to ensure DM1, DM2 are present in all par files (for the shared strategy)
            exclude_from_shared: Canonical timing-model parameter names to keep
                PTA-specific even when their component is in ``combine_components``.
                Defaults to ``("DM",)`` so each PTA keeps its own reference DM while
                shared dispersion still shares ``DM1``/``DM2``. Pass an empty list to
                merge all parameters in selected components.
            parfile_output_dir: Directory to save shared par files (for the shared strategy only).
                If None, par files are not saved to disk.
            timfile_output_dir: Directory to save the ``.tim`` files the engines
                actually consumed, as ``{pulsar}_{pta}.tim``. With
                ``canonicalize_tim=True`` (default) these are standalone Tempo2
                FORMAT 1 files (INCLUDEs flattened) carrying ``-pta``,
                ``-pta_dataset`` and ``-timing_package`` flags, plus ``-pn`` when
                ``use_pulse_numbers`` asks for it. If None, they are not saved.
            use_pulse_numbers: Pulse-number mode (string only; default ``"yes"``).
                Controls pulse numbers only; whether the release ``.tim`` is
                rewritten is ``canonicalize_tim``:

                - ``"no"``: ignore pulse numbers; no ``TRACK -2`` override (Tempo2).
                  Existing ``-pn`` flags are preserved but none are derived.
                - ``"yes"``: reuse complete ``-pn`` on all TOAs, else re-derive from
                  original coherent ``par`` + ``tim``; warn on mixed partial ``-pn``.
                - ``"reuse"``: same as ``"yes"`` when complete; warn and re-derive when
                  incomplete or missing ``-pn``.
                - ``"overwrite"``: always re-derive ``-pn`` from original ``par`` + ``tim``.

                Booleans are rejected. Map legacy ``True`` → ``"yes"``, ``False`` → ``"no"``.
            alignment_policy: :class:`~metapulsar.parameter_manager.AlignmentPolicy`
                controlling the multi-PTA common profile (``unsupported="strip"``
                by default, plus optional ``ephem``/``clock``/``bipm_version``/
                ``ne_sw`` pins, and gated binary-conversion knobs such as
                ``binary_fidelity_tolerance_factor``). Only valid for the
                ``"shared"`` strategy.
            convert_jump_mjd: If True, rewrite each engine-par ``JUMP MJD t1 t2 ...``
                line to ``JUMP -mjd_jump_pta {pta}_{k} ...`` using the same
                ``{pta}_{k}`` values stamped on the canonical tim. Default False
                (tim flags are still stamped when ``canonicalize_tim=True``).
                Requires ``canonicalize_tim=True``.
            canonicalize_tim: If True (default), rewrite every release ``.tim`` into
                a dual-engine-reloadable canonical artifact before load (INCLUDE
                flatten, ``TIME`` bake, safe TOA names, PTA / jump flags). If
                False, engines load the release ``.tim`` tree (plus optional ``-pn``
                derivation); cross-engine ``TIME``/``INCLUDE`` parity and
                ``-mjd_jump_pta`` stamping are not provided. Use as an escape
                hatch when canonicalization refuses a published release.

        Returns:
            MetaPulsar object

        Raises:
            ValueError: If no files found, multiple pulsars detected, or invalid parameters
            RuntimeError: If PTA timing-object creation fails
        """
        combination_strategy = normalize_combination_strategy(combination_strategy)
        self.logger.info(f"Creating MetaPulsar using {combination_strategy} strategy")
        pulse_mode = validate_pulse_number_mode(use_pulse_numbers)
        if convert_jump_mjd and not canonicalize_tim:
            raise ValueError(
                "convert_jump_mjd=True requires canonicalize_tim=True because "
                "JUMP -mjd_jump_pta flags are only stamped on the canonical .tim"
            )
        if alignment_policy is not None and combination_strategy != "shared":
            raise ValueError(
                "alignment_policy only applies to combination_strategy='shared'; "
                f"got {combination_strategy!r}. The per_pta strategy preserves "
                "each PTA's native deterministic model and performs no alignment."
            )

        # 1. Ensure parfile content and TIM metadata are loaded
        validated_data = self._ensure_parfile_content(file_data)
        validated_data = self._ensure_tim_metadata(validated_data)

        # 2. Validate all files belong to same pulsar (coordinate-based)
        self._validate_single_pulsar_data(validated_data)

        # 3. Apply reference PTA ordering if specified
        if reference_pta is not None and reference_pta in validated_data:
            validated_data = reorder_ptas_for_pulsar(validated_data, reference_pta)
        elif reference_pta is not None:
            # Invalid reference_pta - fall back to original ordering (first PTA)
            self.logger.warning(
                f"Reference PTA '{reference_pta}' not found in file data, using original ordering"
            )

        # 4. Get pulsar name for output filename generation
        pulsar_groups = discover_pulsars_by_position(validated_data)
        pulsar_name = list(pulsar_groups.keys())[0] if pulsar_groups else "unknown"

        # 5. Create MetaPulsar
        # Convert file data to single file per PTA format
        single_file_data = {}
        for pta_name, file_list in validated_data.items():
            if not file_list:
                raise ValueError(f"No files found for PTA {pta_name}")
            single_file_data[pta_name] = file_list[0]  # Take first file

        # Create output directory if parfile_output_dir is provided
        if parfile_output_dir:
            parfile_output_dir = Path(parfile_output_dir).resolve()
            parfile_output_dir.mkdir(parents=True, exist_ok=True)
        if timfile_output_dir:
            timfile_output_dir = Path(timfile_output_dir).resolve()
            timfile_output_dir.mkdir(parents=True, exist_ok=True)
        # The files back native timing engines and therefore must live as long
        # as the MetaPulsar. TemporaryDirectory cleans exception paths
        # immediately; ownership is transferred to the completed object below.
        pta_file_owner = tempfile.TemporaryDirectory(prefix="metapulsar_pta_files_")
        pta_file_dir = Path(pta_file_owner.name).resolve()

        # ParameterManager produces the par files the engines consume under both
        # strategies. Orbital-chart alignment is its first step either way; only
        # the shared strategy adds unit normalization and cross-PTA merging.
        parameter_manager = ParameterManager(
            file_data=single_file_data,
            combine_components=combine_components,
            add_dm_derivatives=add_dm_derivatives,
            output_dir=parfile_output_dir,
            pulsar_name=pulsar_name,
            exclude_from_shared=exclude_from_shared,
            alignment_policy=alignment_policy,
        )

        # Process par files based on strategy
        ell1h_shapiro: Ell1hShapiroMode = "full"
        if combination_strategy == "shared":
            # Mixed PINT+Tempo2 shared stacks must evaluate the same orthometric
            # Shapiro expression as tempo2 (see AlignmentPolicy docs).
            ell1h_shapiro = parameter_manager.ell1h_shapiro

            self._warn_single_pta_shared_dmx_strip(single_file_data, combine_components)
            engine_pars = parameter_manager.make_parfiles_shared()
            binary_conversion_report = parameter_manager.last_binary_conversion_report
        else:
            engine_pars = parameter_manager.engine_parfiles()
            binary_conversion_report = None
            if parfile_output_dir:
                # Writes file_dict["par_content"], which is never mutated, so an
                # "original" dump remains the data release's own bytes.
                self._write_original_parfiles(
                    single_file_data, parfile_output_dir, pulsar_name
                )

        if convert_jump_mjd:
            engine_pars, jump_changed = self._apply_jump_mjd_conversion(
                engine_pars=engine_pars,
                file_data=single_file_data,
                pta_file_dir=pta_file_dir,
            )
            if combination_strategy == "shared" and parfile_output_dir is not None:
                self._export_engine_pars(
                    engine_pars, parfile_output_dir, pulsar_name, only=jump_changed
                )

        # Release-tim MODE → engine par (before PN rewrite, which drops MODE).
        engine_pars, mode_changed = self._apply_tim_mode_transfer(
            engine_pars=engine_pars,
            file_data=single_file_data,
            pta_file_dir=pta_file_dir,
        )
        if combination_strategy == "shared" and parfile_output_dir is not None:
            self._export_engine_pars(
                engine_pars, parfile_output_dir, pulsar_name, only=mode_changed
            )

        file_pairs = {
            pta: (engine_pars[pta], single_file_data[pta]["tim"])
            for pta in single_file_data
            if pta in engine_pars
        }

        # Create PINT/Tempo2 objects from file pairs using file data
        created = self._create_pulsar_objects(
            file_pairs=file_pairs,
            file_data=single_file_data,
            use_pulse_numbers=pulse_mode,
            pta_file_dir=pta_file_dir,
            return_pta_files=True,
            ell1h_shapiro=ell1h_shapiro,
            canonicalize_tim=canonicalize_tim,
        )
        if isinstance(created, tuple) and len(created) == 2:
            pulsars, pta_files = created
        else:
            pulsars, pta_files = created, {}

        if timfile_output_dir:
            self._write_canonical_timfiles(pta_files, timfile_output_dir, pulsar_name)

        mp = MetaPulsar(
            pulsars=pulsars,
            combination_strategy=combination_strategy,
            combine_components=combine_components,
            add_dm_derivatives=add_dm_derivatives,
            exclude_from_shared=exclude_from_shared,
            pta_files=pta_files,
            clock_dir=clock_dir,
        )
        mp.binary_conversion_report = binary_conversion_report
        mp._pta_file_owner = pta_file_owner
        return mp

    def _validate_single_pulsar_data(
        self, file_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Validate that file_data contains files for only one pulsar.

        Args:
            file_data: File data to validate

        Raises:
            ValueError: If multiple pulsars detected or no valid files found
        """
        # Group files by pulsar using coordinate-based identification
        pulsar_groups = discover_pulsars_by_position(file_data)

        if not pulsar_groups:
            raise ValueError("No valid pulsar files found in file_data")

        if len(pulsar_groups) > 1:
            pulsar_names = list(pulsar_groups.keys())
            raise ValueError(
                f"Multiple pulsars detected in file_data: {pulsar_names}. "
                f"create_metapulsar() expects data for a single pulsar. "
                f"Use create_all_metapulsars() for multiple pulsars or "
                f"group_files_by_pulsar() to separate the data first."
            )

        # Log the single pulsar being processed
        pulsar_name = list(pulsar_groups.keys())[0]
        self.logger.info(f"Validated single pulsar data for: {pulsar_name}")

    def group_files_by_pulsar(
        self, file_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group file data by pulsar using J2000 position matching and catalog names.

        This utility function takes multi-pulsar file data and groups it by pulsar,
        making it suitable for creating individual MetaPulsars.

        Args:
            file_data: File data from FileDiscovery containing multiple pulsars

        Returns:
            Dictionary mapping pulsar names to their respective file data:
            {
                "J1857+0943": {
                    "epta_dr2": [file_dict1, file_dict2, ...],
                    "ppta_dr2": [file_dict3, file_dict4, ...]
                },
                "J1909-3744": {
                    "epta_dr2": [file_dict5, ...],
                    "ppta_dr2": [file_dict6, ...]
                }
            }

        Raises:
            ValueError: If no valid pulsar files found
        """
        self.logger.info(
            "Grouping files by pulsar using position-based catalog identification"
        )

        pulsar_groups = discover_pulsars_by_position(file_data)

        if not pulsar_groups:
            raise ValueError("No valid pulsar files found in file_data")

        self.logger.info(
            f"Found {len(pulsar_groups)} pulsars: {list(pulsar_groups.keys())}"
        )

        return pulsar_groups

    def _group_files_by_pulsar_with_ordering(
        self, file_data: Dict[str, List[Dict[str, Any]]], reference_pta: str = None
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group files by pulsar with reference PTA ordering.

        Args:
            file_data: File data from FileDiscovery (per data release)
            reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.

        Returns:
            Dictionary mapping pulsar names to ordered PTA data:
            {
                "J1857+0943": {
                    "epta_dr2": [...],  # First PTA = reference
                    "ppta_dr2": [...]
                }
            }
        """
        # First, group by pulsar using coordinate-based identification
        pulsar_groups = discover_pulsars_by_position(file_data)

        if not pulsar_groups:
            raise ValueError("No valid pulsar files found in file_data")

        # For each pulsar, order its PTAs
        ordered_pulsar_groups = {}

        for pulsar_name, pulsar_file_data in pulsar_groups.items():
            # Determine reference PTA for this pulsar
            ref_pta_timespan = self._find_best_reference_pta_by_timespan(
                pulsar_file_data
            )

            # Use specified reference PTA if available, otherwise by timespan
            ref_pta = (
                reference_pta if reference_pta in pulsar_file_data else ref_pta_timespan
            )

            # Order PTAs with reference first
            ordered_ptas = {ref_pta: pulsar_file_data[ref_pta]}
            ordered_ptas.update(
                {k: v for k, v in pulsar_file_data.items() if k != ref_pta}
            )
            ordered_pulsar_groups[pulsar_name] = ordered_ptas

        return ordered_pulsar_groups

    def _find_best_reference_pta_by_timespan(
        self, pulsar_file_data: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Find the PTA with longest timespan for a specific pulsar."""
        best_pta = None
        best_timespan = -1

        for pta_name, files in pulsar_file_data.items():
            if not files:
                continue

            # Get timespan for this PTA's files for this pulsar
            timespan = max(self._timespan_from_file_info(f) for f in files)

            if timespan > best_timespan:
                best_timespan = timespan
                best_pta = pta_name

        return best_pta or list(pulsar_file_data.keys())[0]

    @staticmethod
    def _timespan_from_file_info(file_info: Dict[str, Any]) -> float:
        meta = file_info.get("tim_metadata")
        if isinstance(meta, TimMetadata):
            return meta.timespan_days
        return 0.0

    @staticmethod
    def _toa_count_from_file_info(file_info: Dict[str, Any]) -> int:
        meta = file_info.get("tim_metadata")
        if isinstance(meta, TimMetadata):
            return meta.toa_count
        return 0

    @staticmethod
    def _pn_summary_from_files(files: List[Dict[str, Any]]) -> str:
        total = 0
        with_pn = 0
        without_pn = 0
        for file_info in files:
            meta = file_info.get("tim_metadata")
            if not isinstance(meta, TimMetadata):
                continue
            total += meta.toa_count
            with_pn += meta.pn_with_count
            without_pn += meta.pn_without_count
        if total == 0:
            return "pn=none (0/0)"
        if with_pn == 0:
            status = "none"
        elif without_pn == 0:
            status = "complete"
        else:
            status = "mixed"
        return f"pn={status} ({with_pn}/{total})"

    def create_all_metapulsars(
        self,
        file_data: Dict[str, List[Dict[str, Any]]],
        combination_strategy: str = "shared",
        reference_pta: str = None,
        combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
        add_dm_derivatives: bool = True,
        exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
        parfile_output_dir: Path = None,
        timfile_output_dir: Path = None,
        use_pulse_numbers: str = "yes",
        clock_dir: Path | str | None = None,
        alignment_policy: AlignmentPolicy | None = None,
        convert_jump_mjd: bool = False,
        canonicalize_tim: bool = True,
    ) -> Dict[str, MetaPulsar]:
        """Create MetaPulsars for all available pulsars using file data.

        Args:
            file_data: File data from FileDiscovery (per data release)
            combination_strategy: Strategy for combining PTAs
            reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.
            combine_components: List of components to share
            add_dm_derivatives: Whether to ensure DM1, DM2 are present
            exclude_from_shared: Canonical timing-model parameter names to keep
                PTA-specific even when their component is in ``combine_components``.
                Defaults to ``("DM",)``. Pass an empty list to merge all parameters
                in selected components.
            parfile_output_dir: Directory to save shared par files (for the shared strategy only).
                If None, par files are not saved to disk. Files are named per pulsar.
            timfile_output_dir: Directory to save the ``.tim`` files the engines
                consumed, named ``{pulsar}_{pta}.tim``. If None, they are not
                saved to disk.
            alignment_policy: Alignment policy forwarded to each
                ``create_metapulsar`` call (``"shared"`` strategy only).
            convert_jump_mjd: If True, rewrite each engine-par ``JUMP MJD t1 t2 ...``
                line to ``JUMP -mjd_jump_pta {pta}_{k} ...`` using the same
                ``{pta}_{k}`` values stamped on the canonical tim. Default False.
                Requires ``canonicalize_tim=True``.
            canonicalize_tim: Forwarded to each ``create_metapulsar`` call.
                Default True; see that method for the off-path contract.

        Returns:
            Dictionary mapping pulsar names to MetaPulsar objects
        """
        combination_strategy = normalize_combination_strategy(combination_strategy)
        if convert_jump_mjd and not canonicalize_tim:
            raise ValueError(
                "convert_jump_mjd=True requires canonicalize_tim=True because "
                "JUMP -mjd_jump_pta flags are only stamped on the canonical .tim"
            )
        if alignment_policy is not None and combination_strategy != "shared":
            raise ValueError(
                "alignment_policy only applies to combination_strategy='shared'; "
                f"got {combination_strategy!r}. The per_pta strategy preserves "
                "each PTA's native deterministic model and performs no alignment."
            )
        # 1. Ensure parfile content and TIM metadata are loaded
        validated_data = self._ensure_parfile_content(file_data)
        validated_data = self._ensure_tim_metadata(validated_data)

        # 2. Group files by pulsar with reference PTA ordering
        pulsar_groups = self._group_files_by_pulsar_with_ordering(
            validated_data, reference_pta
        )

        metapulsars = {}

        self.logger.info(f"Creating MetaPulsars for {len(pulsar_groups)} pulsars")
        pulse_mode = validate_pulse_number_mode(use_pulse_numbers)

        for pulsar_name, pulsar_file_data in pulsar_groups.items():
            try:
                # Get reference PTA (first in this pulsar's dictionary)
                reference_pta_for_pulsar = list(pulsar_file_data.keys())[0]
                self.logger.info(
                    f"Pulsar {pulsar_name}: Using reference PTA {reference_pta_for_pulsar}"
                )

                # Create MetaPulsar for this pulsar
                metapulsar = self.create_metapulsar(
                    file_data=pulsar_file_data,
                    combination_strategy=combination_strategy,
                    reference_pta=reference_pta_for_pulsar,
                    combine_components=combine_components,
                    add_dm_derivatives=add_dm_derivatives,
                    exclude_from_shared=exclude_from_shared,
                    parfile_output_dir=parfile_output_dir,
                    timfile_output_dir=timfile_output_dir,
                    use_pulse_numbers=pulse_mode,
                    clock_dir=clock_dir,
                    alignment_policy=alignment_policy,
                    convert_jump_mjd=convert_jump_mjd,
                    canonicalize_tim=canonicalize_tim,
                )

                # Canonical name is automatically calculated from pulsar data
                metapulsars[metapulsar.name] = metapulsar

            except Exception as e:
                self.logger.warning(
                    f"Failed to create MetaPulsar for {pulsar_name}: {e}"
                )

        self.logger.info(f"Successfully created {len(metapulsars)} MetaPulsars")
        return metapulsars

    def _get_display_name_for_pulsar(
        self, pulsar_name: str, pulsar_file_data: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Return catalog display name (group key from position-based discovery)."""
        return pulsar_name

    def pta_summary(self, file_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Display summary statistics for all pulsars and PTAs in the file data.

        Performs coordinate-based discovery to group files by pulsar, then displays
        timespan statistics for each pulsar and PTA combination.

        Args:
            file_data: File data from FileDiscovery (per data release)
        """
        import warnings

        # Suppress PINT warnings and loguru output for clean summary display
        import sys
        from loguru import logger as loguru_logger

        # Store original loguru configuration (for potential future use)

        try:
            # Remove all existing loguru handlers
            loguru_logger.remove()

            # Add a new handler that only shows CRITICAL messages
            loguru_logger.add(lambda msg: None, level="CRITICAL")

            # Also suppress Python warnings
            warnings.filterwarnings("ignore")

            with self.logger.catch():
                print("Quickly processing PTA files...")

                # Note: file_data contains file paths per PTA, but pulsars are not yet matched between PTAs.
                # The coordinate-based discovery groups files by pulsar using coordinate matching, not name matching.
                # 1. Ensure parfile content and TIM metadata are loaded
                validated_data = self._ensure_parfile_content(file_data)
                validated_data = self._ensure_tim_metadata(validated_data)

                # 2. Group files by pulsar with reference PTA ordering
                pulsar_groups = self._group_files_by_pulsar_with_ordering(
                    validated_data
                )

                if not pulsar_groups:
                    print("No valid pulsar files found in file_data")
                    return

                print(f"Found {len(pulsar_groups)} pulsars:")
                print()

                for pulsar_name, pulsar_file_data in pulsar_groups.items():
                    # Get display name using B-name preference logic
                    display_name = self._get_display_name_for_pulsar(
                        pulsar_name, pulsar_file_data
                    )
                    print(display_name)

                    # Calculate timespans and TOA counts for each PTA
                    pta_timespans = []
                    for pta_name, files in pulsar_file_data.items():
                        if not files:
                            continue

                        # Get timespan, TOA count, and pn coverage for this PTA
                        timespan_days = max(
                            self._timespan_from_file_info(f) for f in files
                        )
                        timespan_years = timespan_days / 365.25
                        toa_count = sum(
                            self._toa_count_from_file_info(f) for f in files
                        )
                        pn_summary = self._pn_summary_from_files(files)
                        pta_timespans.append(
                            (
                                pta_name,
                                timespan_days,
                                timespan_years,
                                toa_count,
                                pn_summary,
                            )
                        )

                    # Sort by timespan (longest first)
                    pta_timespans.sort(key=lambda x: x[1], reverse=True)

                    # Display PTAs with reference indicator
                    reference_pta = list(pulsar_file_data.keys())[
                        0
                    ]  # First in original ordering

                    for (
                        pta_name,
                        timespan_days,
                        timespan_years,
                        toa_count,
                        pn_summary,
                    ) in pta_timespans:
                        reference_indicator = (
                            " -- Reference PTA" if pta_name == reference_pta else ""
                        )
                        print(
                            f"- {pta_name}: {timespan_days:.0f} days "
                            f"({timespan_years:.1f} years, {toa_count} TOAs, "
                            f"{pn_summary}){reference_indicator}"
                        )

                    print()

        finally:
            # Restore original loguru configuration
            loguru_logger.remove()
            # Re-add default handler
            loguru_logger.add(sys.stderr, level="DEBUG")

    def _create_pulsar_objects(
        self,
        file_pairs: Dict[str, Tuple[Path, Path]],
        file_data: Dict[str, Dict[str, Any]],
        use_pulse_numbers: PulseNumberMode = "yes",
        pta_file_dir: Path | None = None,
        return_pta_files: bool = False,
        ell1h_shapiro: Ell1hShapiroMode = "full",
        canonicalize_tim: bool = True,
    ) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Create PINT/Tempo2 objects from file pairs using file data.

        Args:
            file_pairs: Dictionary mapping PTA names to (parfile, timfile) tuples
            file_data: Dictionary mapping PTA names to file dictionaries
                      Contains timing_package info from FileDiscovery
            use_pulse_numbers: Pulse-number mode (``no``, ``yes``, ``reuse``, ``overwrite``)
            ell1h_shapiro: ELL1H orthometric Shapiro convention for PINT loads.
                ``"absorbed"`` on mixed-engine shared stacks, ``"full"``
                (PINT's default) otherwise.
            canonicalize_tim: When True (default), rewrite each release ``.tim``
                via :func:`~metapulsar.tim_canonical.write_canonical_tim` before
                load. When False, load the release tree (plus optional ``-pn``
                derivation) without TIME bake / flag stamps / name rewrite.

        Returns:
            Dictionary mapping PTA names to PINT/Tempo2 objects
        """
        pulsar_objects = {}
        pta_files: Dict[str, Dict[str, Any]] = {}
        track_pn = pulse_number_tracking_enabled(use_pulse_numbers)
        if pta_file_dir is None:
            pta_file_dir = Path(
                tempfile.mkdtemp(prefix="metapulsar_pta_files_")
            ).resolve()
        pta_file_dir.mkdir(parents=True, exist_ok=True)
        if not canonicalize_tim:
            self.logger.warning(
                "canonicalize_tim=False: engines load release .tim files without "
                "TIME bake, safe TOA-name rewrite, or -mjd_jump_pta stamps. "
                "Cross-engine INCLUDE/TIME parity is not guaranteed."
            )

        for pta_name, (parfile, timfile) in file_pairs.items():
            # Get timing package info from file data
            timing_package = file_data[pta_name]["timing_package"]
            original_par_text = file_data[pta_name]["par_content"]
            parfile = Path(parfile)
            timfile = Path(timfile)
            derive_backend: Literal["pint", "tempo2"] = (
                "pint" if timing_package == "pint" else "tempo2"
            )
            engine_tim = pta_file_dir / f"{_safe_pta_filename(pta_name)}.tim"

            try:
                if canonicalize_tim:
                    canonical = write_canonical_tim(
                        timfile,
                        pta_name=pta_name,
                        timing_package=timing_package,
                        out_path=engine_tim,
                        par_text=original_par_text,
                    )
                    # Derive only after canonicalization so both backends read
                    # the same standalone, PINT-safe TOA layout. Keep the
                    # canonical artifact and inject only the derived -pn flags.
                    with resolved_tim_for_pulse_numbers(
                        use_pulse_numbers,
                        original_par_text,
                        engine_tim,
                        derive_backend=derive_backend,
                        tim_metadata=canonical.tim_metadata,
                    ) as resolved_tim:
                        resolved_path = Path(resolved_tim)
                        if resolved_path != engine_tim:
                            inject_pulse_numbers(engine_tim, derived_tim=resolved_path)
                else:
                    # Escape hatch: no flatten / TIME bake / flag stamps.
                    # PN derivation (if any) writes a temp; copy the resolved
                    # file into the session dir before the context closes.
                    with resolved_tim_for_pulse_numbers(
                        use_pulse_numbers,
                        original_par_text,
                        timfile,
                        derive_backend=derive_backend,
                        tim_metadata=file_data[pta_name].get("tim_metadata"),
                    ) as resolved_tim:
                        shutil.copy2(Path(resolved_tim), engine_tim)

                if timing_package == "pint":
                    if get_model_and_toas is None:
                        raise RuntimeError("PINT not available for PINT creation")

                    model, toas = get_model_and_toas(
                        str(parfile),
                        str(engine_tim),
                        planets=True,
                        allow_T2=True,
                        ell1h_shapiro=ell1h_shapiro,
                    )
                    if track_pn:
                        ensure_pint_track_minus_2(model)
                    pulsar_objects[pta_name] = (model, toas)
                    if return_pta_files:
                        pta_files[pta_name] = self._retain_pta_files(
                            pta_name=pta_name,
                            timing_package=timing_package,
                            par_path=parfile,
                            tim_path=engine_tim,
                            pta_file_dir=pta_file_dir,
                        )

                else:  # tempo2
                    par_context = (
                        temporary_par_with_track_minus_2(
                            parfile.read_text(encoding="utf-8")
                        )
                        if track_pn
                        else nullcontext(parfile)
                    )
                    # The TRACK -2 par is temporary, so retain it before the
                    # context closes.
                    with par_context as par_for_tempo2:
                        t2_psr = tempopulsar(
                            parfile=str(par_for_tempo2),
                            timfile=str(engine_tim),
                            dofit=False,
                        )
                        if return_pta_files:
                            pta_files[pta_name] = self._retain_pta_files(
                                pta_name=pta_name,
                                timing_package=timing_package,
                                par_path=Path(par_for_tempo2),
                                tim_path=engine_tim,
                                pta_file_dir=pta_file_dir,
                            )
                    pulsar_objects[pta_name] = t2_psr

                self.logger.debug(f"Created {timing_package} object for {pta_name}")

            except Exception as e:
                self.logger.error(f"Failed to create pulsar for {pta_name}: {e}")
                raise RuntimeError(f"Failed to create pulsar for {pta_name}: {e}")

        if return_pta_files:
            return pulsar_objects, pta_files
        return pulsar_objects

    def _retain_pta_files(
        self,
        *,
        pta_name: str,
        timing_package: str,
        par_path: Path,
        tim_path: Path,
        pta_file_dir: Path,
    ) -> Dict[str, Any]:
        """Retain the engine's par next to its session ``.tim``.

        With ``canonicalize_tim=True`` the tim is written straight into
        ``pta_file_dir`` by
        :func:`~metapulsar.tim_canonical.write_canonical_tim`. With
        ``canonicalize_tim=False`` the release (or PN-derived) tim is copied
        there before load. Either way there is a single session file rather
        than a second copy that could drift from it.
        """
        par_src = Path(par_path).resolve()
        tim_src = Path(tim_path).resolve()
        if not par_src.is_file():
            raise FileNotFoundError(f"Session par file does not exist: {par_src}")
        if not tim_src.is_file():
            raise FileNotFoundError(f"Session tim file does not exist: {tim_src}")
        pta_file_dir.mkdir(parents=True, exist_ok=True)
        par_dst = pta_file_dir / f"{_safe_pta_filename(pta_name)}.par"
        shutil.copy2(par_src, par_dst)
        return {
            "par_path": par_dst,
            "tim_path": tim_src,
            "timing_package": timing_package,
        }

    def _write_canonical_timfiles(
        self,
        pta_files: Dict[str, Dict[str, Any]],
        timfile_output_dir: Path,
        pulsar_name: str,
    ) -> None:
        """Export the exact engine-consumed tim files for reuse."""
        for pta_name, files in pta_files.items():
            src = Path(files["tim_path"])
            dst = (
                timfile_output_dir / f"{pulsar_name}_{_safe_pta_filename(pta_name)}.tim"
            )
            shutil.copy2(src, dst)
            self.logger.debug(f"Written engine tim file: {dst}")

    def _create_parfile_dicts_from_files(
        self, parfile_files: Dict[str, Path]
    ) -> Dict[str, Dict]:
        """Create parfile dictionaries from parfile files."""
        from .pint_helpers import create_pint_model

        parfile_dicts = {}
        for pta_name, parfile_path in parfile_files.items():
            with open(parfile_path, "r") as f:
                parfile_content = f.read()

            model = create_pint_model(parfile_content)
            parfile_dicts[pta_name] = model.get_params_dict()

        return parfile_dicts

    def _create_raw_pulsars(
        self,
        file_pairs: Dict[str, Tuple[Path, Path]],
        pta_data_releases: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Create raw PINT/Tempo2 objects from file pairs.

        Args:
            file_pairs: Dictionary mapping PTA names to (parfile, timfile) tuples
            pta_data_releases: Dictionary of PTA data releases

        Returns:
            Dictionary mapping PTA names to raw PINT/Tempo2 objects

        Raises:
            RuntimeError: If raw pulsar creation fails
        """
        raw_pulsars = {}

        for pta_name, (parfile, timfile) in file_pairs.items():
            data_release = pta_data_releases[pta_name]

            try:
                if data_release["timing_package"] == "pint":
                    if get_model_and_toas is None:
                        raise RuntimeError("PINT not available for raw PINT creation")

                    model, toas = get_model_and_toas(
                        str(parfile), str(timfile), planets=True, allow_T2=True
                    )
                    raw_pulsars[pta_name] = (model, toas)

                else:  # tempo2
                    t2_psr = tempopulsar(
                        parfile=str(parfile), timfile=str(timfile), dofit=False
                    )
                    raw_pulsars[pta_name] = t2_psr

                self.logger.debug(
                    f"Created raw {data_release['timing_package']} object for {pta_name}"
                )

            except Exception as e:
                self.logger.error(f"Failed to create raw pulsar for {pta_name}: {e}")
                raise RuntimeError(f"Failed to create raw pulsar for {pta_name}: {e}")

        return raw_pulsars

    def _write_original_parfiles(
        self,
        single_file_data: Dict[str, Dict[str, Any]],
        parfile_output_dir: Path,
        pulsar_name: str,
    ) -> None:
        """Write original par files to output directory for the per_pta strategy.

        Args:
            single_file_data: Single file per PTA data
            parfile_output_dir: Directory to write par files
            pulsar_name: Name of the pulsar for filename generation
        """
        for pta_name, file_dict in single_file_data.items():
            if "par_content" in file_dict:
                # Write original par content
                output_filename = f"{pulsar_name}_original_{pta_name}.par"
                output_path = parfile_output_dir / output_filename

                with open(output_path, "w") as f:
                    f.write(file_dict["par_content"])

                self.logger.debug(f"Written original par file: {output_path}")
            else:
                self.logger.warning(
                    f"No par_content found for {pta_name}, skipping original par file write"
                )

    def _apply_jump_mjd_conversion(
        self,
        *,
        engine_pars: Dict[str, Path],
        file_data: Dict[str, Dict[str, Any]],
        pta_file_dir: Path,
    ) -> Tuple[Dict[str, Path], Set[str]]:
        """Rewrite JUMP MJD in engine pars to flagged JUMP; never mutate release paths."""
        updated = dict(engine_pars)
        changed: Set[str] = set()
        pta_file_dir.mkdir(parents=True, exist_ok=True)
        for pta_name, engine_path in engine_pars.items():
            release_windows = parse_jump_mjd_windows(file_data[pta_name]["par_content"])
            if not release_windows:
                continue
            engine_text = Path(engine_path).read_text(encoding="utf-8")
            new_text = convert_jump_mjd_par_text(
                engine_text,
                pta_name=pta_name,
                release_windows=release_windows,
            )
            if new_text == engine_text:
                continue
            out_path = pta_file_dir / f"{_safe_pta_filename(pta_name)}.jumpmjd.par"
            out_path.write_text(new_text, encoding="utf-8")
            updated[pta_name] = out_path
            changed.add(pta_name)
            self.logger.debug(
                f"Converted JUMP MJD to -mjd_jump_pta for {pta_name}: {out_path}"
            )
        return updated, changed

    def _apply_tim_mode_transfer(
        self,
        *,
        engine_pars: Dict[str, Path],
        file_data: Dict[str, Dict[str, Any]],
        pta_file_dir: Path,
    ) -> Tuple[Dict[str, Path], Set[str]]:
        """Move release-tim MODE onto engine pars; never mutate release paths."""
        updated = dict(engine_pars)
        changed: Set[str] = set()
        pta_file_dir.mkdir(parents=True, exist_ok=True)
        for pta_name, engine_path in engine_pars.items():
            package = (
                "pint" if file_data[pta_name]["timing_package"] == "pint" else "tempo2"
            )
            mode = discover_effective_tim_mode(
                Path(file_data[pta_name]["tim"]), timing_package=package
            )
            if mode is None:
                continue  # no tim override: keep the par's own mode
            engine_text = Path(engine_path).read_text(encoding="utf-8")
            new_text = ensure_par_mode(engine_text, mode)
            if new_text == engine_text:
                continue
            out_path = pta_file_dir / f"{_safe_pta_filename(pta_name)}.mode.par"
            out_path.write_text(new_text, encoding="utf-8")
            updated[pta_name] = out_path
            changed.add(pta_name)
            self.logger.debug(
                f"Transferred release-tim MODE {mode} onto engine par for "
                f"{pta_name}: {out_path}"
            )
        return updated, changed

    def _export_engine_pars(
        self,
        engine_pars: Dict[str, Path],
        parfile_output_dir: Path,
        pulsar_name: str,
        *,
        only: Optional[Set[str]] = None,
    ) -> None:
        """Copy engine pars over the shared exports, skipping no-op self-copies."""
        for pta_name, engine_path in engine_pars.items():
            if only is not None and pta_name not in only:
                continue
            src = Path(engine_path).resolve()
            # Match ParameterManager._get_output_filename(..., tag="shared").
            dst = (
                parfile_output_dir / f"{pulsar_name}_shared_{pta_name}.par"
            ).resolve()
            if src == dst:
                continue
            if dst.exists() and src.samefile(dst):
                continue  # hard links resolve to distinct paths
            shutil.copy2(src, dst)
            self.logger.debug(f"Exported engine par for {pta_name}: {dst}")


def reorder_ptas_for_pulsar(
    pulsar_file_data: Dict[str, List[Dict[str, Any]]], reference_pta: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Reorder PTAs for a specific pulsar to put specified PTA first as reference.

    Args:
        pulsar_file_data: PTA data for a specific pulsar
        reference_pta: PTA name to use as reference (will be first in dict)

    Returns:
        Reordered pulsar data with reference_pta first
    """
    if reference_pta not in pulsar_file_data:
        raise ValueError(f"Reference PTA '{reference_pta}' not found in pulsar data")

    ordered = {reference_pta: pulsar_file_data[reference_pta]}
    ordered.update({k: v for k, v in pulsar_file_data.items() if k != reference_pta})
    return ordered


# Convenience functions for user-facing API
def create_metapulsar(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "shared",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
    parfile_output_dir: Path = None,
    timfile_output_dir: Path = None,
    use_pulse_numbers: str = "yes",
    clock_dir: Path | str | None = None,
    alignment_policy: AlignmentPolicy | None = None,
    convert_jump_mjd: bool = False,
    canonicalize_tim: bool = True,
) -> MetaPulsar:
    """Create MetaPulsar using specified combination strategy.

    Args:
        file_data: File data from FileDiscovery (should contain data for single pulsar only)
        combination_strategy: Strategy for combining PTAs:
            - "shared": shared timing-model params across PTAs (modifies par files
              for consistency; default; ex-"consistent")
            - "per_pta": per-PTA timing-model params preserved (ex-"composite").
            The legacy "consistent"/"composite" spellings are accepted as
            deprecated aliases.
        reference_pta: PTA to use as reference (for the shared strategy). If None, uses first PTA in file_data.
        combine_components: List of components to share (for the shared strategy).
            Defaults to all components: ["astrometry", "spindown", "binary", "dispersion"]
        add_dm_derivatives: Whether to ensure DM1, DM2 are present in all par files (for the shared strategy)
        exclude_from_shared: Canonical timing-model parameter names to keep
            PTA-specific even when their component is in ``combine_components``.
            Defaults to ``("DM",)``. Pass an empty list to merge all parameters
            in selected components.
        parfile_output_dir: Directory to save shared par files (for the shared strategy only).
            If None, par files are not saved to disk.
        timfile_output_dir: Directory to save the ``.tim`` files the engines
            consumed, as ``{pulsar}_{pta}.tim``. If None, they are not saved to disk.
        use_pulse_numbers: Pulse-number mode: ``"no"``, ``"yes"`` (default), ``"reuse"``,
            or ``"overwrite"``. See ``MetaPulsarFactory.create_metapulsar`` for semantics.
        alignment_policy: :class:`~metapulsar.parameter_manager.AlignmentPolicy`
            controlling the multi-PTA common profile (including gated binary
            conversion knobs such as ``binary_fidelity_tolerance_factor``).
            ``None`` means ``AlignmentPolicy()``. Passing a policy with
            ``"per_pta"`` raises ``ValueError``.
        convert_jump_mjd: If True, rewrite each engine-par ``JUMP MJD t1 t2 ...``
            line to ``JUMP -mjd_jump_pta {pta}_{k} ...`` using the same
            ``{pta}_{k}`` values stamped on the canonical tim. Default False.
            Requires ``canonicalize_tim=True``.
        canonicalize_tim: If True (default), rewrite every release ``.tim`` into
            a dual-engine-reloadable canonical artifact before load. If False,
            engines load the release ``.tim`` tree (escape hatch; see
            ``MetaPulsarFactory.create_metapulsar``).

    Returns:
        MetaPulsar object

    Raises:
        ValueError: If no files found, multiple pulsars detected, or invalid parameters
        RuntimeError: If PTA timing-object creation fails
    """
    factory = MetaPulsarFactory()
    return factory.create_metapulsar(
        file_data=file_data,
        combination_strategy=combination_strategy,
        reference_pta=reference_pta,
        combine_components=combine_components,
        add_dm_derivatives=add_dm_derivatives,
        exclude_from_shared=exclude_from_shared,
        parfile_output_dir=parfile_output_dir,
        timfile_output_dir=timfile_output_dir,
        use_pulse_numbers=use_pulse_numbers,
        clock_dir=clock_dir,
        alignment_policy=alignment_policy,
        convert_jump_mjd=convert_jump_mjd,
        canonicalize_tim=canonicalize_tim,
    )


def create_all_metapulsars(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "shared",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    exclude_from_shared: List[str] | tuple[str, ...] = ("DM",),
    parfile_output_dir: Path = None,
    timfile_output_dir: Path = None,
    use_pulse_numbers: str = "yes",
    clock_dir: Path | str | None = None,
    alignment_policy: AlignmentPolicy | None = None,
    convert_jump_mjd: bool = False,
    canonicalize_tim: bool = True,
) -> Dict[str, MetaPulsar]:
    """Create MetaPulsars for all available pulsars using file data.

    Args:
        file_data: File data from FileDiscovery (per data release)
        combination_strategy: Strategy for combining PTAs
        reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.
        combine_components: List of components to share
        add_dm_derivatives: Whether to ensure DM1, DM2 are present
        exclude_from_shared: Canonical timing-model parameter names to keep
            PTA-specific even when their component is in ``combine_components``.
            Defaults to ``("DM",)``. Pass an empty list to merge all parameters
            in selected components.
        parfile_output_dir: Directory to save shared par files (for the shared strategy only).
            If None, par files are not saved to disk. Files are named per pulsar.
        timfile_output_dir: Directory to save the ``.tim`` files the engines
            consumed, as ``{pulsar}_{pta}.tim``. If None, they are not saved to disk.
        use_pulse_numbers: Pulse-number mode passed to each ``create_metapulsar`` call
            (``"no"``, ``"yes"``, ``"reuse"``, or ``"overwrite"``; default ``"yes"``).
        alignment_policy: Alignment policy forwarded to each ``create_metapulsar``
            call (``"shared"`` strategy only).
        convert_jump_mjd: If True, rewrite each engine-par ``JUMP MJD t1 t2 ...``
            line to ``JUMP -mjd_jump_pta {pta}_{k} ...`` using the same
            ``{pta}_{k}`` values stamped on the canonical tim. Default False.
            Requires ``canonicalize_tim=True``.
        canonicalize_tim: Forwarded to each ``create_metapulsar`` call. Default True.

    Returns:
        Dictionary mapping pulsar names to MetaPulsar objects
    """
    factory = MetaPulsarFactory()
    return factory.create_all_metapulsars(
        file_data=file_data,
        combination_strategy=combination_strategy,
        reference_pta=reference_pta,
        combine_components=combine_components,
        add_dm_derivatives=add_dm_derivatives,
        exclude_from_shared=exclude_from_shared,
        parfile_output_dir=parfile_output_dir,
        timfile_output_dir=timfile_output_dir,
        use_pulse_numbers=use_pulse_numbers,
        clock_dir=clock_dir,
        alignment_policy=alignment_policy,
        convert_jump_mjd=convert_jump_mjd,
        canonicalize_tim=canonicalize_tim,
    )


def pta_summary(file_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Display summary statistics for all pulsars and PTAs in the file data.

    Performs coordinate-based discovery to group files by pulsar, then displays
    timespan statistics for each pulsar and PTA combination.

    Args:
        file_data: File data from FileDiscovery (per data release)
    """
    factory = MetaPulsarFactory()
    factory.pta_summary(file_data)
