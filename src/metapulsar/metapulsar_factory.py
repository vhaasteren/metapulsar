"""Meta-Pulsar Factory for creating MetaPulsars by orchestrating Enterprise Pulsar creation.

This module provides a factory class that creates MetaPulsars by discovering files,
creating Enterprise Pulsars, and wrapping them with metadata.
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path
import shutil
import tempfile
from loguru import logger

# Import Enterprise Pulsar classes
try:
    from enterprise.pulsar import PintPulsar, Tempo2Pulsar
except ImportError:
    PintPulsar = None
    Tempo2Pulsar = None

# Import MetaPulsar and ParameterManager
from .metapulsar import MetaPulsar, normalize_combination_strategy
from .parameter_manager import ParameterManager
from .position_helpers import discover_pulsars_by_coordinates_optimized

# Import PINT for model creation
try:
    from pint.models import get_model_and_toas
except ImportError:
    get_model_and_toas = None

# Import sandbox for robust libstempo usage
from .sandbox_tempo2 import tempopulsar
from .tim_file_analyzer import TimFileAnalyzer, TimMetadata
from .pint_helpers import (
    PulseNumberMode,
    ensure_pint_track_minus_2,
    pulse_number_tracking_enabled,
    resolved_tim_for_pulse_numbers,
    temporary_par_with_track_minus_2,
    validate_pulse_number_mode,
)


def _collect_included_tim_rel_paths(tim_path: Path) -> frozenset[Path]:
    """Return INCLUDE'd .tim paths relative to the root tim's parent directory."""
    root = Path(tim_path).resolve()
    root_dir = root.parent
    seen: set[Path] = set()
    included: set[Path] = set()

    def walk(current: Path) -> None:
        current = current.resolve()
        if current in seen:
            return
        seen.add(current)
        try:
            lines = current.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.upper().startswith("C ")
            ):
                continue
            tokens = stripped.split()
            if len(tokens) < 2 or tokens[0].upper() != "INCLUDE":
                continue
            inc_abs = (current.parent / tokens[1]).resolve()
            if not inc_abs.is_file():
                continue
            try:
                included.add(inc_abs.relative_to(root_dir))
            except ValueError:
                included.add(Path(tokens[1]))
            walk(inc_abs)

    walk(root)
    return frozenset(included)


# Default components for the shared combination strategy
DEFAULT_COMBINE_COMPONENTS: List[str] = [
    "astrometry",
    "spindown",
    "binary",
    "dispersion",
]


class MetaPulsarFactory:
    """Factory for creating MetaPulsars by orchestrating Enterprise Pulsar creation.

    This class provides methods to discover files, create Enterprise Pulsars,
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
        parfile_output_dir: Path = None,
        use_pulse_numbers: str = "yes",
        clock_dir: Path | str | None = None,
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
            parfile_output_dir: Directory to save shared par files (for the shared strategy only).
                If None, par files are not saved to disk.
            use_pulse_numbers: Pulse-number mode (string only; default ``"yes"``):

                - ``"no"``: ignore pulse numbers; no ``TRACK -2`` override (Tempo2).
                - ``"yes"``: reuse complete ``-pn`` on all TOAs, else re-derive from
                  original coherent ``par`` + ``tim``; warn on mixed partial ``-pn``.
                - ``"reuse"``: same as ``"yes"`` when complete; warn and re-derive when
                  incomplete or missing ``-pn``.
                - ``"overwrite"``: always re-derive ``-pn`` from original ``par`` + ``tim``.

                Booleans are rejected. Map legacy ``True`` → ``"yes"``, ``False`` → ``"no"``.

        Returns:
            MetaPulsar object

        Raises:
            ValueError: If no files found, multiple pulsars detected, or invalid parameters
            RuntimeError: If Enterprise Pulsar creation fails
        """
        combination_strategy = normalize_combination_strategy(combination_strategy)
        self.logger.info(f"Creating MetaPulsar using {combination_strategy} strategy")
        pulse_mode = validate_pulse_number_mode(use_pulse_numbers)

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
        pulsar_groups = discover_pulsars_by_coordinates_optimized(validated_data)
        pulsar_name = list(pulsar_groups.keys())[0] if pulsar_groups else "unknown"

        # 5. Create MetaPulsar
        # Convert file data to single file per PTA format
        single_file_data = {}
        for pta_name, file_list in validated_data.items():
            if not file_list:
                raise ValueError(f"No files found for PTA {pta_name}")
            single_file_data[pta_name] = file_list[0]  # Take first file

        # Create file_pairs from the file data
        file_pairs = {
            pta: (file_dict["par"], file_dict["tim"])
            for pta, file_dict in single_file_data.items()
        }

        # Create output directory if parfile_output_dir is provided
        if parfile_output_dir:
            parfile_output_dir = Path(parfile_output_dir).resolve()
            parfile_output_dir.mkdir(parents=True, exist_ok=True)
        pta_file_dir = Path(tempfile.mkdtemp(prefix="metapulsar_pta_files_")).resolve()

        # Process par files based on strategy
        if combination_strategy == "shared":
            # Create ParameterManager for parfile consistency
            parameter_manager = ParameterManager(
                file_data=single_file_data,
                combine_components=combine_components,
                add_dm_derivatives=add_dm_derivatives,
                output_dir=parfile_output_dir,
                pulsar_name=pulsar_name,
            )

            # Make par files consistent
            shared_parfiles = parameter_manager.make_parfiles_shared()

            # Update file_pairs with shared par files
            file_pairs = {
                pta: (shared_parfiles[pta], single_file_data[pta]["tim"])
                for pta in single_file_data.keys()
                if pta in shared_parfiles
            }
        elif parfile_output_dir:
            # For per_pta strategy, write original par files
            self._write_original_parfiles(
                single_file_data, parfile_output_dir, pulsar_name
            )

        # Create PINT/Tempo2 objects from file pairs using file data
        created = self._create_pulsar_objects(
            file_pairs=file_pairs,
            file_data=single_file_data,
            use_pulse_numbers=pulse_mode,
            pta_file_dir=pta_file_dir,
            return_pta_files=True,
        )
        if isinstance(created, tuple) and len(created) == 2:
            pulsars, pta_files = created
        else:
            pulsars, pta_files = created, {}

        return MetaPulsar(
            pulsars=pulsars,
            combination_strategy=combination_strategy,
            combine_components=combine_components,
            add_dm_derivatives=add_dm_derivatives,
            pta_files=pta_files,
            clock_dir=clock_dir,
        )

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
        pulsar_groups = discover_pulsars_by_coordinates_optimized(file_data)

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
        """Group file data by pulsar using coordinate-based identification.

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
            "Grouping files by pulsar using coordinate-based identification"
        )

        pulsar_groups = discover_pulsars_by_coordinates_optimized(file_data)

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
        pulsar_groups = discover_pulsars_by_coordinates_optimized(file_data)

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
        parfile_output_dir: Path = None,
        use_pulse_numbers: str = "yes",
        clock_dir: Path | str | None = None,
    ) -> Dict[str, MetaPulsar]:
        """Create MetaPulsars for all available pulsars using file data.

        Args:
            file_data: File data from FileDiscovery (per data release)
            combination_strategy: Strategy for combining PTAs
            reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.
            combine_components: List of components to share
            add_dm_derivatives: Whether to ensure DM1, DM2 are present
            parfile_output_dir: Directory to save shared par files (for the shared strategy only).
                If None, par files are not saved to disk. Creates subdirectories for each pulsar.

        Returns:
            Dictionary mapping pulsar names to MetaPulsar objects
        """
        combination_strategy = normalize_combination_strategy(combination_strategy)
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
                    parfile_output_dir=parfile_output_dir,
                    use_pulse_numbers=pulse_mode,
                    clock_dir=clock_dir,
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
        """Get display name for pulsar using B-name preference logic.

        Returns B-name if any PTA uses B-names internally, otherwise J-name.
        This matches the naming logic used in MetaPulsar.

        Args:
            pulsar_name: J-name from coordinate-based discovery
            pulsar_file_data: File data for this pulsar

        Returns:
            Display name (B-name or J-name)
        """
        # Check if any PTA uses B-names internally
        if self._any_pta_uses_b_names(pulsar_file_data):
            # Extract coordinates from first file and convert to B-name
            from .position_helpers import (
                extract_coordinates_from_parfile_optimized,
                bj_name_from_coordinates_optimized,
            )

            first_pta = list(pulsar_file_data.keys())[0]
            first_file = pulsar_file_data[first_pta][0]
            coords = extract_coordinates_from_parfile_optimized(
                first_file["par_content"]
            )

            if coords:
                return bj_name_from_coordinates_optimized(coords[0], coords[1], "B")

        return pulsar_name

    def _any_pta_uses_b_names(
        self, pulsar_file_data: Dict[str, List[Dict[str, Any]]]
    ) -> bool:
        """Check if any PTA uses B-names internally."""
        for pta_name, files in pulsar_file_data.items():
            for file_info in files:
                from .pint_helpers import create_pint_model

                model = create_pint_model(file_info["par_content"])
                pta_pulsar_name = model.PSR.value
                if pta_pulsar_name.startswith("B") and len(pta_pulsar_name) >= 6:
                    return True
        return False

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
    ) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Create PINT/Tempo2 objects from file pairs using file data.

        Args:
            file_pairs: Dictionary mapping PTA names to (parfile, timfile) tuples
            file_data: Dictionary mapping PTA names to file dictionaries
                      Contains timing_package info from FileDiscovery
            use_pulse_numbers: Pulse-number mode (``no``, ``yes``, ``reuse``, ``overwrite``)

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

        for pta_name, (parfile, timfile) in file_pairs.items():
            # Get timing package info from file data
            timing_package = file_data[pta_name]["timing_package"]
            original_par_text = file_data[pta_name]["par_content"]
            tim_metadata = file_data[pta_name].get("tim_metadata")
            parfile = Path(parfile)
            timfile = Path(timfile)

            try:
                if timing_package == "pint":
                    # Create PINT objects
                    if get_model_and_toas is None:
                        raise RuntimeError("PINT not available for PINT creation")

                    with resolved_tim_for_pulse_numbers(
                        use_pulse_numbers,
                        original_par_text,
                        timfile,
                        derive_backend="pint",
                        tim_metadata=tim_metadata,
                    ) as tim_path:
                        model, toas = get_model_and_toas(
                            str(parfile),
                            tim_path,
                            planets=True,
                            allow_T2=True,
                        )
                        if track_pn:
                            ensure_pint_track_minus_2(model)
                        if return_pta_files:
                            pta_files[pta_name] = self._retain_pta_files(
                                pta_name=pta_name,
                                timing_package=timing_package,
                                par_path=parfile,
                                tim_path=Path(tim_path),
                                pta_file_dir=pta_file_dir,
                            )
                    pulsar_objects[pta_name] = (model, toas)

                else:  # tempo2
                    # Create Tempo2 object using sandbox
                    with resolved_tim_for_pulse_numbers(
                        use_pulse_numbers,
                        original_par_text,
                        timfile,
                        derive_backend="tempo2",
                        tim_metadata=tim_metadata,
                    ) as tim_path:
                        if track_pn:
                            with temporary_par_with_track_minus_2(
                                Path(parfile).read_text(encoding="utf-8")
                            ) as par_for_tempo2:
                                t2_psr = tempopulsar(
                                    parfile=str(par_for_tempo2),
                                    timfile=tim_path,
                                    dofit=False,
                                )
                                if return_pta_files:
                                    pta_files[pta_name] = self._retain_pta_files(
                                        pta_name=pta_name,
                                        timing_package=timing_package,
                                        par_path=Path(par_for_tempo2),
                                        tim_path=Path(tim_path),
                                        pta_file_dir=pta_file_dir,
                                    )
                        else:
                            t2_psr = tempopulsar(
                                parfile=str(parfile),
                                timfile=tim_path,
                                dofit=False,
                            )
                            if return_pta_files:
                                pta_files[pta_name] = self._retain_pta_files(
                                    pta_name=pta_name,
                                    timing_package=timing_package,
                                    par_path=parfile,
                                    tim_path=Path(tim_path),
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
        """Copy exact timing-package input files into durable pulsar-owned paths."""
        pta_safe = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in pta_name
        )
        par_src = Path(par_path).resolve()
        tim_src = Path(tim_path).resolve()
        if not par_src.is_file():
            raise FileNotFoundError(f"Session par file does not exist: {par_src}")
        if not tim_src.is_file():
            raise FileNotFoundError(f"Session tim file does not exist: {tim_src}")
        pta_file_dir.mkdir(parents=True, exist_ok=True)
        par_dst = pta_file_dir / f"{pta_safe}.par"
        tim_dst = pta_file_dir / f"{pta_safe}.tim"
        shutil.copy2(par_src, par_dst)
        shutil.copy2(tim_src, tim_dst)
        for rel in _collect_included_tim_rel_paths(tim_src):
            inc_src = (tim_src.parent / rel).resolve()
            inc_dst = pta_file_dir / rel
            inc_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inc_src, inc_dst)
        return {
            "par_path": par_dst,
            "tim_path": tim_dst,
            "timing_package": timing_package,
        }

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
    parfile_output_dir: Path = None,
    use_pulse_numbers: str = "yes",
    clock_dir: Path | str | None = None,
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
        parfile_output_dir: Directory to save shared par files (for the shared strategy only).
            If None, par files are not saved to disk.
        use_pulse_numbers: Pulse-number mode: ``"no"``, ``"yes"`` (default), ``"reuse"``,
            or ``"overwrite"``. See ``MetaPulsarFactory.create_metapulsar`` for semantics.

    Returns:
        MetaPulsar object

    Raises:
        ValueError: If no files found, multiple pulsars detected, or invalid parameters
        RuntimeError: If Enterprise Pulsar creation fails
    """
    factory = MetaPulsarFactory()
    return factory.create_metapulsar(
        file_data=file_data,
        combination_strategy=combination_strategy,
        reference_pta=reference_pta,
        combine_components=combine_components,
        add_dm_derivatives=add_dm_derivatives,
        parfile_output_dir=parfile_output_dir,
        use_pulse_numbers=use_pulse_numbers,
        clock_dir=clock_dir,
    )


def create_all_metapulsars(
    file_data: Dict[str, List[Dict[str, Any]]],
    combination_strategy: str = "shared",
    reference_pta: str = None,
    combine_components: List[str] = DEFAULT_COMBINE_COMPONENTS,
    add_dm_derivatives: bool = True,
    parfile_output_dir: Path = None,
    use_pulse_numbers: str = "yes",
    clock_dir: Path | str | None = None,
) -> Dict[str, MetaPulsar]:
    """Create MetaPulsars for all available pulsars using file data.

    Args:
        file_data: File data from FileDiscovery (per data release)
        combination_strategy: Strategy for combining PTAs
        reference_pta: PTA to use as reference for all pulsars. If None, auto-selects by timespan.
        combine_components: List of components to share
        add_dm_derivatives: Whether to ensure DM1, DM2 are present
        parfile_output_dir: Directory to save shared par files (for the shared strategy only).
            If None, par files are not saved to disk. Creates subdirectories for each pulsar.
        use_pulse_numbers: Pulse-number mode passed to each ``create_metapulsar`` call
            (``"no"``, ``"yes"``, ``"reuse"``, or ``"overwrite"``; default ``"yes"``).

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
        parfile_output_dir=parfile_output_dir,
        use_pulse_numbers=use_pulse_numbers,
        clock_dir=clock_dir,
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
