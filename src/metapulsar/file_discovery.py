"""File Discovery Service for PTA data files.

This service handles all file discovery operations and data release directory layout management.
It is completely independent - NO external dependencies on PINT, libstempo, or other components.
Uses only regex patterns for file matching and pattern extraction.
"""

from typing import Dict, List, Any, Union
from pathlib import Path
import re
from loguru import logger

from .tim_file_analyzer import TimFileAnalyzer, TimMetadata

__all__ = [
    "FileDiscovery",
    "PTA_DATA_RELEASES",
    "discover_files",
    "get_pulsar_names_from_file_data",
    "filter_file_data_by_pulsars",
]

PTA_DATA_RELEASES = {
    "epta_dr1_v2_2": {
        "base_dir": "EPTA_v2.2/",
        "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})/\1\.par",
        "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})/\1_all\.tim",
        "timing_package": "tempo2",
        "description": "EPTA Data Release 1 v2.2",
    },
    "epta_dr2": {
        "base_dir": "EPTA_DR2/",
        "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})/\1\.par",
        "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})/\1_all\.tim",
        "timing_package": "tempo2",
        "description": "EPTA Data Release 2",
    },
    "ppta_dr2": {
        "base_dir": "PPTA_dr1dr2/",
        "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})_dr1dr2\.par",
        "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})_dr1dr2\.tim",
        "timing_package": "tempo2",
        "description": "PPTA Data Release 1+2",
    },
    "nanograv_9y": {
        "base_dir": "NANOGrav_9y/",
        "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})_NANOGrav_9yv1\.gls\.par",
        "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})_NANOGrav_9yv1\.tim",
        "timing_package": "pint",
        "description": "NANOGrav 9-year Data Release",
    },
    "inpta_dr1": {
        "base_dir": "InPTA_DR1/",
        "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})\/\1\.par",
        "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})\/\1_all\.tim",
        "timing_package": "tempo2",
        "description": "InPTA Data Release 1",
    },
    "mpta_dr1": {
        "base_dir": "MPTA_DR1/",
        "par_pattern": r"MTMSP-([BJ]\d{4}[+-]\d{2,4})-\.par",
        "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})_16ch\.tim",
        "timing_package": "tempo2",
        "description": "MPTA Data Release 1",
    },
    "mpta_dr2": {
        # Flat release: one <PSR>.par and one <PSR>.tim per pulsar.
        "base_dir": "MPTA_DR2/",
        "par_pattern": r"MPTA_DR2/([BJ]\d{4}[+-]\d{2,4})\.par$",
        "tim_pattern": r"MPTA_DR2/([BJ]\d{4}[+-]\d{2,4})\.tim$",
        "timing_package": "tempo2",
        "description": "MPTA Data Release 2",
    },
    "ppta_dr3": {
        # Flat release. The plain <PSR>.par is the Tempo2 solution; the release
        # also ships a derived <PSR>_pint.par, and working subdirectories
        # (dr2/, uwl/, temp/) that repeat the pulsar names, so both patterns are
        # anchored to a file sitting directly in PPTA_DR3/. Optional trailing
        # letter covers globular-cluster names such as J1824-2452A.
        "base_dir": "PPTA_DR3/",
        "par_pattern": r"PPTA_DR3/([BJ]\d{4}[+-]\d{2,4}[A-Z]?)\.par$",
        "tim_pattern": r"PPTA_DR3/([BJ]\d{4}[+-]\d{2,4}[A-Z]?)\.tim$",
        "timing_package": "tempo2",
        "description": "PPTA Data Release 3",
    },
    "nanograv_12y": {
        "base_dir": "NANOGrav_12y/",
        "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})(?!.*\.t2)_NANOGrav_12yv2\.gls\.par",
        "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})_NANOGrav_12yv2\.tim",
        "timing_package": "pint",
        "description": "NANOGrav 12-year Data Release",
    },
    "nanograv_15y": {
        "base_dir": "NANOGrav_15y/",
        "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})(?!.*(ao|gbt)).*\.par",
        "tim_pattern": r"tim/([BJ]\d{4}[+-]\d{2,4})(?!.*(ao|gbt)).*\.tim",
        "timing_package": "pint",
        "description": "NANOGrav 15-year Data Release",
    },
}


def extract_pulsar_name_from_path(
    file_path: Path, pulsar_name_pattern: str = r"([BJ]\d{4}[+-]\d{2,4}[A-Z]?)"
) -> str:
    """Extract pulsar name from file path using regex pattern.

    Args:
        file_path: Path to the par file
        pulsar_name_pattern: Regex pattern for extracting canonical pulsar names.
                           Default matches: J1234-5678, J1234+5678, B2144-09, B1234+67A, J5432-2235C

    Returns:
        Extracted pulsar name (e.g., "J1857+0943", "B1855+09")

    Raises:
        ValueError: If no match found or pattern is invalid
    """
    import re

    try:
        regex = re.compile(pulsar_name_pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pulsar_name_pattern}': {e}")

    match = regex.search(str(file_path))
    if not match:
        raise ValueError(
            f"No match found for file {file_path} with pattern {pulsar_name_pattern}"
        )

    # Extract pattern from regex capture group
    pulsar_name = match.group(1) if match.groups() else match.group(0)
    return pulsar_name


class FileDiscovery:
    """Independent service for discovering PTA data files and managing data release directory layouts.

    This service handles all data release-related operations and can be used
    independently of MetaPulsarFactory and ParFileManager.

    Key Features:
    - NO external dependencies (PINT, libstempo, etc.)
    - Uses only regex patterns for file matching
    - Does NOT validate pulsar names - just extracts patterns
    - Completely isolated and testable
    """

    def __init__(
        self,
        working_dir: str = None,
        pta_data_releases: Dict = None,
        verbose: bool = True,
    ):
        """Initialize the file discovery service.

        Args:
            working_dir: Working directory for resolving relative paths. If None, uses current working directory.
            pta_data_releases: Dictionary of data releases. If None, uses default presets.
            verbose: Default verbosity setting for method calls. Can be overridden in individual method calls.
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.data_releases = pta_data_releases or PTA_DATA_RELEASES.copy()
        self.verbose = verbose
        self.logger = logger
        self._tim_analyzer = TimFileAnalyzer()

    def discover_patterns_in_data_release(self, data_release_name: str) -> List[str]:
        """Discover all file patterns in a single data release using regex.

        Args:
            data_release_name: Name of the data release to search

        Returns:
            List of regex-extracted patterns (NOT validated pulsar names)

        Raises:
            KeyError: If data release not found in directory layouts
        """
        if data_release_name not in self.data_releases:
            raise KeyError(
                f"Data release '{data_release_name}' not found in data releases"
            )

        data_release = self.data_releases[data_release_name]
        return self._discover_patterns_in_data_release(data_release)

    def discover_patterns_in_data_releases(
        self, data_release_names: List[str]
    ) -> Dict[str, List[str]]:
        """Discover all file patterns in multiple data releases using regex.

        Args:
            data_release_names: List of data release names to search

        Returns:
            Dictionary mapping data release names to lists of regex-extracted patterns
        """
        result = {}
        for data_release_name in data_release_names:
            try:
                result[data_release_name] = self.discover_patterns_in_data_release(
                    data_release_name
                )
            except KeyError as e:
                self.logger.error(
                    f"Data release '{data_release_name}' not found in directory layouts"
                )
                raise e
        return result

    def _discover_all_files_in_data_releases(
        self, data_release_names: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Internal method for discovering all file pairs in selected data releases using regex patterns.

        Args:
            data_release_names: List of data release names to search. If None, searches all data releases.

        Returns:
            Dictionary mapping data release names to lists of enriched file dictionaries
            Format: {data_release_name: [{'par': parfile_path, 'tim': timfile_path, 'timing_package': 'pint', 'tim_metadata': TimMetadata(...)}, ...]}
        """
        if data_release_names is None:
            data_release_names = self.list_data_releases()

        result = {}

        for data_release_name in data_release_names:
            if data_release_name not in self.data_releases:
                self.logger.error(
                    f"Data release '{data_release_name}' not found in data releases"
                )
                raise KeyError(
                    f"Data release '{data_release_name}' not found in data releases"
                )

            result[data_release_name] = self._discover_all_file_pairs_in_data_release(
                self.data_releases[data_release_name]
            )

        return result

    def list_data_releases(self) -> List[str]:
        """Get list of all data release names in the directory layouts.

        Returns:
            List of data release names, sorted alphabetically
        """
        return sorted(self.data_releases.keys())

    def discover_files(
        self,
        data_release_names: Union[str, List[str], None] = None,
        verbose: bool = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Discover files with user-friendly name and verbose output.

        Args:
            data_release_names: Single data release name, list of data release names, or None to search all.
            verbose: If True, prints summary of found files to console. If None, uses instance default.

        Returns:
            Dictionary mapping data release names to lists of file dictionaries
        """
        # Use instance default if verbose not specified
        if verbose is None:
            verbose = self.verbose

        # Convert single string to list for internal processing
        if isinstance(data_release_names, str):
            data_release_names = [data_release_names]

        result = self._discover_all_files_in_data_releases(data_release_names)

        if verbose:
            print("Found:")
            for pta_name, files in result.items():
                if files:
                    print(f"  - {pta_name}: {len(files)} pulsars")
                else:
                    print(f"  (No pulsars for: {pta_name})")

        return result

    def add_data_release(self, name: str, data_release: Dict) -> None:
        """Add a data release.

        Args:
            name: Name of the data release
            data_release: Dictionary containing data release specification

        Raises:
            ValueError: If data release with same name already exists or data_release is invalid
        """
        if name in self.data_releases:
            raise ValueError(f"Data release '{name}' already exists in data releases")

        self._validate_data_release(data_release)
        self.data_releases[name] = data_release
        self.logger.debug(f"Added data release: {name}")

    def _validate_data_release(self, data_release: Dict) -> None:
        """Validate a data release dictionary.

        Args:
            data_release: Data release dictionary to validate

        Raises:
            ValueError: If data release is invalid
        """
        required_keys = {
            "base_dir",
            "par_pattern",
            "tim_pattern",
            "timing_package",
        }
        missing_keys = required_keys - data_release.keys()

        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if data_release["timing_package"] not in ["pint", "tempo2"]:
            raise ValueError(
                f"Invalid timing_package: {data_release['timing_package']}. Must be 'pint' or 'tempo2'"
            )

        # Validate regex patterns
        try:
            re.compile(data_release["par_pattern"])
            re.compile(data_release["tim_pattern"])
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _discover_patterns_in_data_release(self, data_release: Dict) -> List[str]:
        """Discover all file patterns in a single data release using regex.

        Args:
            data_release: Data release dictionary

        Returns:
            List of regex-extracted patterns (NOT validated pulsar names)
        """
        base_path = self.working_dir / data_release["base_dir"]
        if not base_path.exists():
            return []

        patterns = set()

        # Use regex for file discovery and pattern extraction
        try:
            regex = re.compile(data_release["par_pattern"])
        except re.error as e:
            self.logger.error(
                f"Invalid regex pattern '{data_release['par_pattern']}': {e}"
            )
            return []

        for file_path in base_path.rglob("*.par"):
            match = regex.search(str(file_path))
            if match:
                # Extract pattern from regex capture group
                pattern = match.group(1) if match.groups() else match.group(0)
                patterns.add(pattern)

        return list(patterns)

    def _discover_all_file_pairs_in_data_release(
        self, data_release: Dict
    ) -> List[Dict[str, Path]]:
        """Discover all par/tim file pairs in a data release.

        Files are matched by their canonical pulsar name (e.g., J1857+0943, B1855+09A).
        """
        base_path = self.working_dir / data_release["base_dir"]
        if not base_path.exists():
            return []

        file_pairs = []
        par_regex = re.compile(data_release["par_pattern"])
        tim_regex = re.compile(data_release["tim_pattern"])

        # Step 1: Find all par files and extract their canonical pulsar names
        par_files_by_pulsar = {}
        for par_file in base_path.rglob("*.par"):
            par_match = par_regex.search(str(par_file))
            if par_match:
                # Extract canonical pulsar name using helper function
                try:
                    pulsar_name = extract_pulsar_name_from_path(par_file)
                    par_files_by_pulsar[pulsar_name] = par_file
                except ValueError:
                    # Skip files that don't match pulsar name pattern
                    continue

        # Step 2: Find all tim files and extract their canonical pulsar names
        tim_files_by_pulsar = {}
        for tim_file in base_path.rglob("*.tim"):
            tim_match = tim_regex.search(str(tim_file))
            if tim_match:
                # Extract canonical pulsar name using helper function
                try:
                    pulsar_name = extract_pulsar_name_from_path(tim_file)
                    tim_files_by_pulsar[pulsar_name] = tim_file
                except ValueError:
                    # Skip files that don't match pulsar name pattern
                    continue

        # Step 3: Match par and tim files by canonical pulsar name
        for pulsar_name in par_files_by_pulsar:
            if pulsar_name in tim_files_by_pulsar:
                tim_metadata = self._get_tim_metadata(tim_files_by_pulsar[pulsar_name])

                file_pairs.append(
                    {
                        "par": par_files_by_pulsar[pulsar_name],
                        "tim": tim_files_by_pulsar[pulsar_name],
                        "timing_package": data_release["timing_package"],
                        "tim_metadata": tim_metadata,
                        "par_content": par_files_by_pulsar[pulsar_name].read_text(
                            encoding="utf-8"
                        ),
                    }
                )

        return file_pairs

    def _get_tim_metadata(self, tim_file_path: Path) -> TimMetadata:
        """Extract unified TIM metadata using the shared analyzer cache."""
        try:
            return self._tim_analyzer.get_tim_metadata(tim_file_path)
        except Exception as e:
            self.logger.warning(
                f"Could not parse TIM metadata for {tim_file_path}: {e}"
            )
            return TimMetadata(
                toa_count=0,
                mjd_min=None,
                mjd_max=None,
                timespan_days=0.0,
                pn_with_count=0,
                pn_without_count=0,
                pn_status="none",
                parse_warnings=(f"metadata extraction failed: {e}",),
            )


# Convenience function for easy access
def discover_files(
    pta_data_releases: Dict,
    working_dir: str = None,
    data_release_names: Union[str, List[str], None] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function for file discovery.

    Args:
        pta_data_releases: Dictionary of data releases (typically from layout discovery).
        working_dir: Working directory for resolving relative paths. If None, uses current working directory.
        data_release_names: Single data release name, list of data release names, or None to search all.
        verbose: If True, prints summary of found files to console.

    Returns:
        Dictionary mapping data release names to lists of file dictionaries
    """
    service = FileDiscovery(working_dir, pta_data_releases, verbose)
    return service.discover_files(data_release_names, verbose)


def get_pulsar_names_from_file_data(
    file_data: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """
    Extract catalog pulsar names from file data using position-based discovery.

    Args:
        file_data: File data from FileDiscovery (per data release)

    Returns:
        List of B-preferred catalog names (e.g. ['J0613-0200', 'B1855+09'])

    Raises:
        ValueError: If no valid pulsar files found
    """
    from .metapulsar_factory import MetaPulsarFactory

    factory = MetaPulsarFactory()
    pulsar_groups = factory.group_files_by_pulsar(file_data)

    if not pulsar_groups:
        raise ValueError("No valid pulsar files found in file_data")

    return list(pulsar_groups.keys())


def filter_file_data_by_pulsars(
    file_data: Dict[str, List[Dict[str, Any]]], pulsar_names: Union[str, List[str]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter file data to include only specified pulsars.

    Accepts catalog names (PSRJ/PSR/PSRB) and path-derived aliases from par/tim
    filenames. Truncated coordinate designators are not registered.

    Args:
        file_data: File data from FileDiscovery (per data release)
        pulsar_names: Single pulsar name or list of catalog/path aliases

    Returns:
        Filtered file data containing only the specified pulsars

    Raises:
        ValueError: If no matching pulsars found
    """
    from .metapulsar_factory import MetaPulsarFactory
    from .position_helpers import build_alias_map

    # Normalize input to list
    if isinstance(pulsar_names, str):
        pulsar_names = [pulsar_names]

    factory = MetaPulsarFactory()
    pulsar_groups = factory.group_files_by_pulsar(file_data)

    if not pulsar_groups:
        raise ValueError("No valid pulsar files found in file_data")

    name_mapping = build_alias_map(pulsar_groups)

    matching_group_names = []
    for requested_name in pulsar_names:
        if requested_name in name_mapping:
            canonical_name = name_mapping[requested_name]
            if canonical_name not in matching_group_names:
                matching_group_names.append(canonical_name)
        else:
            raise ValueError(f"Pulsar '{requested_name}' not found in file data")

    if not matching_group_names:
        raise ValueError(f"No matching pulsars found for: {pulsar_names}")

    filtered_file_data = {}
    for group_name in matching_group_names:
        if group_name in pulsar_groups:
            for pta_name, files in pulsar_groups[group_name].items():
                if pta_name not in filtered_file_data:
                    filtered_file_data[pta_name] = []
                filtered_file_data[pta_name].extend(files)

    return filtered_file_data
