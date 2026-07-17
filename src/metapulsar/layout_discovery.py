#!/usr/bin/env python3
"""
Phase 1: Heuristic-based Pattern Detection for PTA Data Releases

This module provides automatic pattern discovery for PTA data releases
without requiring machine learning or external dependencies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence
import re
from collections import defaultdict, Counter
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from loguru import logger

DEFAULT_EXCLUDED_DIRS: Sequence[str] = (
    "alternate",
    "extratim",
    "clock",
    "template",
    "wideband",
)


class LayoutDiscoveryService:
    """Heuristic-based pattern discovery for PTA data releases."""

    def __init__(
        self,
        working_dir: str = None,
        verbose: bool = True,
        excluded_dirs: Sequence[str] = DEFAULT_EXCLUDED_DIRS,
        name: Optional[str] = None,
    ):
        """Initialize the layout discovery service.

        Args:
            working_dir: Working directory for resolving relative paths. If None, uses current working directory.
            verbose: Default verbosity setting for method calls. Can be overridden in individual method calls.
            excluded_dirs: List of directory names to exclude from analysis. Defaults to common problematic directories.
            name: Optional name to use for the discovered layout when returning results.
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.verbose = verbose
        self.excluded_dirs = excluded_dirs
        self.logger = logger
        self.name = name
        # Common PTA patterns we've seen
        self.known_pulsar_patterns = [
            r"([BJ]\d{4}[+-]\d{2,4}[A-Z]?)",  # Standard B/J names
            r"([BJ]\d{4}[+-]\d{2,4})",  # Without optional suffix
        ]

        # Common directory structures
        self.common_subdirs = ["par", "tim", "data", "pulsars"]

    def discover_layout(
        self, working_dir: str = None, verbose: bool = None, name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Discover PTA data release layout with user-friendly name.

        Args:
            working_dir: Directory to analyze. If None, uses instance default.
            verbose: If True, prints discovered layout to console. If None, uses instance default.
            name: Optional override for the returned layout name key.

        Returns:
            Dictionary of data release configurations
        """
        # Use instance defaults if not specified
        if working_dir is None:
            working_dir = self.working_dir
        if verbose is None:
            verbose = self.verbose

        base_path = Path(working_dir)

        structure = self._analyze_directory_structure(base_path)
        data_release = self._generate_pta_data_release(structure)

        # Determine the layout name key with override precedence: method arg > instance > default
        layout_name = (
            name or getattr(self, "name", None) or f"discovered_{base_path.name}"
        )

        if verbose:
            print(f"Discovered layout in {base_path}:")
            for key, value in data_release.items():
                if key != "discovery_confidence":
                    print(f"  - {key} = {repr(value)}")

        return {layout_name: data_release}

    def _analyze_directory_structure(self, base_path: Path) -> Dict[str, Any]:
        """Analyze a directory structure and infer PTA patterns."""
        if not base_path.exists():
            raise ValueError(f"Directory {base_path} does not exist")

        self.logger.info(f"Analyzing directory structure: {base_path}")

        # Find all par and tim files, filtering out wideband data and excluded directories
        par_files = [
            f
            for f in base_path.rglob("*.par")
            if not self._is_wideband_file(f)
            and not self._is_excluded_directory(f, base_path)
        ]
        tim_files = [
            f
            for f in base_path.rglob("*.tim")
            if not self._is_wideband_file(f)
            and not self._is_excluded_directory(f, base_path)
        ]

        if not par_files:
            raise ValueError(f"No .par files found in {base_path}")

        self.logger.info(
            f"Found {len(par_files)} par files and {len(tim_files)} tim files"
        )

        # Analyze structure
        structure = {
            "base_path": str(base_path),
            "par_files": [str(f) for f in par_files],
            "tim_files": [str(f) for f in tim_files],
            "directory_depth": self._analyze_depth(base_path),
            "subdirectory_structure": self._analyze_subdirs(base_path),
            "file_naming_patterns": self._analyze_naming_patterns(par_files, tim_files),
            "pulsar_names": self._extract_pulsar_names(par_files),
        }

        return structure

    def _generate_pta_data_release(
        self, structure: Dict, timing_package: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a complete PTA data release from structure analysis."""

        # Use user input if provided, otherwise detect
        if timing_package:
            detected_timing_package = timing_package
        else:
            detected_timing_package = self._detect_timing_package(
                structure["par_files"]
            )

        # Generate patterns
        par_pattern = self._generate_par_pattern(structure)
        tim_pattern = self._generate_tim_pattern(structure)

        # Determine base directory (relative to structure base)
        base_dir = self._determine_base_dir(structure)

        data_release = {
            "base_dir": base_dir,
            "par_pattern": par_pattern,
            "tim_pattern": tim_pattern,
            "timing_package": detected_timing_package,
            "description": f"Auto-discovered PTA from {structure['base_path']}",
            "discovery_confidence": self._calculate_confidence(structure),
        }

        return data_release

    def _analyze_depth(self, base_path: Path) -> int:
        """Analyze maximum directory depth."""
        max_depth = 0
        for path in base_path.rglob("*"):
            if path.is_file():
                depth = len(path.relative_to(base_path).parts) - 1
                max_depth = max(max_depth, depth)
        return max_depth

    def _analyze_subdirs(self, base_path: Path) -> Dict[str, int]:
        """Analyze subdirectory usage patterns."""
        subdir_counts = defaultdict(int)

        for path in base_path.rglob("*"):
            if path.is_dir():
                rel_path = path.relative_to(base_path)
                if len(rel_path.parts) == 1:  # Direct subdirectories
                    subdir_counts[rel_path.name] += 1

        return dict(subdir_counts)

    def _analyze_naming_patterns(
        self, par_files: List[Path], tim_files: List[Path]
    ) -> Dict[str, Any]:
        """Analyze file naming patterns."""
        patterns = {
            "par_naming": self._analyze_file_naming(par_files),
            "tim_naming": self._analyze_file_naming(tim_files),
            "common_prefixes": self._find_common_prefixes(par_files + tim_files),
            "common_suffixes": self._find_common_suffixes(par_files + tim_files),
        }
        return patterns

    def _analyze_file_naming(self, files: List[Path]) -> Dict[str, Any]:
        """Analyze naming patterns in a list of files."""
        if not files:
            return {}

        # Get relative paths from base
        base = files[0].parent
        while not all(f.is_relative_to(base) for f in files):
            base = base.parent

        rel_paths = [f.relative_to(base) for f in files]

        # Analyze patterns
        naming = {
            "has_subdirs": any(len(p.parts) > 1 for p in rel_paths),
            "common_subdirs": self._find_common_subdirs(rel_paths),
            "file_stems": [p.stem for p in rel_paths],
            "extensions": [p.suffix for p in rel_paths],
        }

        return naming

    def _find_common_subdirs(self, paths: List[Path]) -> List[str]:
        """Find common subdirectory patterns."""
        subdirs = []
        for path in paths:
            if len(path.parts) > 1:
                subdirs.extend(path.parts[:-1])  # All parts except filename

        # Count and return most common
        subdir_counts = Counter(subdirs)
        return [subdir for subdir, count in subdir_counts.most_common(3)]

    def _find_common_prefixes(self, files: List[Path]) -> List[str]:
        """Find common prefixes in filenames."""
        stems = [f.stem for f in files]
        if not stems:
            return []

        # Find longest common prefix
        common_prefix = ""
        for i in range(min(len(s) for s in stems)):
            if all(s[i] == stems[0][i] for s in stems):
                common_prefix += stems[0][i]
            else:
                break

        return [common_prefix] if common_prefix else []

    def _find_common_suffixes(self, files: List[Path]) -> List[str]:
        """Find common suffixes in filenames."""
        stems = [f.stem for f in files]
        if not stems:
            return []

        # Find longest common suffix
        common_suffix = ""
        for i in range(1, min(len(s) for s in stems) + 1):
            if all(s[-i:] == stems[0][-i:] for s in stems):
                common_suffix = stems[0][-i:] + common_suffix
            else:
                break

        return [common_suffix] if common_suffix else []

    def _extract_pulsar_names(self, par_files: List[Path]) -> List[str]:
        """Extract pulsar names using existing pattern matching."""
        pulsar_names = []

        for file_path in par_files:
            for pattern in self.known_pulsar_patterns:
                try:
                    match = re.search(pattern, str(file_path))
                    if match:
                        pulsar_names.append(match.group(1))
                        break
                except re.error:
                    continue

        return list(set(pulsar_names))  # Remove duplicates

    def _is_wideband_file(self, file_path: Path) -> bool:
        """
        Check if a file is wideband data that should be ignored.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file is wideband data and should be ignored
        """
        # Convert to string for easier checking
        path_str = str(file_path).lower()

        # Check for wideband indicators in the path
        wideband_indicators = ["wideband", "wb_", "_wb", "wide_band", "wide-band"]

        for indicator in wideband_indicators:
            if indicator in path_str:
                return True

    def _is_excluded_directory(self, file_path: Path, base_path: Path) -> bool:
        """
        Check if a file is in an excluded directory that should be ignored.

        Args:
            file_path: Path to the file to check
            base_path: Base directory being analyzed

        Returns:
            True if the file is in an excluded directory
        """
        try:
            rel_path = file_path.relative_to(base_path)
            # Check if any part of the path contains excluded directories
            for part in rel_path.parts:
                if part.lower() in self.excluded_dirs:
                    return True
            return False
        except ValueError:
            # File is not relative to base_path, exclude it
            return True

    def _find_common_path_parts(self, rel_paths: List[Path]) -> List[str]:
        """
        Find common path parts in a list of relative paths.

        Args:
            rel_paths: List of relative paths

        Returns:
            List of common path parts
        """
        if not rel_paths:
            return []

        # Start with the first path
        common_parts = list(rel_paths[0].parts)

        # Find common parts with other paths
        for path in rel_paths[1:]:
            path_parts = list(path.parts)
            # Keep only the parts that match
            common_parts = [
                part
                for i, part in enumerate(common_parts)
                if i < len(path_parts) and path_parts[i] == part
            ]

            if not common_parts:
                break

        return common_parts

    def _detect_timing_package(self, par_files: List[str]) -> str:
        """Detect timing package from par file content."""
        if not par_files:
            return "tempo2"  # Default

        # Check more files for better detection
        sample_files = par_files[:10]  # Check first 10 files

        for par_file_path in sample_files:
            try:
                par_file = Path(par_file_path)
                content = par_file.read_text(encoding="utf-8", errors="ignore")

                # BINARY T2 is definitive for tempo2 - check this first
                if "BINARY" in content and "T2" in content:
                    # Check if BINARY and T2 are on the same line
                    lines = content.split("\n")
                    for line in lines:
                        if "BINARY" in line and "T2" in line:
                            self.logger.info(
                                f"Found BINARY T2 in {par_file.name} - detected tempo2"
                            )
                            return "tempo2"

            except Exception as e:
                self.logger.warning(f"Could not read {par_file_path}: {e}")
                continue

        # Check for PINT comments in par files
        for par_file_path in sample_files:
            try:
                par_file = Path(par_file_path)
                content = par_file.read_text(encoding="utf-8", errors="ignore")

                # Look for PINT comments - lines starting with #
                lines = content.split("\n")
                for line in lines:
                    if line.strip().startswith("#") and "pint" in line.lower():
                        self.logger.info(
                            f"Found PINT comment in {par_file.name} - detected pint"
                        )
                        return "pint"

            except Exception as e:
                self.logger.warning(f"Could not read {par_file_path}: {e}")
                continue

        # Check if this is a NANOGrav PTA (always uses PINT)
        for par_file_path in sample_files:
            if "nanograv" in par_file_path.lower():
                self.logger.info("NANOGrav PTA detected - defaulting to PINT")
                return "pint"

        # If no BINARY T2 found, use other heuristics
        for par_file_path in sample_files:
            try:
                par_file = Path(par_file_path)
                content = par_file.read_text(encoding="utf-8", errors="ignore")

                # Look for PINT-specific indicators
                if any(
                    indicator in content.lower()
                    for indicator in ["pint", "enterprise", "gls", "glitch", "nanograv"]
                ):
                    return "pint"

                # Look for tempo2-specific indicators
                if any(
                    indicator in content.lower()
                    for indicator in ["tempo2", "t2", "jodrell", "epta", "ppta"]
                ):
                    return "tempo2"

            except Exception as e:
                self.logger.warning(f"Could not read {par_file_path}: {e}")
                continue

        return "tempo2"  # Default fallback

    def _generate_par_pattern(self, structure: Dict) -> str:
        """Generate par file pattern from structure analysis."""
        pulsar_names = structure["pulsar_names"]

        if not pulsar_names:
            # Fallback to generic pattern with flexible extension
            return r"([BJ]\d{4}[+-]\d{2,4}[A-Z]?).*\.par"

        # Use the most common pulsar name pattern found
        pulsar_pattern = self.known_pulsar_patterns[0]  # Default to standard pattern

        # Analyze the actual file structure
        par_files = [Path(f) for f in structure["par_files"]]
        if not par_files:
            return f"{pulsar_pattern}.*\\.par"

        # Find the common base directory - start from the structure base path
        structure_base = Path(structure["base_path"])
        base = structure_base

        # Get relative paths from base
        rel_paths = [f.relative_to(base) for f in par_files]

        # Check directory structure patterns
        if all(len(p.parts) > 1 for p in rel_paths):
            # Find common path parts for complex nested structures
            common_parts = self._find_common_path_parts(rel_paths)

            if common_parts:
                # For very deep nested structures, use a more flexible pattern
                if len(common_parts) > 3:
                    # Use wildcard for deep nested structures with flexible suffix
                    return f".*{pulsar_pattern}.*\\.par"
                else:
                    # Build pattern with common path parts
                    path_pattern = "/".join(common_parts)
                    return f"{path_pattern}/{pulsar_pattern}.*\\.par"
            else:
                # Fallback to simple subdirectory analysis
                subdir_names = [p.parts[0] for p in rel_paths]
                subdir_counts = Counter(subdir_names)
                most_common_subdir = subdir_counts.most_common(1)[0][0]

                # Check for common patterns
                if most_common_subdir == "par":
                    return f"par/{pulsar_pattern}.*\\.par"
                elif most_common_subdir == "tim":
                    return f"tim/{pulsar_pattern}.*\\.par"
                else:
                    # Check if subdirectory name matches file stem (pulsar-specific dirs)
                    first_path = rel_paths[0]
                    file_stem = first_path.stem
                    if most_common_subdir == file_stem:
                        return f"{pulsar_pattern}/{pulsar_pattern}.*\\.par"
                    else:
                        # Generic subdirectory pattern
                        return f"{pulsar_pattern}/{pulsar_pattern}.*\\.par"
        else:
            # Files are in root directory
            return f"{pulsar_pattern}.*\\.par"

    def _generate_tim_pattern(self, structure: Dict) -> str:
        """Generate tim file pattern from structure analysis."""

        # Analyze tim file structure separately from par files
        tim_files = [Path(f) for f in structure["tim_files"]]
        if not tim_files:
            # Fallback to generic pattern with flexible extension
            return r"([BJ]\d{4}[+-]\d{2,4}[A-Z]?).*\.tim"

        # Use the most common pulsar name pattern found
        pulsar_pattern = self.known_pulsar_patterns[0]  # Default to standard pattern

        # Find the common base directory - start from the structure base path
        structure_base = Path(structure["base_path"])
        base = structure_base

        # Get relative paths from base
        rel_paths = [f.relative_to(base) for f in tim_files]

        # Check directory structure patterns
        if all(len(p.parts) > 1 for p in rel_paths):
            # Find common path parts for complex nested structures
            common_parts = self._find_common_path_parts(rel_paths)

            if common_parts:
                # For very deep nested structures, use a more flexible pattern
                if len(common_parts) > 3:
                    # Use wildcard for deep nested structures with flexible suffix
                    return f".*{pulsar_pattern}.*\\.tim"
                else:
                    # Build pattern with common path parts
                    path_pattern = "/".join(common_parts)
                    return f"{path_pattern}/{pulsar_pattern}.*\\.tim"
            else:
                # Fallback to simple subdirectory analysis
                subdir_names = [p.parts[0] for p in rel_paths]
                subdir_counts = Counter(subdir_names)
                most_common_subdir = subdir_counts.most_common(1)[0][0]

                # Check for common patterns
                if most_common_subdir == "tim":
                    return f"tim/{pulsar_pattern}.*\\.tim"
                elif most_common_subdir == "par":
                    return f"par/{pulsar_pattern}.*\\.tim"
                else:
                    # Check if subdirectory name matches file stem (pulsar-specific dirs)
                    first_path = rel_paths[0]
                    file_stem = first_path.stem
                    if most_common_subdir == file_stem:
                        return f"{pulsar_pattern}/{pulsar_pattern}.*\\.tim"
                    else:
                        # Generic subdirectory pattern
                        return f"{pulsar_pattern}/{pulsar_pattern}.*\\.tim"
        else:
            # Files are in root directory
            return f"{pulsar_pattern}.*\\.tim"

    def _determine_base_dir(self, structure: Dict) -> str:
        """Determine the base directory for the PTA."""
        # In practice, this might need to be relative to some data root
        return structure["base_path"]

    def _calculate_confidence(self, structure: Dict) -> float:
        """Calculate confidence score for the discovered patterns."""
        confidence = 0.0

        # Base confidence
        confidence += 0.3

        # Bonus for finding pulsar names
        if structure["pulsar_names"]:
            confidence += 0.3

        # Bonus for consistent naming patterns
        par_naming = structure["file_naming_patterns"]["par_naming"]
        tim_naming = structure["file_naming_patterns"]["tim_naming"]

        if par_naming.get("common_prefixes") or par_naming.get("common_suffixes"):
            confidence += 0.2

        if tim_naming.get("common_prefixes") or tim_naming.get("common_suffixes"):
            confidence += 0.2

        return min(confidence, 1.0)


# Convenience function for easy access
def discover_layout(
    working_dir: str = None,
    verbose: bool = True,
    excluded_dirs: Sequence[str] = DEFAULT_EXCLUDED_DIRS,
    name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Convenience function for layout discovery.

    Args:
        working_dir: Directory to analyze. If None, uses current directory.
        verbose: If True, prints discovered layout to console.
        excluded_dirs: List of directory names to exclude from analysis.
        name: Optional name to use for the returned layout key.

    Returns:
        Dictionary of data release configurations
    """
    engine = LayoutDiscoveryService(working_dir, verbose, excluded_dirs, name=name)
    return engine.discover_layout(working_dir, verbose, name=name)


def combine_layouts(
    *layouts: Dict[str, Dict[str, Any]], include_defaults: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Combine multiple discovered data release layouts into a single dictionary.

    Args:
        *layouts: Variable number of layout dictionaries from discover_layout()
        include_defaults: If True, includes default PTA_DATA_RELEASES in the combination

    Returns:
        Combined dictionary with all data releases

    Example:
        layout1 = discover_layout("../../data/ipta-dr2/EPTA_v2.2")
        layout2 = discover_layout("../../data/ipta-dr2/NANOGrav_9y")
        layout3 = discover_layout("../../data/ipta-dr2/PPTA_dr1dr2")
        combined = combine_layouts(layout1, layout2, layout3, include_defaults=True)
    """
    combined = {}

    # Add default PTA data releases if requested
    if include_defaults:
        from .file_discovery_service import PTA_DATA_RELEASES

        combined.update(PTA_DATA_RELEASES)

    # Add custom layouts
    for layout in layouts:
        combined.update(layout)

    return combined
