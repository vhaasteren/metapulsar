"""File Discovery Service for PTA data files.

This service handles all file discovery operations and data release directory layout management.
It is completely independent - NO external dependencies on PINT, libstempo, or other components.
Uses only regex patterns for file matching and pattern extraction.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from pathlib import Path
import re
from loguru import logger

from .tim_file_analyzer import TimFileAnalyzer, TimMetadata

__all__ = [
    "FileDiscovery",
    "PTA_DATA_RELEASES",
    "FileSelectionError",
    "AmbiguousFileError",
    "MissingOverrideError",
    "select_release_file",
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
        "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4}[A-Z]?)_dr1dr2\.par",
        "tim_pattern": r"tim/v3/([BJ]\d{4}[+-]\d{2,4}[A-Z]?)_dr1dr2\.tim",
        "timing_package": "tempo2",
        "description": "PPTA Data Release 1+2",
    },
    "nanograv_9y": {
        "base_dir": "NANOGrav_9y/",
        # J1713+0747 ships two separately-fitted solutions: the tempo1 par
        # (BINARY DD + PAASCNODE, which neither PINT nor tempo2 implements) and
        # the t2 par the release README designates for tempo2 (BINARY T2 +
        # KOM/KIN, which PINT ingests as DDK). Both are matched; precedence
        # picks the one our engines can actually evaluate. The fallback carries a
        # negative lookbehind to keep the two rules disjoint: a plain
        # `\.gls\.par$` would also match the t2 file, so the list would then give
        # the right answer only by virtue of its order.
        "par_pattern": r"par/([BJ]\d{4}[+-]\d{2,4})_NANOGrav_9yv1(?:\.t2)?\.gls\.par$",
        "par_precedence": [r"\.t2\.gls\.par$", r"(?<!\.t2)\.gls\.par$"],
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
    "inpta_dr2": {
        "base_dir": "InPTA.DR2/",
        "par_pattern": r"([BJ]\d{4}[+-]\d{2,4})/\1\.DMX\.par$",
        "tim_pattern": r"([BJ]\d{4}[+-]\d{2,4})/\1_all\.tim$",
        "timing_package": "tempo2",
        "description": "InPTA Data Release 2",
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
        # Flat release. Both the Tempo2 solution (<PSR>.par) and the derived PINT
        # solution (<PSR>_pint.par) are matched; precedence follows the spec's
        # timing_package, so flipping that key selects the matching par. Working
        # subdirectories (dr2/, uwl/, temp/) repeat the pulsar names, so both
        # patterns stay anchored to a file sitting directly in PPTA_DR3/.
        # Optional trailing letter covers globular-cluster names such as J1824-2452A.
        "base_dir": "PPTA_DR3/",
        "par_pattern": r"PPTA_DR3/([BJ]\d{4}[+-]\d{2,4}[A-Z]?)(?:_pint)?\.par$",
        "par_precedence": [
            {"pattern": r"_pint\.par$", "timing_package": "pint"},
            r"(?<!_pint)\.par$",
        ],
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


_TIMING_PACKAGES = ("pint", "tempo2")


class FileSelectionError(ValueError):
    """Raised when a release layout cannot name exactly one file for a pulsar."""


class AmbiguousFileError(FileSelectionError):
    """Several release files match one pulsar with equal precedence."""


class MissingOverrideError(FileSelectionError):
    """A release override names a file that is not on disk."""


class _PrecedenceRule(NamedTuple):
    """One compiled ``{par,tim}_precedence`` entry."""

    regex: "re.Pattern[str]"
    timing_package: Optional[str]
    pattern: str


_PRECEDENCE_ENTRY_KEYS = frozenset({"pattern", "timing_package"})


def _normalize_precedence(
    entries: Sequence[Any], kind: str, release_name: str
) -> Tuple[_PrecedenceRule, ...]:
    """Compile a ``{par,tim}_precedence`` list into ordered rules.

    Entries are either a regex string or a ``{"pattern", "timing_package"}`` dict.

    Raises:
        ValueError: If an entry has the wrong type, an unknown key, no pattern,
            an unknown timing package, or an uncompilable regex.
    """
    rules: List[_PrecedenceRule] = []
    for position, entry in enumerate(entries):
        where = f"{release_name!r} {kind}_precedence[{position}]"
        if isinstance(entry, str):
            pattern, timing_package = entry, None
        elif isinstance(entry, dict):
            unknown = set(entry) - _PRECEDENCE_ENTRY_KEYS
            if unknown:
                raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
            if "pattern" not in entry:
                raise ValueError(f"{where}: missing required key 'pattern'")
            pattern = entry["pattern"]
            timing_package = entry.get("timing_package")
            if timing_package is not None and timing_package not in _TIMING_PACKAGES:
                raise ValueError(
                    f"{where}: invalid timing_package {timing_package!r}. "
                    f"Must be one of {list(_TIMING_PACKAGES)}"
                )
        else:
            raise ValueError(
                f"{where}: expected a regex string or a dict, got {type(entry).__name__}"
            )
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{where}: invalid regex {pattern!r}: {exc}")
        rules.append(_PrecedenceRule(regex, timing_package, pattern))
    return tuple(rules)


def _normalize_overrides(
    overrides: Any, kind: str, release_name: str
) -> Dict[str, str]:
    """Validate a ``{par,tim}_overrides`` mapping of pulsar name to relative path."""
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError(
            f"{release_name!r} {kind}_overrides: expected a dict, "
            f"got {type(overrides).__name__}"
        )
    for pulsar_name, relative_path in overrides.items():
        if not isinstance(pulsar_name, str) or not isinstance(relative_path, str):
            raise ValueError(
                f"{release_name!r} {kind}_overrides: expected str -> str, got "
                f"{type(pulsar_name).__name__} -> {type(relative_path).__name__}"
            )
    return dict(overrides)


def _precedence_rank(
    path: Path, rules: Sequence[_PrecedenceRule], timing_package: str
) -> int:
    """Index of the first applicable rule matching ``path``, else ``len(rules)``."""
    text = path.as_posix()
    for rank, rule in enumerate(rules):
        if rule.timing_package is not None and rule.timing_package != timing_package:
            continue
        if rule.regex.search(text):
            return rank
    return len(rules)


def _ambiguous_message(
    release_name: str,
    kind: str,
    pulsar_name: str,
    base_path: Path,
    tied: Sequence[Path],
    rank: int,
    rules: Sequence[_PrecedenceRule],
) -> str:
    listed = "\n".join(f"  {p.relative_to(base_path).as_posix()}" for p in tied)
    ranked_by = (
        f"precedence rank {rank}"
        if rank < len(rules)
        else "no matching precedence rule"
    )
    return (
        f"Release {release_name!r} matches {len(tied)} {kind} files for pulsar "
        f"{pulsar_name} with equal precedence ({ranked_by}):\n{listed}\n"
        f"Resolve with a {kind}_precedence entry that ranks one above the others, or "
        f"{kind}_overrides={{{pulsar_name!r}: '<release-relative path>'}}."
    )


def select_release_file(
    candidates: Sequence[Path],
    *,
    pulsar_name: str,
    kind: str,
    release_name: str,
    base_path: Path,
    rules: Sequence[_PrecedenceRule],
    override: Optional[str],
    timing_package: str,
) -> Tuple[Path, Dict[str, Any]]:
    """Choose one release file for one pulsar and describe the choice.

    Args:
        candidates: Pattern-matched paths for this pulsar, sorted, possibly empty
            when the pulsar was seeded by an override.
        pulsar_name: Canonical pulsar name the candidates were grouped under.
        kind: ``"par"`` or ``"tim"`` — used for messages and provenance only.
        release_name: Data release key, for messages.
        base_path: ``working_dir / base_dir``; overrides resolve against it.
        rules: Compiled precedence rules, in order.
        override: Release-relative path from ``{kind}_overrides``, or None.
        timing_package: The spec's timing package, matched by qualified rules.

    Returns:
        ``(chosen_path, provenance)``, where provenance is
        ``{"chosen", "candidates", "reason", "rule"}``. ``reason`` is ``"sole"``
        (one candidate, ``rule`` None), ``"precedence"`` (``rule`` is the matched
        entry's pattern) or ``"override"`` (``rule`` is the override string).

    Raises:
        MissingOverrideError: If ``override`` does not name an existing file.
        AmbiguousFileError: If several candidates tie at the best rank.
        FileSelectionError: If there is nothing to choose from.
    """
    provenance: Dict[str, Any] = {"chosen": None, "candidates": list(candidates)}

    if override is not None:
        chosen = base_path / override
        if not chosen.is_file():
            raise MissingOverrideError(
                f"Release {release_name!r} {kind}_overrides[{pulsar_name!r}] = "
                f"{override!r} does not exist (looked for {chosen})"
            )
        provenance.update(chosen=chosen, reason="override", rule=override)
        return chosen, provenance

    if not candidates:
        raise FileSelectionError(
            f"Release {release_name!r} has no {kind} candidates for pulsar "
            f"{pulsar_name} and no {kind}_overrides entry"
        )

    ranked = [(_precedence_rank(p, rules, timing_package), p) for p in candidates]
    best = min(rank for rank, _ in ranked)
    tied = [path for rank, path in ranked if rank == best]
    if len(tied) > 1:
        raise AmbiguousFileError(
            _ambiguous_message(
                release_name, kind, pulsar_name, base_path, tied, best, rules
            )
        )

    chosen = tied[0]
    if len(candidates) == 1:
        provenance.update(chosen=chosen, reason="sole", rule=None)
    else:
        # best < len(rules) is guaranteed: an unmatched candidate ranks len(rules),
        # so a best rank of len(rules) means every candidate tied and we raised above.
        provenance.update(chosen=chosen, reason="precedence", rule=rules[best].pattern)
    return chosen, provenance


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
            Format: {data_release_name: [{'par': parfile_path, 'tim': timfile_path, 'timing_package': 'pint', 'tim_metadata': TimMetadata(...), 'par_selection': {...}, 'tim_selection': {...}}, ...]}
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
                self.data_releases[data_release_name], data_release_name
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
                for entry in files:
                    for kind in ("par", "tim"):
                        selection = entry[f"{kind}_selection"]
                        if selection["reason"] == "sole":
                            continue
                        print(
                            f"      {kind}: {selection['chosen'].name} "
                            f"({selection['reason']} {selection['rule']!r}, "
                            f"{len(selection['candidates'])} candidates)"
                        )

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

        self._validate_data_release(data_release, name)
        self.data_releases[name] = data_release
        self.logger.debug(f"Added data release: {name}")

    def _validate_data_release(self, data_release: Dict, release_name: str) -> None:
        """Validate a data release dictionary.

        Args:
            data_release: Data release dictionary to validate
            release_name: Data release key, used in selection-key error messages

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

        for kind in ("par", "tim"):
            _normalize_precedence(
                data_release.get(f"{kind}_precedence") or (), kind, release_name
            )
            _normalize_overrides(
                data_release.get(f"{kind}_overrides"), kind, release_name
            )

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

    def _collect_candidates(
        self, base_path: Path, glob: str, pattern: str
    ) -> Dict[str, List[Path]]:
        """Group pattern-matching files under ``base_path`` by canonical pulsar name.

        Each group is sorted by POSIX path so selection, error text and provenance
        do not depend on filesystem iteration order.
        """
        regex = re.compile(pattern)
        candidates: Dict[str, List[Path]] = {}
        for path in base_path.rglob(glob):
            if not regex.search(path.as_posix()):
                continue
            try:
                pulsar_name = extract_pulsar_name_from_path(path)
            except ValueError:
                continue
            candidates.setdefault(pulsar_name, []).append(path)
        for paths in candidates.values():
            paths.sort(key=lambda p: p.as_posix())
        return candidates

    def _discover_all_file_pairs_in_data_release(
        self, data_release: Dict, release_name: str
    ) -> List[Dict[str, Any]]:
        """Discover one par/tim pair per pulsar in a data release.

        Files are grouped by canonical pulsar name, then reduced to a single file
        per kind by ``{par,tim}_overrides`` and ``{par,tim}_precedence``. A pulsar
        contributes a pair only when both kinds resolve.

        Raises:
            AmbiguousFileError: If a release matches several equally-ranked files.
            MissingOverrideError: If an override names a file that is not on disk.
        """
        base_path = self.working_dir / data_release["base_dir"]
        if not base_path.exists():
            return []

        timing_package = data_release["timing_package"]
        picked: Dict[str, Dict[str, Tuple[Path, Dict[str, Any]]]] = {}

        for kind, glob in (("par", "*.par"), ("tim", "*.tim")):
            candidates = self._collect_candidates(
                base_path, glob, data_release[f"{kind}_pattern"]
            )
            overrides = _normalize_overrides(
                data_release.get(f"{kind}_overrides"), kind, release_name
            )
            rules = _normalize_precedence(
                data_release.get(f"{kind}_precedence") or (), kind, release_name
            )
            # An override may name a file the pattern does not match, so the pulsar
            # can be absent from the pattern candidates entirely.
            for pulsar_name in overrides:
                candidates.setdefault(pulsar_name, [])

            for pulsar_name, paths in candidates.items():
                picked.setdefault(pulsar_name, {})[kind] = select_release_file(
                    paths,
                    pulsar_name=pulsar_name,
                    kind=kind,
                    release_name=release_name,
                    base_path=base_path,
                    rules=rules,
                    override=overrides.get(pulsar_name),
                    timing_package=timing_package,
                )

        file_pairs: List[Dict[str, Any]] = []
        for pulsar_name in sorted(picked):
            selection = picked[pulsar_name]
            if "par" not in selection or "tim" not in selection:
                continue
            par_file, par_selection = selection["par"]
            tim_file, tim_selection = selection["tim"]
            file_pairs.append(
                {
                    "par": par_file,
                    "tim": tim_file,
                    "timing_package": timing_package,
                    "tim_metadata": self._get_tim_metadata(tim_file),
                    "par_content": par_file.read_text(encoding="utf-8"),
                    "par_selection": par_selection,
                    "tim_selection": tim_selection,
                }
            )

        return file_pairs

    def _get_tim_metadata(self, tim_file_path: Path) -> TimMetadata:
        """Extract unified TIM metadata using the shared analyzer cache."""
        return self._tim_analyzer.get_tim_metadata(tim_file_path)


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
