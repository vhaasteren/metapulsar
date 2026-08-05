"""TimFileAnalyzer - lightweight TIM file metadata parser.

Single source of truth for file-level .tim metadata: TOA count, MJD range,
timespan, and pulse-number (-pn) coverage. No PINT dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple
from loguru import logger

TimPulseNumberStatus = Literal["complete", "mixed", "none"]

# Known directive prefixes (uppercase first token)
_DIRECTIVE_PREFIXES = frozenset(
    {
        "MODE",
        "JUMP",
        "EFAC",
        "EQUAD",
        "TIME",
        "PHASE",
        "SKIP",
        "NOSKIP",
        "FORMAT",
        "INCLUDE",
    }
)


def is_tim_comment_line(line: str) -> bool:
    """True when a .tim line is a comment to both PINT and tempo2.

    Tempo2 comments out any FORMAT 1 line whose first character is ``C``; PINT
    honors ``C ``, ``c ``, ``CC `` and ``#``. The two-character ``CC`` marker
    matters: EPTA Effelsberg files reject TOAs with it, and a walker that only
    knows ``C `` reads those lines as data and shifts every field left by one.
    """
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    upper = stripped.upper()
    return upper in ("C", "CC") or upper.startswith(("C ", "CC "))


@dataclass(frozen=True)
class TimMetadata:
    """Structured metadata extracted from a .tim file (including INCLUDEs)."""

    toa_count: int
    mjd_min: Optional[float]
    mjd_max: Optional[float]
    timespan_days: float
    pn_with_count: int
    pn_without_count: int
    pn_status: TimPulseNumberStatus
    lines_seen: int = 0
    lines_skipped: int = 0
    parse_warnings: Tuple[str, ...] = ()


class _ParseAccumulator:
    """Mutable accumulator for a single metadata parse."""

    def __init__(self) -> None:
        self.toa_count = 0
        self.mjd_min: Optional[float] = None
        self.mjd_max: Optional[float] = None
        self.pn_with_count = 0
        self.pn_without_count = 0
        self.lines_seen = 0
        self.lines_skipped = 0
        self.warnings: List[str] = []

    def record_toa(self, mjd: float, has_pn: bool) -> None:
        self.toa_count += 1
        if self.mjd_min is None or mjd < self.mjd_min:
            self.mjd_min = mjd
        if self.mjd_max is None or mjd > self.mjd_max:
            self.mjd_max = mjd
        if has_pn:
            self.pn_with_count += 1
        else:
            self.pn_without_count += 1

    def skip_line(self) -> None:
        self.lines_skipped += 1

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_metadata(self) -> TimMetadata:
        if self.toa_count == 0:
            timespan = 0.0
            pn_status: TimPulseNumberStatus = "none"
        else:
            timespan = float(self.mjd_max - self.mjd_min)  # type: ignore[operator]
            if self.pn_with_count == 0:
                pn_status = "none"
            elif self.pn_without_count == 0:
                pn_status = "complete"
            else:
                pn_status = "mixed"
        return TimMetadata(
            toa_count=self.toa_count,
            mjd_min=self.mjd_min,
            mjd_max=self.mjd_max,
            timespan_days=timespan,
            pn_with_count=self.pn_with_count,
            pn_without_count=self.pn_without_count,
            pn_status=pn_status,
            lines_seen=self.lines_seen,
            lines_skipped=self.lines_skipped,
            parse_warnings=tuple(self.warnings),
        )


class TimFileAnalyzer:
    """Lightweight TIM file analyzer for unified metadata extraction.

    Parses .tim files recursively (including INCLUDE) with explicit FORMAT
    state tracking. Results are cached by resolved path and mtime.
    """

    def __init__(self) -> None:
        self.logger = logger
        self._file_cache: Dict[Path, Tuple[float, TimMetadata]] = {}

    def get_tim_metadata(self, tim_file_path: Path) -> TimMetadata:
        """Return unified metadata for a .tim file (cached by path + mtime)."""
        resolved = Path(tim_file_path).resolve()
        mtime = self._safe_mtime(resolved)

        cached = self._file_cache.get(resolved)
        if cached is not None and cached[0] == mtime:
            self.logger.debug(f"Using cached metadata for {resolved}")
            return cached[1]

        try:
            acc = _ParseAccumulator()
            active: Set[Path] = set()
            self._parse_file(resolved, format_tokenized=False, acc=acc, active=active)
            meta = acc.to_metadata()
            self._file_cache[resolved] = (mtime, meta)
            if meta.toa_count > 0:
                self.logger.debug(
                    f"Cached metadata for {resolved}: "
                    f"{meta.timespan_days:.1f} days, {meta.toa_count} TOAs, "
                    f"pn={meta.pn_status}"
                )
            else:
                self.logger.debug(f"Cached metadata for {resolved}: no TOAs found")
            return meta
        except Exception as e:
            self.logger.warning(f"Parsing failed for {resolved}: {e}")
            empty = TimMetadata(
                toa_count=0,
                mjd_min=None,
                mjd_max=None,
                timespan_days=0.0,
                pn_with_count=0,
                pn_without_count=0,
                pn_status="none",
                parse_warnings=(f"parse failed: {e}",),
            )
            self._file_cache[resolved] = (mtime, empty)
            return empty

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._file_cache.clear()
        self.logger.debug("Metadata cache cleared")

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0

    def _parse_file(
        self,
        tim_file_path: Path,
        *,
        format_tokenized: bool,
        acc: _ParseAccumulator,
        active: Set[Path],
    ) -> None:
        if tim_file_path in active:
            msg = f"Circular INCLUDE detected: {tim_file_path}"
            self.logger.warning(msg)
            acc.add_warning(msg)
            return

        active.add(tim_file_path)
        try:
            with open(tim_file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    acc.lines_seen += 1
                    format_tokenized = self._process_line(
                        stripped,
                        tim_file_path,
                        format_tokenized=format_tokenized,
                        acc=acc,
                        active=active,
                    )
        except OSError as e:
            msg = f"Error reading TIM file {tim_file_path}: {e}"
            self.logger.error(msg)
            acc.add_warning(msg)
        finally:
            active.discard(tim_file_path)

    def _process_line(
        self,
        line: str,
        current_file: Path,
        *,
        format_tokenized: bool,
        acc: _ParseAccumulator,
        active: Set[Path],
    ) -> bool:
        """Process one line; return updated format_tokenized state."""
        if is_tim_comment_line(line):
            return format_tokenized

        tokens = line.split()
        if not tokens:
            return format_tokenized

        first = tokens[0].upper()

        if first == "FORMAT":
            if len(tokens) >= 2 and tokens[1] == "1":
                return True
            return False

        if first == "INCLUDE":
            if len(tokens) < 2:
                msg = f"INCLUDE command without filename in {current_file}"
                self.logger.warning(msg)
                acc.add_warning(msg)
                acc.skip_line()
                return format_tokenized
            include_path = (current_file.parent / tokens[1]).resolve()
            if include_path.exists():
                self.logger.debug(f"Processing included TOA file {include_path}")
                self._parse_file(
                    include_path,
                    format_tokenized=format_tokenized,
                    acc=acc,
                    active=active,
                )
            else:
                msg = f"INCLUDE file not found: {include_path}"
                self.logger.warning(msg)
                acc.add_warning(msg)
            return format_tokenized

        if self._is_directive(tokens):
            return format_tokenized

        if format_tokenized:
            parsed = self._parse_tokenized_toa(tokens)
            if parsed is None:
                acc.skip_line()
                return format_tokenized
            mjd, has_pn = parsed
            acc.record_toa(mjd, has_pn)
            return format_tokenized

        parsed = self._parse_legacy_toa(line, tokens)
        if parsed is None:
            acc.skip_line()
            return format_tokenized
        mjd, has_pn = parsed
        acc.record_toa(mjd, has_pn)
        return format_tokenized

    @staticmethod
    def _is_directive(tokens: List[str]) -> bool:
        first = tokens[0].upper()
        if first in _DIRECTIVE_PREFIXES:
            return True
        if first.startswith("T2E") or first.startswith("TNE"):
            return True
        return False

    @staticmethod
    def _parse_tokenized_toa(tokens: List[str]) -> Optional[Tuple[float, bool]]:
        """Parse FORMAT 1 tokenized TOA: name freq mjd err site [flags...]."""
        if len(tokens) < 5:
            return None
        try:
            mjd = float(tokens[2])
        except ValueError:
            return None

        has_pn = False
        i = 5
        while i + 1 < len(tokens):
            flag_key = tokens[i].lstrip("-").lower()
            if flag_key == "pn":
                has_pn = True
            i += 2
        return mjd, has_pn

    @staticmethod
    def _parse_legacy_toa(line: str, tokens: List[str]) -> Optional[Tuple[float, bool]]:
        """Minimal legacy TOA support (Princeton / whitespace tokenized)."""
        # Princeton: first char alnum/@, tokens include freq and mjd
        if tokens[0][0].isalnum() or tokens[0][0] == "@":
            if len(tokens) >= 3:
                try:
                    return float(tokens[2]), False
                except ValueError:
                    pass

        # Parkes-like: decimal at column 42 (0-based index 41)
        if len(line) > 42 and line[41] == ".":
            mjd_str = line[10:27].strip()
            if mjd_str:
                try:
                    return float(mjd_str), False
                except ValueError:
                    pass

        return None
