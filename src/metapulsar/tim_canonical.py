"""Canonical .tim writer: flatten INCLUDEs and stamp MetaPulsar metadata flags.

MetaPulsar always hands its timing engines a standalone Tempo2 ``FORMAT 1``
file carrying authoritative ``-pta``, ``-pta_dataset`` and ``-timing_package``
flags, so the PTA identity of every TOA travels with the data instead of being
synthesized in memory. This module owns that text transformation; it never
changes TOA values, uncertainties, or any other flag.

No PINT dependency at import time (the legacy-format converter imports its
backend lazily).
"""

from __future__ import annotations

import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from loguru import logger

# Flags MetaPulsar owns. An input that already uses one of these names has it
# renamed to ``<name>_orig`` so the release's own value stays auditable.
CANONICAL_METADATA_FLAGS: Tuple[str, ...] = ("pta", "pta_dataset", "timing_package")

# tempo2 hard limits (ref-packages/tempo2/tempo2.h:96-97). Overflow makes
# tempo2 call exit(1) (readTimfile.C:323-327), which would kill the
# interpreter under in-process libstempo, so we refuse before writing.
TEMPO2_MAX_FLAGS = 40
TEMPO2_MAX_FLAG_LEN = 32
# readTimfile.C reads with fgets(line, 1000): 998 content bytes plus newline is
# the longest complete line it can consume in one call.
TEMPO2_MAX_TIM_LINE_BYTES = 998

_DIRECTIVE_NAMES = frozenset(
    {
        "MODE",
        "JUMP",
        "EFAC",
        "EQUAD",
        "EMAX",
        "EMIN",
        "ESET",
        "FMAX",
        "FMIN",
        "SIGMA",
        "TIME",
        "PHASE",
        "PHA1",
        "PHA2",
        "INFO",
        "SKIP",
        "NOSKIP",
        "SEARCH",
        "TRACK",
        "ZAWGT",
        "END",
        "EMAP",
        "FORMAT",
        "INCLUDE",
        "GLOBAL_EFAC",
        "EFLOOR",
        "PROFILE_DIR",
        "SIM",
        "DILATEFREQ",
        "NOSEARCH",
    }
)

# Tempo2 resets these function-local values for every recursive readTim() call.
# Flattening is safe only when they are at their defaults at an INCLUDE
# boundary. PINT deliberately shares its command dictionary across INCLUDEs, so
# this guard is tempo2-only: flattening already reproduces PINT's behavior.
_TEMPO2_ASSIGNING_DEFAULTS = {
    "EFAC": Decimal("1"),
    "EFLOOR": Decimal("-1"),
    "EQUAD": Decimal("0"),
    "SIGMA": Decimal("0"),
    "EMIN": Decimal("-1"),
    "EMAX": Decimal("-1"),
    "ESET": Decimal("-1"),
    "FMIN": Decimal("-1"),
    "FMAX": Decimal("-1"),
}

_TOKEN_RE = re.compile(r"\S+")


class TimCanonicalizationError(ValueError):
    """A .tim file cannot be turned into a canonical standalone FORMAT 1 file."""


class TimLegacyFormatError(TimCanonicalizationError):
    """A .tim file holds TOAs in a legacy (non-FORMAT 1) layout.

    Per-TOA flags only exist in Tempo2 FORMAT 1, so such input must first be
    converted by its own timing package (see ``convert_legacy_tim_to_format1``).
    """


class TimIncludeScopeError(TimCanonicalizationError):
    """A stateful directive is live across an INCLUDE boundary.

    Flattening such a file would change which TOAs the directive applies to on
    a tempo2 leg, so MetaPulsar refuses instead of writing shifted TOAs.
    """


class _ScopeState:
    """Tracks stateful tim directives so INCLUDE boundaries can be validated."""

    def __init__(self) -> None:
        self.time = Decimal("0")
        self.assignments = dict(_TEMPO2_ASSIGNING_DEFAULTS)
        self.invalid: set[str] = set()
        self.profile_dir: Optional[str] = None
        self.skipping = False

    def observe(self, directive: str, tokens: List[str]) -> None:
        if directive == "TIME":
            if len(tokens) >= 2:
                try:
                    self.time += Decimal(tokens[1])
                except InvalidOperation:
                    self.invalid.add(directive)
        elif directive in _TEMPO2_ASSIGNING_DEFAULTS:
            if len(tokens) >= 2:
                try:
                    self.assignments[directive] = Decimal(tokens[1])
                    self.invalid.discard(directive)
                except InvalidOperation:
                    self.invalid.add(directive)
        elif directive == "PROFILE_DIR":
            self.profile_dir = " ".join(tokens[1:]) if len(tokens) >= 2 else ""
        elif directive == "SKIP":
            self.skipping = True
        elif directive == "NOSKIP":
            self.skipping = False

    def live_directives(self) -> List[str]:
        """Return directives whose state would leak across an INCLUDE boundary."""
        live = ["TIME"] if self.time != 0 else []
        live.extend(
            name
            for name, default in _TEMPO2_ASSIGNING_DEFAULTS.items()
            if self.assignments[name] != default
        )
        live.extend(sorted(self.invalid))
        if self.profile_dir is not None:
            live.append("PROFILE_DIR")
        if self.skipping:
            live.append("SKIP")
        return live


def _read_lines(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TimCanonicalizationError(f"Cannot read .tim file {path}: {exc}") from exc
    return text.splitlines()


def _classify(line: str) -> Tuple[str, List[str]]:
    """Return ``(kind, tokens)`` where kind is comment/blank/directive/data."""
    stripped = line.strip()
    if not stripped:
        return "blank", []
    if stripped.startswith("#"):
        return "comment", []
    upper = stripped.upper()
    if upper == "C" or upper.startswith("C "):
        return "comment", []
    tokens = stripped.split()
    name = tokens[0].upper()
    if name in _DIRECTIVE_NAMES or name.startswith("T2E") or name.startswith("TNE"):
        return "directive", tokens
    return "data", tokens


def flatten_tim(
    tim_path: Path, *, timing_package: Literal["pint", "tempo2"] = "tempo2"
) -> str:
    """Return standalone Tempo2 FORMAT 1 text for ``tim_path``.

    INCLUDE directives are inlined in place so line order, repeated includes,
    comments, and every other directive survive unchanged. TOA lines are copied
    verbatim, preserving their exact MJD strings.

    Raises:
        TimLegacyFormatError: A TOA appears without ``FORMAT 1`` in effect.
        TimIncludeScopeError: A stateful directive is live across an INCLUDE.
        TimCanonicalizationError: Missing or circular INCLUDE, or unreadable file.
    """
    root = Path(tim_path).resolve()
    out: List[str] = ["FORMAT 1"]
    _flatten_into(root, out=out, stack=[], timing_package=timing_package)
    return "\n".join(out) + "\n"


def _flatten_into(
    path: Path,
    *,
    out: List[str],
    stack: List[Path],
    timing_package: Literal["pint", "tempo2"],
) -> bool:
    if path in stack:
        chain = " -> ".join(str(p) for p in [*stack, path])
        raise TimCanonicalizationError(f"Circular INCLUDE detected: {chain}")

    stack.append(path)
    # Both PINT and tempo2 start each included file with its own FORMAT and
    # directive state, so we do the same rather than inheriting the parent's.
    tokenized = False
    scope = _ScopeState()
    try:
        for line in _read_lines(path):
            kind, tokens = _classify(line)

            if kind in ("blank", "comment"):
                out.append(line)
                continue

            if kind == "data":
                if not tokenized:
                    raise TimLegacyFormatError(
                        f"{path} holds TOAs without 'FORMAT 1' in effect; per-TOA "
                        f"flags require Tempo2 FORMAT 1. Convert this file with its "
                        f"own timing package first. Offending line: {line.strip()!r}"
                    )
                out.append(line)
                continue

            directive = tokens[0].upper()

            # Tempo2 ignores every directive except NOSKIP while skipping. In
            # particular, an unreachable INCLUDE must not be opened and an END
            # must not terminate the file. PINT intentionally processes command
            # lines before applying SKIP, so this is engine-specific.
            if timing_package == "tempo2" and scope.skipping and directive != "NOSKIP":
                if directive != "INCLUDE":
                    out.append(line)
                continue

            if directive == "FORMAT":
                # The flattened file declares FORMAT 1 once, in its header.
                tokenized = len(tokens) >= 2 and tokens[1] == "1"
                if not tokenized:
                    raise TimLegacyFormatError(
                        f"{path} declares unsupported {' '.join(tokens)!r}; only "
                        f"'FORMAT 1' can carry per-TOA flags."
                    )
                continue

            if directive == "END":
                # Tempo2's END is local to each recursive readTim() call. PINT
                # shares END through its command dictionary, so an included END
                # terminates the whole top-level read there.
                if timing_package == "tempo2" and len(stack) > 1:
                    live = scope.live_directives()
                    if live:
                        raise TimIncludeScopeError(
                            f"Included file {path} reaches END with "
                            f"{', '.join(live)} still active. Flattening would "
                            f"leak that tempo2 file-local state into the parent."
                        )
                    return False
                out.append(line)
                return True

            if directive == "INCLUDE":
                if len(tokens) < 2:
                    raise TimCanonicalizationError(
                        f"INCLUDE without a filename in {path}"
                    )
                live = scope.live_directives()
                if timing_package == "tempo2" and live:
                    raise TimIncludeScopeError(
                        f"{path} reaches an INCLUDE with {', '.join(live)} still "
                        f"active. tempo2 scopes these per included file while PINT "
                        f"leaks them into it, so flattening would change which TOAs "
                        f"they apply to. Balance the directive around the INCLUDE."
                    )
                include_path = (path.parent / tokens[1]).resolve()
                if not include_path.is_file():
                    raise TimCanonicalizationError(
                        f"INCLUDE file not found: {include_path} (from {path})"
                    )
                logger.debug(f"Flattening included TOA file {include_path}")
                stop_all = _flatten_into(
                    include_path,
                    out=out,
                    stack=stack,
                    timing_package=timing_package,
                )
                if stop_all:
                    return True
                continue

            scope.observe(directive, tokens)
            out.append(line)

        live = scope.live_directives()
        if timing_package == "tempo2" and live and len(stack) > 1:
            raise TimIncludeScopeError(
                f"Included file {path} ends with {', '.join(live)} still active. "
                f"tempo2 drops this state at the end of the file while flattening "
                f"would carry it into the parent's following TOAs. Balance the "
                f"directive inside the included file."
            )
        return False
    finally:
        stack.pop()


def _flag_key_indices(n_tokens: int) -> List[int]:
    """Return token indices holding flag keys (FORMAT 1: flags start at index 5).

    Values may themselves start with '-' (e.g. ``-to -0.897e-6``), so pairs are
    walked positionally rather than by looking for leading dashes.
    """
    return list(range(5, n_tokens - 1, 2))


def _validate_flag_text(kind: str, text: str) -> None:
    if not text or any(ch.isspace() for ch in text):
        raise TimCanonicalizationError(
            f"Canonical tim flag {kind} must be a non-empty whitespace-free string, "
            f"got {text!r}"
        )
    byte_length = len(text.encode("utf-8"))
    if byte_length > TEMPO2_MAX_FLAG_LEN - 1:
        raise TimCanonicalizationError(
            f"Canonical tim flag {kind} {text!r} is {byte_length} encoded bytes; "
            f"tempo2 truncates at {TEMPO2_MAX_FLAG_LEN - 1} (MAX_FLAG_LEN)."
        )


def stamp_metadata_flags(tim_text: str, *, pta_name: str, timing_package: str) -> str:
    """Return ``tim_text`` with authoritative MetaPulsar metadata flags on each TOA.

    Existing ``pta``/``pta_dataset``/``timing_package`` flags are matched
    case-insensitively and every occurrence is renamed to ``<name>_orig``,
    keeping its raw value. PINT lowercases flag keys and keeps only the last
    duplicate while tempo2 is case-sensitive and keeps them all, so matching
    case-insensitively is what keeps both engines seeing the same thing.

    Raises:
        TimCanonicalizationError: A ``<name>_orig`` flag already exists, a value
            is unusable, or stamping would exceed tempo2's per-TOA flag limits.
    """
    values = {
        "pta": pta_name,
        "pta_dataset": pta_name,
        "timing_package": timing_package,
    }
    for name, value in values.items():
        _validate_flag_text(f"-{name} value", value)
        _validate_flag_text("name", name)

    out: List[str] = []
    tempo2_info_active = False
    tempo2_skipping = False
    for line in tim_text.splitlines():
        kind, tokens = _classify(line)
        if timing_package == "tempo2" and kind == "directive":
            directive = tokens[0].upper()
            if directive == "SKIP":
                tempo2_skipping = True
            elif directive == "NOSKIP":
                tempo2_skipping = False
            elif directive == "INFO" and not tempo2_skipping:
                # Tempo2 materializes INFO as an implicit per-TOA -i flag.
                tempo2_info_active = len(tokens) >= 2 and tokens[1] != "-1"
        if kind != "data" or len(tokens) < 5:
            out.append(line)
            continue
        out.append(
            _stamp_toa_line(
                line,
                values=values,
                implicit_flag_count=int(tempo2_info_active),
            )
        )
    return "\n".join(out) + "\n"


def _stamp_toa_line(
    line: str, *, values: Dict[str, str], implicit_flag_count: int = 0
) -> str:
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]
    tokens = [line[start:end] for start, end in spans]

    key_indices = _flag_key_indices(len(tokens))
    present: Dict[str, List[int]] = {}

    for index in key_indices:
        token = tokens[index]
        if not token.startswith("-"):
            continue
        name = token.lstrip("-").lower()
        present.setdefault(name, []).append(index)

    # Only the complete, exact lowercase trio identifies one of our artifacts.
    # A release flag that happens to have the same value (or uses different
    # case) is still release metadata and must become *_orig.
    already_canonical = all(
        len(present.get(name, [])) == 1
        and tokens[present[name][0]] == f"-{name}"
        and tokens[present[name][0] + 1] == values[name]
        for name in CANONICAL_METADATA_FLAGS
    )

    renames: List[Tuple[int, int, str]] = []
    to_append: List[str] = []
    if not already_canonical:
        for name in CANONICAL_METADATA_FLAGS:
            occurrences = present.get(name, [])
            if occurrences:
                if f"{name}_orig" in present:
                    raise TimCanonicalizationError(
                        f"Cannot preserve existing -{name} as -{name}_orig because "
                        f"-{name}_orig is already present on TOA line: "
                        f"{line.strip()!r}"
                    )
                for index in occurrences:
                    start, end = spans[index]
                    renames.append((start, end, f"-{name}_orig"))
            to_append.append(f" -{name} {values[name]}")

    # Tempo2 checks after incrementing and exits at nFlags == MAX_FLAGS, so
    # MAX_FLAGS itself is not a usable count.
    n_flags = len(key_indices) + len(to_append) + implicit_flag_count
    if n_flags >= TEMPO2_MAX_FLAGS:
        raise TimCanonicalizationError(
            f"Stamping MetaPulsar flags would give this TOA {n_flags} flags, "
            f"reaching tempo2's fatal MAX_FLAGS={TEMPO2_MAX_FLAGS}: "
            f"{line.strip()!r}"
        )

    stamped = line
    for start, end, replacement in reversed(renames):
        stamped = stamped[:start] + replacement + stamped[end:]
    stamped = stamped.rstrip("\r") + "".join(to_append)
    line_bytes = len(stamped.encode("utf-8"))
    if line_bytes > TEMPO2_MAX_TIM_LINE_BYTES:
        raise TimCanonicalizationError(
            f"Stamped TOA line is {line_bytes} bytes; tempo2 can safely read at "
            f"most {TEMPO2_MAX_TIM_LINE_BYTES} bytes before the newline: "
            f"{stamped.strip()!r}"
        )
    return stamped


def convert_legacy_tim_to_format1(
    par_text: str,
    tim_path: Path,
    *,
    timing_package: str,
    out_path: Path,
) -> Path:
    """Convert a legacy-format .tim to Tempo2 FORMAT 1 using its own package.

    Each dataset is parsed by the timing package it belongs to, so conversion
    uses that package rather than silently substituting the other one.
    """
    tim_path = Path(tim_path)
    out_path = Path(out_path)

    with tempfile.TemporaryDirectory(prefix="metapulsar_legacy_tim_") as td:
        par_tmp = Path(td) / "legacy.par"
        par_tmp.write_text(par_text, encoding="utf-8")

        if timing_package == "pint":
            try:
                from pint.toa import get_TOAs
            except ImportError as exc:  # pragma: no cover - PINT is a hard dep
                raise TimCanonicalizationError(
                    f"Converting legacy .tim {tim_path} requires PINT."
                ) from exc
            from .pint_helpers import create_pint_model

            model = create_pint_model(par_text)
            toas = get_TOAs(str(tim_path), model=model, include_pn=True)
            toas.write_TOA_file(str(out_path), format="Tempo2", include_pn=True)
        else:
            from .sandbox_tempo2 import tempopulsar

            psr = tempopulsar(parfile=str(par_tmp), timfile=str(tim_path), dofit=False)
            psr.savetim(str(out_path))

    if not out_path.is_file():
        raise TimCanonicalizationError(
            f"{timing_package} did not produce a FORMAT 1 .tim for {tim_path}"
        )
    logger.info(f"Converted legacy .tim {tim_path} to FORMAT 1 via {timing_package}")
    return out_path


def write_canonical_tim(
    tim_path: Path,
    *,
    pta_name: str,
    timing_package: str,
    out_path: Path,
    par_text: Optional[str] = None,
) -> Path:
    """Write the canonical standalone FORMAT 1 .tim MetaPulsar hands its engines.

    Args:
        tim_path: Source .tim (may use INCLUDE and may already carry ``-pn``).
        pta_name: MetaPulsar's PTA key, stamped as ``-pta`` and ``-pta_dataset``.
        timing_package: ``"pint"`` or ``"tempo2"``, stamped as ``-timing_package``.
        out_path: Destination path.
        par_text: Par content, required only to convert legacy-format input.

    Returns:
        ``out_path``.
    """
    tim_path = Path(tim_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    package: Literal["pint", "tempo2"] = (
        "pint" if timing_package == "pint" else "tempo2"
    )
    try:
        text = flatten_tim(tim_path, timing_package=package)
    except TimLegacyFormatError:
        if par_text is None:
            raise
        with tempfile.TemporaryDirectory(prefix="metapulsar_format1_") as td:
            converted = Path(td) / "format1.tim"
            convert_legacy_tim_to_format1(
                par_text,
                tim_path,
                timing_package=timing_package,
                out_path=converted,
            )
            text = flatten_tim(converted, timing_package=package)

    text = stamp_metadata_flags(text, pta_name=pta_name, timing_package=timing_package)
    out_path.write_text(text, encoding="utf-8")
    return out_path
