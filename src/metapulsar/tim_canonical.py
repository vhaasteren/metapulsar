"""Canonical .tim writer: flatten INCLUDEs, bake TIME, and stamp metadata flags.

MetaPulsar always hands its timing engines a standalone Tempo2 ``FORMAT 1``
file carrying authoritative ``-pta``, ``-pta_dataset`` and ``-timing_package``
flags, so the PTA identity of every TOA travels with the data instead of being
synthesized in memory. Cumulative ``TIME`` offsets are baked into TOA MJDs with
exact decimal arithmetic (``sat += TIME / 86400``, rounded once at output — not
Tempo2 ``double``/``longdouble`` bit-equivalence). ``TIME`` and ``MODE`` lines
are omitted from the artifact; effective ``MODE`` is discovered from the release
tim tree and transferred onto the engine-facing ``.par``. Every TOA name is
rewritten to a safe ``toaNNNNN`` token. When the release par contains
``JUMP MJD`` windows this module also stamps combination-safe ``-mjd_jump_pta``
flags on the selected (post-bake) TOAs.

No PINT dependency at import time (the legacy-format converter imports its
backend lazily).
"""

from __future__ import annotations

import re
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

# Flags MetaPulsar owns. An input that already uses one of these names has it
# renamed to ``<name>_orig`` so the release's own value stays auditable.
CANONICAL_METADATA_FLAGS: Tuple[str, ...] = ("pta", "pta_dataset", "timing_package")
MJD_JUMP_PTA_FLAG = "mjd_jump_pta"

# tempo2 hard limits (ref-packages/tempo2/tempo2.h:96-97). Overflow makes
# tempo2 call exit(1) (readTimfile.C:323-327), which would kill the
# interpreter under in-process libstempo, so we refuse before writing.
TEMPO2_MAX_FLAGS = 40
TEMPO2_MAX_FLAG_LEN = 32
# readTimfile.C reads with fgets(line, 1000): 998 content bytes plus newline is
# the longest complete line it can consume in one call.
TEMPO2_MAX_TIM_LINE_BYTES = 998

_SECDAY = Fraction(86400)
_MIN_MJD_FRAC_DIGITS = 17
_MAX_FRAC_DIGITS = 40
_MAX_TIME_ABS_SECONDS = Decimal("1e9")
_MIN_MJD = Decimal(0)
_MAX_MJD = Decimal("1e6")
_MIN_MJD_EXACT = Fraction(_MIN_MJD)
_MAX_MJD_EXACT = Fraction(_MAX_MJD)
_PAR_MODE_ALIASES = frozenset({"MODE", "WEIGHT"})
_PRINCETON_RE = re.compile(r"[0-9a-z@] ")
_TWO_NONSPACE_RE = re.compile(r"\S\S")

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


@dataclass(frozen=True)
class FlattenTimResult:
    """Standalone FORMAT 1 text plus the last observed ``MODE`` in the tree."""

    text: str
    effective_mode: Optional[int]


@dataclass(frozen=True)
class CanonicalTimResult:
    """Path of a written canonical ``.tim`` plus diagnostic ``MODE`` from that file."""

    path: Path
    effective_mode: Optional[int]


@dataclass
class _TimeAccum:
    """Exact cumulative TIME offset in seconds."""

    total: Fraction = Fraction(0)

    def add(self, delta: Fraction) -> None:
        self.total += delta


class _ScopeState:
    """Tracks stateful tim directives so INCLUDE boundaries can be validated.

    ``TIME`` liveness is supplied by the walker's exact accumulator, not stored
    here.
    """

    def __init__(self) -> None:
        self.assignments = dict(_TEMPO2_ASSIGNING_DEFAULTS)
        self.invalid: set[str] = set()
        self.profile_dir: Optional[str] = None
        self.skipping = False

    def observe(self, directive: str, tokens: List[str]) -> None:
        if directive in _TEMPO2_ASSIGNING_DEFAULTS:
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

    def live_directives(self, *, time_live: bool) -> List[str]:
        """Return directives whose state would leak across an INCLUDE boundary."""
        live = ["TIME"] if time_live else []
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


def _parse_bounded_decimal(
    token: str, *, what: str, minimum: Decimal, maximum: Decimal
) -> Fraction:
    """Parse a finite, in-range decimal literal into an exact Fraction.

    Bounds are checked on the Decimal *before* Fraction conversion: a compact
    token such as ``1e999999999`` is finite but would materialize a ~10**9-digit
    integer. Decimal comparisons at those exponents are cheap and do not expand.
    """
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise TimCanonicalizationError(f"Invalid {what}: {token!r}") from exc
    if not value.is_finite():
        raise TimCanonicalizationError(f"Non-finite {what}: {token!r}")
    if not (minimum <= value <= maximum):
        raise TimCanonicalizationError(
            f"{what} out of range [{minimum}, {maximum}]: {token!r}"
        )
    exponent = value.as_tuple().exponent
    # A tiny magnitude passes the range check, so guard the exponent floor too.
    if isinstance(exponent, int) and exponent < -_MAX_FRAC_DIGITS:
        raise TimCanonicalizationError(
            f"{what} needs more than {_MAX_FRAC_DIGITS} fractional digits: {token!r}"
        )
    return Fraction(value)


def _parse_time_offset(token: str) -> Fraction:
    return _parse_bounded_decimal(
        token,
        what="TIME offset",
        minimum=-_MAX_TIME_ABS_SECONDS,
        maximum=_MAX_TIME_ABS_SECONDS,
    )


def _frac_digits_for_mjd_token(token: str) -> int:
    """Digits to emit, from the Decimal exponent (never str.split('.'))."""
    exponent = Decimal(token).as_tuple().exponent
    explicit = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return min(max(explicit, _MIN_MJD_FRAC_DIGITS), _MAX_FRAC_DIGITS)


def _format_fraction(value: Fraction, digits: int) -> str:
    """Render an exact Fraction in fixed point with one half-even rounding."""
    if digits > _MAX_FRAC_DIGITS:
        raise TimCanonicalizationError(
            f"Refusing to format {digits} fractional digits (max {_MAX_FRAC_DIGITS})"
        )
    scale = 10**digits
    scaled = value * scale
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (
        twice == scaled.denominator and quotient % 2 == 1
    ):
        quotient += 1
    sign = "-" if quotient < 0 else ""
    quotient = abs(quotient)
    integral, fractional = divmod(quotient, scale)
    if digits == 0:
        return f"{sign}{integral}"
    return f"{sign}{integral}.{str(fractional).zfill(digits)}"


def _bake_mjd_token(mjd_token: str, accum: _TimeAccum) -> str:
    mjd = _parse_bounded_decimal(
        mjd_token, what="TOA MJD token", minimum=_MIN_MJD, maximum=_MAX_MJD
    )
    if accum.total == 0:
        return mjd_token  # validated above; preserve exact source bytes
    baked = mjd + accum.total / _SECDAY
    # In-range tokens can still bake out of range (e.g. MJD 0.5 with TIME -1e9).
    if not (_MIN_MJD_EXACT <= baked <= _MAX_MJD_EXACT):
        raise TimCanonicalizationError(
            f"Baked TOA MJD {baked} out of range [{_MIN_MJD}, {_MAX_MJD}], "
            f"from source {mjd_token!r} and cumulative TIME {accum.total} s"
        )
    return _format_fraction(baked, _frac_digits_for_mjd_token(mjd_token))


def _pint_legacy_heuristic_hit(line: str) -> bool:
    """True if PINT's _toa_format would classify this Princeton/Parkes/ITOA."""
    if _PRINCETON_RE.match(line):
        return True
    if line.startswith(" ") and len(line) > 41 and line[41] == ".":
        return True
    if _TWO_NONSPACE_RE.match(line) and len(line) > 14 and line[14] == ".":
        return True
    return False


def _render_canonical_toa_line(
    line: str, *, tokens: List[str], accum: _TimeAccum, name_counter: List[int]
) -> str:
    if len(tokens) < 5:
        raise TimCanonicalizationError(
            f"FORMAT 1 TOA line has too few tokens: {line.strip()!r}"
        )
    name_counter[0] += 1
    name = f"toa{name_counter[0]:05d}"
    mjd = _bake_mjd_token(tokens[2], accum)
    flags = (" " + " ".join(tokens[5:])) if len(tokens) > 5 else ""
    rebuilt = f" {name} {tokens[1]} {mjd} {tokens[3]} {tokens[4]}{flags}"
    if _pint_legacy_heuristic_hit(rebuilt):
        raise TimCanonicalizationError(
            f"Canonical TOA line still hits a PINT legacy heuristic: {rebuilt!r}"
        )
    return rebuilt


class _TimWalker:
    """The one traversal used by both flatten and mode discovery.

    ``emit=False`` discards rendered lines but performs identical parsing,
    validation and state transitions. ``on_legacy_toa="skip"`` tolerates legacy
    input -- both an explicit non-``1`` ``FORMAT`` declaration and untagged TOA
    lines -- which ``write_canonical_tim`` supports by converting through the
    owning engine; directive processing (MODE, TIME, INCLUDE) continues so
    release metadata is still discovered.
    """

    def __init__(
        self,
        *,
        timing_package: Literal["pint", "tempo2"],
        emit: bool,
        on_legacy_toa: Literal["raise", "skip"] = "raise",
    ):
        self.timing_package = timing_package
        self.emit = emit
        self.on_legacy_toa = on_legacy_toa
        self.out: List[str] = ["FORMAT 1"] if emit else []
        self.name_counter: List[int] = [0]
        self.effective_mode: Optional[int] = None
        self.shared_time = _TimeAccum()  # PINT shares; tempo2 is file-local

    def _write(self, line: str) -> None:
        if self.emit:
            self.out.append(line)

    def walk(self, path: Path, stack: List[Path]) -> bool:
        """Walk one file. Returns True when a top-level END stops everything."""
        if path in stack:
            chain = " -> ".join(str(p) for p in [*stack, path])
            raise TimCanonicalizationError(f"Circular INCLUDE detected: {chain}")

        stack.append(path)
        tokenized = False
        scope = _ScopeState()
        accum = self.shared_time if self.timing_package == "pint" else _TimeAccum()
        try:
            for line in _read_lines(path):
                kind, tokens = _classify(line)

                if kind in ("blank", "comment"):
                    self._write(line)
                    continue

                if kind == "data":
                    if not tokenized:
                        if self.on_legacy_toa == "skip":
                            continue  # converter will handle this file
                        raise TimLegacyFormatError(
                            f"{path} holds TOAs without 'FORMAT 1' in effect; per-TOA "
                            f"flags require Tempo2 FORMAT 1. Convert this file with its "
                            f"own timing package first. Offending line: {line.strip()!r}"
                        )
                    self._write(
                        _render_canonical_toa_line(
                            line,
                            tokens=tokens,
                            accum=accum,
                            name_counter=self.name_counter,
                        )
                    )
                    continue

                directive = tokens[0].upper()

                if (
                    self.timing_package == "tempo2"
                    and scope.skipping
                    and directive != "NOSKIP"
                ):
                    if directive not in ("INCLUDE", "TIME", "MODE"):
                        self._write(line)
                    continue

                if directive == "FORMAT":
                    # A FORMAT line with no argument is malformed, not legacy:
                    # it is never convertible, so both entry points reject it.
                    if len(tokens) < 2:
                        raise TimCanonicalizationError(
                            f"FORMAT without a value in {path}: {line.strip()!r}"
                        )
                    tokenized = tokens[1] == "1"
                    if not tokenized:
                        # Any other value is a legacy declaration (tempo2 never
                        # reads the value; PINT honors only "1"). Route it
                        # through the same switch as untagged TOA lines so
                        # discovery does not preempt engine conversion.
                        if self.on_legacy_toa == "raise":
                            raise TimLegacyFormatError(
                                f"{path} declares unsupported {' '.join(tokens)!r}; "
                                f"only 'FORMAT 1' can carry per-TOA flags."
                            )
                    continue

                if directive == "END":
                    if self.timing_package == "tempo2" and len(stack) > 1:
                        live = scope.live_directives(time_live=accum.total != 0)
                        if live:
                            raise TimIncludeScopeError(
                                f"Included file {path} reaches END with "
                                f"{', '.join(live)} still active. Flattening would "
                                f"leak that tempo2 file-local state into the parent."
                            )
                        return False
                    self._write(line)
                    return True

                if directive == "INCLUDE":
                    if len(tokens) < 2:
                        raise TimCanonicalizationError(
                            f"INCLUDE without a filename in {path}"
                        )
                    live = scope.live_directives(time_live=accum.total != 0)
                    if self.timing_package == "tempo2" and live:
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
                    if self.walk(include_path, stack):
                        return True
                    continue

                if directive == "TIME":
                    if len(tokens) < 2:
                        raise TimCanonicalizationError(
                            f"TIME without offset in {path}: {line.strip()!r}"
                        )
                    accum.add(_parse_time_offset(tokens[1]))
                    continue  # never emit

                if directive == "MODE":
                    if len(tokens) < 2:
                        raise TimCanonicalizationError(
                            f"MODE without value in {path}: {line.strip()!r}"
                        )
                    try:
                        self.effective_mode = int(tokens[1], 10)
                    except ValueError as exc:
                        raise TimCanonicalizationError(
                            f"Invalid MODE value in {path}: {line.strip()!r}"
                        ) from exc
                    continue  # never emit

                scope.observe(directive, tokens)
                self._write(line)

            live = scope.live_directives(time_live=accum.total != 0)
            if self.timing_package == "tempo2" and live and len(stack) > 1:
                raise TimIncludeScopeError(
                    f"Included file {path} ends with {', '.join(live)} still active. "
                    f"tempo2 drops this state at the end of the file while flattening "
                    f"would carry it into the parent's following TOAs. Balance the "
                    f"directive inside the included file."
                )
            return False
        finally:
            stack.pop()


def flatten_tim(
    tim_path: Path, *, timing_package: Literal["pint", "tempo2"] = "tempo2"
) -> FlattenTimResult:
    """Return standalone Tempo2 FORMAT 1 text for ``tim_path``.

    INCLUDE directives are inlined in place. Cumulative ``TIME`` offsets are
    baked into TOA MJDs with exact decimal arithmetic and omitted from the
    output; ``MODE`` lines are omitted (their value is returned as
    ``effective_mode``). Every TOA name is rewritten to ``toaNNNNN``.

    Raises:
        TimLegacyFormatError: A TOA appears without ``FORMAT 1`` in effect, or a
            non-``1`` ``FORMAT`` declaration is present.
        TimIncludeScopeError: A stateful directive is live across an INCLUDE.
        TimCanonicalizationError: Missing or circular INCLUDE, unreadable file,
            or invalid observed ``TIME``/``MODE``/MJD.
    """
    walker = _TimWalker(timing_package=timing_package, emit=True, on_legacy_toa="raise")
    walker.walk(Path(tim_path).resolve(), [])
    return FlattenTimResult(
        text="\n".join(walker.out) + "\n", effective_mode=walker.effective_mode
    )


def discover_effective_tim_mode(
    tim_path: Path, *, timing_package: Literal["pint", "tempo2"] = "tempo2"
) -> Optional[int]:
    """Last observed .tim MODE, using the same traversal as canonicalization.

    Runs the same structural directive traversal as :func:`flatten_tim` --
    INCLUDE resolution and circularity, SKIP/END semantics, TIME parsing and
    tempo2 scope guards, FORMAT 1 TOA validation -- and raises the same errors
    for FORMAT 1 input. Legacy input (a non-``1`` ``FORMAT`` declaration, or TOAs
    with no ``FORMAT 1`` in effect) is tolerated rather than rejected, because
    ``write_canonical_tim`` supports it by converting through the owning engine;
    directives are still processed so release MODE is discovered before any lossy
    conversion or pulse-number rewrite. Success here does *not* imply that the
    later conversion will succeed.

    Returns ``None`` when the tim tree asserts no MODE, meaning "keep whatever
    mode the engine par implies" -- not "mode 0".
    """
    walker = _TimWalker(timing_package=timing_package, emit=False, on_legacy_toa="skip")
    walker.walk(Path(tim_path).resolve(), [])
    return walker.effective_mode


def ensure_par_mode(par_text: str, mode: int) -> str:
    """Return par text whose only fit-mode assignment is ``MODE {mode}``.

    ``WEIGHT`` is a tempo2 alias for the same ``fitMode`` slot
    (``readParfile.C``), and tempo2 applies assignments in file order, so every
    alias line is dropped and the canonical line is appended last.
    """
    out = [
        raw
        for raw in par_text.splitlines()
        if not (raw.split() and raw.split()[0].upper() in _PAR_MODE_ALIASES)
    ]
    out.append(f"MODE {mode}")
    return "\n".join(out) + ("\n" if par_text.endswith("\n") else "")


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


def parse_jump_mjd_windows(
    par_text: str,
) -> List[Tuple[Decimal, Decimal, Tuple[str, ...]]]:
    """Return ordered ``(t1, t2, value_tokens)`` for every active ``JUMP MJD`` line.

    Skips blank/comment lines. A line is a JUMP MJD line iff, after strip+split,
    ``tokens[0].upper() == "JUMP"`` and ``tokens[1].upper() == "MJD"`` and
    ``len(tokens) >= 4``. ``SATJUMP`` is therefore excluded.
    """
    out: List[Tuple[Decimal, Decimal, Tuple[str, ...]]] = []
    for raw in par_text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        upper = stripped.upper()
        if upper == "C" or upper.startswith("C "):
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            continue
        if tokens[0].upper() != "JUMP" or tokens[1].upper() != "MJD":
            continue
        try:
            t1 = Decimal(tokens[2])
            t2 = Decimal(tokens[3])
        except InvalidOperation as exc:
            raise TimCanonicalizationError(
                f"Invalid JUMP MJD bounds in par line: {stripped!r}"
            ) from exc
        out.append((t1, t2, tuple(tokens[4:])))
    return out


def jump_mjd_flag_value(pta_name: str, index: int) -> str:
    """Return the MetaPulsar ``-mjd_jump_pta`` value for window ``index`` (1-based)."""
    value = f"{pta_name}_{index}"
    _validate_flag_text("mjd_jump_pta value", value)
    _validate_flag_text("name", MJD_JUMP_PTA_FLAG)
    return value


def _sat_in_jump_mjd_window(
    sat_token: str,
    t1: Decimal,
    t2: Decimal,
    *,
    timing_package: str,
    toa_line: str,
) -> bool:
    """Match the numeric coercion and endpoint rule of the parsing leg."""
    try:
        if timing_package == "tempo2":
            # tempo2 reads SAT into longdouble, JUMP bounds with sscanf("%lf").
            sat = np.longdouble(sat_token)
            lower = np.float64(str(t1))
            upper = np.float64(str(t2))
            return bool(lower <= sat < upper)
        if timing_package == "pint":
            # PINT's FORMAT 1 parser splits integer/fractional MJD, converts
            # the fraction to float, then exposes mjd_float as float64.
            if "." in sat_token:
                day_text, fraction_text = sat_token.split(".", 1)
                sat = np.float64(int(day_text) + float(f"0.{fraction_text}"))
            else:
                sat = np.float64(int(sat_token))
            lower = np.float64(str(t1))
            upper = np.float64(str(t2))
            return bool(lower <= sat <= upper)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimCanonicalizationError(
            f"Invalid TOA SAT for {timing_package} JUMP MJD selection: "
            f"{toa_line.strip()!r}"
        ) from exc
    raise TimCanonicalizationError(
        f"Unsupported timing package for JUMP MJD selection: {timing_package!r}"
    )


def _stamp_mjd_jump_toa_line(
    line: str,
    *,
    expected_flag_value: Optional[str],
    implicit_flag_count: int = 0,
) -> str:
    """Apply ownership rename / exact-own-pair preserve for one TOA line."""
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

    occurrences = present.get(MJD_JUMP_PTA_FLAG, [])
    exact_own = (
        expected_flag_value is not None
        and len(occurrences) == 1
        and tokens[occurrences[0]] == f"-{MJD_JUMP_PTA_FLAG}"
        and tokens[occurrences[0] + 1] == expected_flag_value
    )

    renames: List[Tuple[int, int, str]] = []
    to_append: List[str] = []
    if exact_own:
        pass
    else:
        if occurrences:
            if f"{MJD_JUMP_PTA_FLAG}_orig" in present:
                raise TimCanonicalizationError(
                    f"Cannot preserve existing -{MJD_JUMP_PTA_FLAG} as "
                    f"-{MJD_JUMP_PTA_FLAG}_orig because "
                    f"-{MJD_JUMP_PTA_FLAG}_orig is already present on TOA line: "
                    f"{line.strip()!r}"
                )
            for index in occurrences:
                start, end = spans[index]
                renames.append((start, end, f"-{MJD_JUMP_PTA_FLAG}_orig"))
        if expected_flag_value is not None:
            to_append.append(f" -{MJD_JUMP_PTA_FLAG} {expected_flag_value}")

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


def stamp_mjd_jump_pta_flags(
    tim_text: str,
    *,
    pta_name: str,
    windows: Sequence[Tuple[Decimal, Decimal]],
    timing_package: str,
) -> str:
    """Append ``-mjd_jump_pta {pta_name}_{k}`` on TOAs selected by each window.

    ``windows[i]`` corresponds to flag value ``f"{pta_name}_{i+1}"``. Selection
    follows the parsing leg: tempo2 ``[t1, t2)``, PINT ``[t1, t2]``. Empty
    ``windows`` returns ``tim_text`` unchanged (still ends with a newline).
    """
    if not windows:
        return tim_text if tim_text.endswith("\n") else tim_text + "\n"

    flag_values = [
        jump_mjd_flag_value(pta_name, index) for index in range(1, len(windows) + 1)
    ]

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
                tempo2_info_active = len(tokens) >= 2 and tokens[1] != "-1"
        if kind != "data" or len(tokens) < 5:
            out.append(line)
            continue

        matches = [
            i
            for i, (t1, t2) in enumerate(windows)
            if _sat_in_jump_mjd_window(
                tokens[2],
                t1,
                t2,
                timing_package=timing_package,
                toa_line=line,
            )
        ]
        if len(matches) > 1:
            matched_values = [flag_values[i] for i in matches]
            raise TimCanonicalizationError(
                f"overlapping JUMP MJD windows select the same TOA "
                f"(flag values {matched_values}): {line.strip()!r}. "
                f"Native JUMP MJD overlaps are additive, but converting them to "
                f"repeated -{MJD_JUMP_PTA_FLAG} keys is lossy under PINT, so "
                f"MetaPulsar rejects multi-window TOAs for a uniform contract."
            )
        expected = flag_values[matches[0]] if matches else None
        out.append(
            _stamp_mjd_jump_toa_line(
                line,
                expected_flag_value=expected,
                implicit_flag_count=int(tempo2_info_active),
            )
        )
    return "\n".join(out) + "\n"


def convert_jump_mjd_par_text(
    engine_par_text: str,
    *,
    pta_name: str,
    release_windows: Sequence[Tuple[Decimal, Decimal, Tuple[str, ...]]],
) -> str:
    """Replace engine-par ``JUMP MJD`` lines with flagged ``JUMP -mjd_jump_pta``.

    ``release_windows`` is the output of :func:`parse_jump_mjd_windows` on the
    release par (assigns ``{pta}_{k}``). Engine lines are matched by equal
    ``(t1, t2)`` Decimal values, consuming identical windows in document order.
    Trailing value/fit/err tokens come from the **engine** line.
    """
    queues: dict[Tuple[Decimal, Decimal], deque[int]] = defaultdict(deque)
    for i, (t1, t2, _vals) in enumerate(release_windows):
        queues[(t1, t2)].append(i)

    out_lines: List[str] = []
    for raw in engine_par_text.splitlines():
        stripped = raw.strip()
        tokens = stripped.split() if stripped else []
        is_jump_mjd = (
            len(tokens) >= 4
            and tokens[0].upper() == "JUMP"
            and tokens[1].upper() == "MJD"
            and not stripped.startswith("#")
            and stripped.upper() != "C"
            and not stripped.upper().startswith("C ")
        )
        if not is_jump_mjd:
            out_lines.append(raw)
            continue
        try:
            t1 = Decimal(tokens[2])
            t2 = Decimal(tokens[3])
        except InvalidOperation as exc:
            raise TimCanonicalizationError(
                f"Invalid JUMP MJD bounds in engine par line: {stripped!r}"
            ) from exc
        key = (t1, t2)
        if not queues[key]:
            raise TimCanonicalizationError(
                f"Engine par has JUMP MJD {t1} {t2} with no matching release "
                f"window for PTA {pta_name!r}: {stripped!r}"
            )
        i = queues[key].popleft()
        flag = jump_mjd_flag_value(pta_name, i + 1)
        trailing = (" " + " ".join(tokens[4:])) if len(tokens) > 4 else ""
        out_lines.append(f"JUMP -{MJD_JUMP_PTA_FLAG} {flag}{trailing}")

    leftovers = [(t1, t2) for (t1, t2), q in queues.items() for _ in q]
    if leftovers:
        raise TimCanonicalizationError(
            f"Release JUMP MJD windows missing from engine par for PTA "
            f"{pta_name!r}: {leftovers}"
        )

    text = "\n".join(out_lines)
    if engine_par_text.endswith("\n"):
        text += "\n"
    return text


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
) -> CanonicalTimResult:
    """Write the canonical standalone FORMAT 1 .tim MetaPulsar hands its engines.

    Args:
        tim_path: Source .tim (may use INCLUDE and may already carry ``-pn``).
        pta_name: MetaPulsar's PTA key, stamped as ``-pta`` and ``-pta_dataset``.
        timing_package: ``"pint"`` or ``"tempo2"``, stamped as ``-timing_package``.
        out_path: Destination path.
        par_text: Par content. Required to convert legacy-format input, and used
            as the source of ``JUMP MJD`` windows for ``-mjd_jump_pta`` stamping.

    Returns:
        :class:`CanonicalTimResult` with the written path. ``effective_mode``
        reflects the *converted/flattened* file and is diagnostic only; the
        factory discovers MODE from the release path via
        :func:`discover_effective_tim_mode`.
    """
    tim_path = Path(tim_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    package: Literal["pint", "tempo2"] = (
        "pint" if timing_package == "pint" else "tempo2"
    )
    try:
        flattened = flatten_tim(tim_path, timing_package=package)
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
            flattened = flatten_tim(converted, timing_package=package)

    text = stamp_metadata_flags(
        flattened.text, pta_name=pta_name, timing_package=timing_package
    )
    if par_text is not None:
        raw_windows = parse_jump_mjd_windows(par_text)
        if raw_windows:
            text = stamp_mjd_jump_pta_flags(
                text,
                pta_name=pta_name,
                windows=[(t1, t2) for t1, t2, _ in raw_windows],
                timing_package=timing_package,
            )
    out_path.write_text(text, encoding="utf-8")
    return CanonicalTimResult(path=out_path, effective_mode=flattened.effective_mode)
