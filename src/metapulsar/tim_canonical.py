"""Canonical .tim writer: flatten INCLUDEs, bake TIME, and stamp metadata flags.

Opt in with factory ``canonicalize_tim=True`` (default is False). MetaPulsar then
hands its timing engines a standalone Tempo2 ``FORMAT 1`` file carrying
authoritative ``-pta``, ``-pta_dataset`` and ``-timing_package`` flags, so the PTA
identity of every TOA travels with the data instead of being synthesized in
memory. Cumulative ``TIME`` offsets are baked into TOA MJDs with exact decimal
arithmetic (``sat += TIME / 86400``, rounded once at output — not Tempo2
``double``/``longdouble`` bit-equivalence) under the leg's own package scoping
rule, and a ``TIME`` left live at an ``INCLUDE`` boundary is recorded as an
:class:`IncludeScopeResolution` rather than refused. ``TIME`` and ``MODE`` lines
are omitted from the artifact; effective ``MODE`` is discovered from the release
tim tree and transferred onto the engine-facing ``.par``. A data line with fewer
than five fields — which tempo2 discards in silence and PINT raises on — is
dropped to match the release's own package and recorded as a
:class:`DroppedTimLine`. Every TOA name is rewritten to a safe ``toaNNNNN``
token. Bare FORMAT 1 flags (InPTA ``-cycle_post34``, a trailing ``-chan``,
EPTA ``-gis``) are rewritten as key/value pairs with dummy value ``1`` so
Tempo2 cannot steal the following ``-pta`` / ``-pn``. When the release par contains
``JUMP MJD`` windows this module also stamps combination-safe ``-mjd_jump_pta``
flags on the selected (post-bake) TOAs. Zero-valued ``JUMP MJD`` lines (delay
``0`` / omitted) are dropped: they are not stamped and are removed from the
engine par under ``convert_jump_mjd`` (PPTA DR1+DR2 v3 ships nested empty
windows that would otherwise collide under flag conversion). Callers that
flatten INCLUDE trees pass ``canonicalize_tim=True`` explicitly.

No PINT dependency at import time (the legacy-format converter imports its
backend lazily).
"""

from __future__ import annotations

import os
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

from .parfile_lines import (
    is_active_par_line,
    is_flag_token,
    iter_active_par_lines,
    join_par_lines,
)
from .tim_file_analyzer import (
    TimFileAnalyzer,
    TimMetadata,
    is_tim_comment_line,
    read_tim_text_lines,
)

# Flags MetaPulsar owns. An input that already uses one of these names has it
# renamed to ``<name>_orig`` so the release's own value stays auditable.
CANONICAL_METADATA_FLAGS: Tuple[str, ...] = ("pta", "pta_dataset", "timing_package")
MJD_JUMP_PTA_FLAG = "mjd_jump_pta"
# Dummy value written for a valueless flag so the canonical line is strict
# key/value pairs (Tempo2 always consumes the next token as the value).
VALUELESS_FLAG_VALUE = "1"

# tempo2 hard limits (tempo2.h). Overflow makes tempo2 call exit(1)
# (readTimfile.C), which would kill the interpreter under in-process
# libstempo, so we refuse before writing.
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
_PINT_SAFE_COMMENT_RE = re.compile(r"CC? \S")
_TWO_NONSPACE_RE = re.compile(r"\S\S")
_PN_INTEGER_RE = re.compile(r"([+-]?\d+)(?:\.0+)?\Z")
_MAX_COLUMN_DODGE = 8

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
    """An *emitted* stateful directive is live across an INCLUDE boundary.

    ``EFAC``/``EQUAD``/``SKIP``/``PROFILE_DIR`` and friends survive verbatim
    into the flattened file, which no longer carries the file boundary that
    scoped them on a tempo2 leg, so MetaPulsar refuses rather than silently
    widening them. ``TIME`` is exempt: it is baked into MJDs and never emitted,
    so the walker's per-file accumulator already realizes tempo2's scope
    exactly -- those boundaries are recorded instead (see
    :class:`IncludeScopeResolution`).
    """


BoundaryKind = Literal["include_entry", "include_eof", "include_end"]


@dataclass(frozen=True)
class IncludeScopeResolution:
    """One recorded ``TIME``-scoping divergence between tempo2 and PINT.

    tempo2 keeps its ``TIME`` accumulator function-local to each recursive
    ``readTim()`` call (``readTimfile.C``) while PINT shares one command dict
    across ``INCLUDE`` (``toa.py``), so a live ``TIME`` at an include boundary
    is read differently by the two engines. Flattening must pick one reading and
    picks the leg's own package; this records where that happened, so the choice
    is auditable instead of silent.

    Attributes:
        path: File whose boundary this is -- the *parent* for ``include_entry``,
            the included file for ``include_eof`` / ``include_end``.
        directive: Always ``"TIME"`` today.
        boundary: Which boundary, so ``path`` is never ambiguous.
        offset_seconds: The exact offset the two engines disagree about. For
            ``include_entry`` this is the live accumulator the parent holds at
            the ``INCLUDE``; for the closing boundaries it is the included
            file's *own* net contribution, measured against its entry value
            (nonzero only on a PINT leg, which inherits the parent's total).
        toas_emitted_before: TOAs already emitted when the boundary was reached.
            A position marker, deliberately *not* a count of affected TOAs: the
            counterfactual reach of a PINT-style leak runs on through later
            siblings and ancestors.
        disposition: ``"scoped"`` when the offset stops at the boundary (tempo2
            rule), ``"carried"`` when it crosses it (PINT rule).
        include_path: The included file, for ``include_entry`` only.
    """

    path: Path
    directive: str
    boundary: BoundaryKind
    offset_seconds: Fraction
    toas_emitted_before: int
    disposition: Literal["scoped", "carried"]
    include_path: Optional[Path] = None


@dataclass(frozen=True)
class DroppedTimLine:
    """One source line tempo2 reads as neither a TOA nor a directive.

    Tempo2's free-format reader (``readTimfile.C``) parses a data line with
    ``sscanf(line, "%s %lf %s %lf %s", ...)``; fewer than five fields leaves
    ``valid == 0``, so ``psr->nobs`` is never incremented and the line matches no
    directive keyword either -- it is discarded in silence. PINT, by contrast,
    raises ``IndexError`` on the same line.

    Released trees do contain such lines: EPTA DR2's Jodrell files continue a
    TOA's flags onto the next physical line (``-padd <value>`` alone), so the
    published solution was fitted *without* those flags. Canonicalization matches
    tempo2 and drops them, and records each one here so the loss is auditable
    rather than silent.

    Attributes:
        path: File the line came from (post-``INCLUDE`` flattening, this is the
            file that literally held it, not the root).
        line_number: 1-based line number within ``path``.
        text: The offending line, stripped.
        toas_emitted_before: TOAs already emitted when the line was reached, so
            the drop can be located in the canonical output.
    """

    path: Path
    line_number: int
    text: str
    toas_emitted_before: int


@dataclass(frozen=True)
class FlattenTimResult:
    """Standalone FORMAT 1 text plus the last observed ``MODE`` in the tree."""

    text: str
    effective_mode: Optional[int]
    column_dodge_count: int = 0
    include_scope_resolutions: Tuple[IncludeScopeResolution, ...] = ()
    dropped_lines: Tuple[DroppedTimLine, ...] = ()


@dataclass(frozen=True)
class CanonicalTimResult:
    """Written canonical ``.tim`` plus metadata and layout diagnostics."""

    path: Path
    effective_mode: Optional[int]
    tim_metadata: TimMetadata
    column_dodge_count: int = 0
    include_scope_resolutions: Tuple[IncludeScopeResolution, ...] = ()
    dropped_lines: Tuple[DroppedTimLine, ...] = ()


@dataclass
class _TimeAccum:
    """Exact cumulative TIME offset in seconds."""

    total: Fraction = Fraction(0)

    def add(self, delta: Fraction) -> None:
        self.total += delta


class _ScopeState:
    """Tracks the *emitted* stateful tim directives across INCLUDE boundaries.

    ``TIME`` is deliberately absent: it is baked into MJDs and never emitted, so
    its scope is carried by the walker's exact accumulator instead.
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

    def live_directives(self) -> List[str]:
        """Return emitted directives whose state would leak across an INCLUDE.

        Flattening drops the file boundary that scoped these on a tempo2 leg,
        and unlike ``TIME`` they reach the artifact as directive lines, so a
        live one here is unrepresentable rather than merely ambiguous.
        """
        live = [
            name
            for name, default in _TEMPO2_ASSIGNING_DEFAULTS.items()
            if self.assignments[name] != default
        ]
        live.extend(sorted(self.invalid))
        if self.profile_dir is not None:
            live.append("PROFILE_DIR")
        if self.skipping:
            live.append("SKIP")
        return live


def _read_lines(path: Path) -> List[str]:
    try:
        return read_tim_text_lines(path)
    except OSError as exc:
        raise TimCanonicalizationError(f"Cannot read .tim file {path}: {exc}") from exc


def _canonical_comment_line(line: str) -> str:
    """Render a source comment so both engines re-read it as a comment.

    Tempo2 comments any leading uppercase ``C`` (including ``C?`` / ``CC?`` /
    glued ``Cc…``). PINT only honors ``C `` / ``CC `` / ``#`` at column 0 with
    text after the marker, and sniffs lowercase ``c `` as Princeton format, so
    anything outside the shape PINT shares is re-marked with ``#`` — the one
    introducer both packages honor unconditionally — keeping the original text
    as comment payload.
    """
    stripped = line.strip()
    if stripped.startswith("#") or _PINT_SAFE_COMMENT_RE.match(stripped):
        return stripped
    return f"# {stripped}"


def _classify(line: str) -> Tuple[str, List[str]]:
    """Return ``(kind, tokens)`` where kind is comment/blank/directive/data."""
    stripped = line.strip()
    if not stripped:
        return "blank", []
    if is_tim_comment_line(stripped):
        return "comment", []
    tokens = stripped.split()
    name = tokens[0].upper()
    if name in _DIRECTIVE_NAMES or name.startswith("T2E") or name.startswith("TNE"):
        return "directive", tokens
    return "data", tokens


def classify_tim_line(line: str) -> Tuple[str, List[str]]:
    """Public internal wrapper for canonical tim line classification."""
    return _classify(line)


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


def _parse_sat_flag_offset(token: str, *, what: str) -> Fraction:
    """Exact value for a Tempo2 SAT-shifting flag (-addsat seconds, -padd turns)."""
    return _parse_bounded_decimal(
        token,
        what=what,
        minimum=-_MAX_TIME_ABS_SECONDS,
        maximum=_MAX_TIME_ABS_SECONDS,
    )


def _first_par_f0(par_text: str) -> Optional[Fraction]:
    """Exact spin frequency ``F0`` in Hz from a par, or None if absent/unusable.

    Needed only to bake ``-padd`` (a phase offset) into the arrival time as
    ``padd / F0`` seconds; ``-addsat`` (seconds) needs no model quantity.
    """
    for _index, line in iter_active_par_lines(par_text):
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "F0":
            try:
                value = Decimal(parts[1].replace("D", "E").replace("d", "E"))
            except InvalidOperation:
                return None
            if not value.is_finite() or value == 0:
                return None
            return Fraction(value)
    return None


def _extract_sat_corrections(
    flag_tokens: List[str], *, f0: Optional[Fraction] = None
) -> Tuple[Fraction, List[str]]:
    """Split Tempo2 SAT-shifting per-TOA flags out of a TOA's flag list.

    Two Tempo2 flags shift a single TOA in ways PINT does not implement, so the
    canonical writer bakes both into the arrival time (exactly as it bakes
    ``TIME``) and drops them, leaving one SAT both engines agree on:

    * ``-addsat N`` -- add ``N`` seconds directly.
    * ``-padd P`` -- add ``P`` turns of *phase*, i.e. ``P / F0`` seconds. This is
      the one place a model quantity (``F0``) enters, unavoidable because a phase
      offset is a time offset only through the spin frequency; the F1 curvature
      this ignores is ~1e-12 turns over a data span, far below the output round.
      Baked only when ``f0`` is known -- mode discovery walks without a par and
      discards the rendered line, so an unbaked ``-padd`` is simply left in place.

    Returns the summed exact offset in seconds and the flag list with every baked
    pair removed. It scans for the flags rather than walking strict flag/value
    pairs, so a valueless neighbour (a bare ``-gis`` in EPTA DR1 tims) is
    tolerated.
    """
    total = Fraction(0)
    kept: List[str] = []
    index = 0
    count = len(flag_tokens)
    while index < count:
        token = flag_tokens[index]
        key = token.lower()
        if key == "-addsat":
            if index + 1 >= count:
                raise TimCanonicalizationError(
                    f"-addsat flag without a value: {' '.join(flag_tokens)!r}"
                )
            total += _parse_sat_flag_offset(
                flag_tokens[index + 1], what="-addsat offset"
            )
            index += 2
            continue
        if key == "-padd" and f0 is not None:
            if index + 1 >= count:
                raise TimCanonicalizationError(
                    f"-padd flag without a value: {' '.join(flag_tokens)!r}"
                )
            turns = _parse_sat_flag_offset(flag_tokens[index + 1], what="-padd offset")
            total += turns / f0
            index += 2
            continue
        kept.append(token)
        index += 1
    return total, kept


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


def _bake_mjd_token(
    mjd_token: str, accum: _TimeAccum, *, extra_seconds: Fraction = Fraction(0)
) -> str:
    mjd = _parse_bounded_decimal(
        mjd_token, what="TOA MJD token", minimum=_MIN_MJD, maximum=_MAX_MJD
    )
    # Cumulative TIME plus this TOA's own -addsat, summed once so both bake
    # through the same exact rational arithmetic and are rounded only at output.
    offset = accum.total + extra_seconds
    if offset == 0:
        return mjd_token  # validated above; preserve exact source bytes
    baked = mjd + offset / _SECDAY
    # In-range tokens can still bake out of range (e.g. MJD 0.5 with TIME -1e9).
    if not (_MIN_MJD_EXACT <= baked <= _MAX_MJD_EXACT):
        raise TimCanonicalizationError(
            f"Baked TOA MJD {baked} out of range [{_MIN_MJD}, {_MAX_MJD}], "
            f"from source {mjd_token!r}, TIME {accum.total} s and "
            f"-addsat {extra_seconds} s"
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
    line: str,
    *,
    tokens: List[str],
    accum: _TimeAccum,
    name_counter: List[int],
    column_dodge_counter: List[int],
    f0: Optional[Fraction] = None,
) -> str:
    if len(tokens) < 5:
        # Internal invariant: the walker filters these out (see
        # `DroppedTimLine`), so reaching here means a caller bypassed it.
        raise TimCanonicalizationError(
            f"FORMAT 1 TOA line has too few tokens: {line.strip()!r}"
        )
    name_counter[0] += 1
    name = f"toa{name_counter[0]:05d}"
    # Bake this TOA's Tempo2 SAT shifts (-addsat seconds, -padd turns via F0)
    # into the SAT alongside cumulative TIME and drop the flags, so PINT (which
    # implements neither) and Tempo2 read one arrival time. Precision matches the
    # TIME bake: exact Fraction, rounded once to >=17 MJD fractional digits.
    sat_shift_seconds, flag_tokens = _extract_sat_corrections(tokens[5:], f0=f0)
    flag_tokens = _pair_flag_tokens(flag_tokens)
    mjd = _bake_mjd_token(tokens[2], accum, extra_seconds=sat_shift_seconds)
    flags = (" " + " ".join(flag_tokens)) if flag_tokens else ""
    for pad in range(_MAX_COLUMN_DODGE):
        rebuilt = (
            f" {name}{' ' * (pad + 1)}{tokens[1]} {mjd} "
            f"{tokens[3]} {tokens[4]}{flags}"
        )
        if not _pint_legacy_heuristic_hit(rebuilt):
            if pad:
                column_dodge_counter[0] += 1
            return rebuilt
    raise TimCanonicalizationError(
        "Cannot lay out canonical TOA line clear of PINT's legacy format "
        f"heuristics after {_MAX_COLUMN_DODGE} attempts: {rebuilt!r}"
    )


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
        f0: Optional[Fraction] = None,
    ):
        self.timing_package = timing_package
        self.emit = emit
        self.on_legacy_toa = on_legacy_toa
        self.f0 = f0  # for -padd phase->time bake; None on par-less mode discovery
        self.out: List[str] = ["FORMAT 1"] if emit else []
        self.name_counter: List[int] = [0]
        self.column_dodge_counter: List[int] = [0]
        self.effective_mode: Optional[int] = None
        self.shared_time = _TimeAccum()  # PINT shares; tempo2 is file-local
        self.include_scope_resolutions: List[IncludeScopeResolution] = []
        self.dropped_lines: List[DroppedTimLine] = []

    def _write(self, line: str) -> None:
        if self.emit:
            self.out.append(line)

    def _record_dropped_line(self, path: Path, lineno: int, line: str) -> None:
        """Record a data line tempo2 discards, and warn once per file."""
        first_in_file = not any(d.path == path for d in self.dropped_lines)
        self.dropped_lines.append(
            DroppedTimLine(
                path=path,
                line_number=lineno,
                text=line.strip(),
                toas_emitted_before=self.name_counter[0],
            )
        )
        # Discovery walks the same tree just before canonicalization, so warning
        # on both would double every message.
        if self.emit and first_in_file:
            logger.warning(
                f"{path} holds data lines with fewer than 5 fields, the first at "
                f"line {lineno}: {line.strip()!r}. tempo2 discards these silently "
                f"(readTimfile.C: nread < 5 leaves the observation invalid) and "
                f"PINT raises on them, so canonicalization drops them to match the "
                f"release's own package. Any per-TOA flags they carried are lost, "
                f"exactly as they were for the published solution."
            )

    def _record_time_scope(
        self,
        path: Path,
        offset: Fraction,
        *,
        boundary: BoundaryKind,
        include_path: Optional[Path] = None,
    ) -> None:
        """Record -- and, when emitting, warn about -- a TIME scope divergence.

        ``offset`` is the exact quantity the two engines disagree about at this
        boundary; callers supply the included file's *own* net contribution at
        the closing boundaries, so an inherited PINT total is not miscounted as
        child-emitted state.
        """
        if offset == 0:
            return
        disposition: Literal["scoped", "carried"] = (
            "scoped" if self.timing_package == "tempo2" else "carried"
        )
        self.include_scope_resolutions.append(
            IncludeScopeResolution(
                path=path,
                directive="TIME",
                boundary=boundary,
                offset_seconds=offset,
                toas_emitted_before=self.name_counter[0],
                disposition=disposition,
                include_path=include_path,
            )
        )
        # Discovery walks the same tree just before canonicalization, so warning
        # on both would double every message.
        if not self.emit:
            return
        rule = (
            "stops at the boundary (tempo2 scope; PINT would carry it)"
            if disposition == "scoped"
            else "crosses the boundary (PINT scope; tempo2 would drop it)"
        )
        logger.warning(
            f"TIME scope divergence at {boundary} of {path}: "
            f"{float(offset):g} s (exact {offset}) {rule}. "
            f"{self.name_counter[0]} TOAs emitted before this boundary."
        )

    def walk(self, path: Path, stack: List[Path]) -> bool:
        """Walk one file. Returns True when a top-level END stops everything."""
        if path in stack:
            chain = " -> ".join(str(p) for p in [*stack, path])
            raise TimCanonicalizationError(f"Circular INCLUDE detected: {chain}")

        stack.append(path)
        tokenized = False
        scope = _ScopeState()
        accum = self.shared_time if self.timing_package == "pint" else _TimeAccum()
        # Zero on a tempo2 leg (fresh accumulator); the inherited total on a
        # PINT leg, which is what makes the closing boundaries measure this
        # file's own contribution rather than its parent's.
        entry_total = accum.total
        try:
            for lineno, line in enumerate(_read_lines(path), start=1):
                kind, tokens = classify_tim_line(line)

                if kind == "blank":
                    self._write(line)
                    continue

                if kind == "comment":
                    self._write(_canonical_comment_line(line))
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
                    if len(tokens) < 5:
                        # Neither a TOA nor a directive to tempo2's free-format
                        # reader, which drops it without a word. Match that, but
                        # record it. Checked only under FORMAT 1: the legacy
                        # column formats are short by construction and belong to
                        # the converter above, not here.
                        self._record_dropped_line(path, lineno, line)
                        continue
                    self._write(
                        _render_canonical_toa_line(
                            line,
                            tokens=tokens,
                            accum=accum,
                            name_counter=self.name_counter,
                            column_dodge_counter=self.column_dodge_counter,
                            f0=self.f0,
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
                        # tempo2's `endit` is function-local too, so END and EOF
                        # are the same event for the state this file holds.
                        live = scope.live_directives()
                        if live:
                            raise TimIncludeScopeError(
                                f"Included file {path} reaches END with "
                                f"{', '.join(live)} still active. These directives "
                                f"are emitted verbatim, so flattening would leak "
                                f"that tempo2 file-local state into the parent."
                            )
                        self._record_time_scope(
                            path,
                            accum.total - entry_total,
                            boundary="include_end",
                        )
                        return False
                    self._write(line)
                    return True

                if directive == "INCLUDE":
                    if len(tokens) < 2:
                        raise TimCanonicalizationError(
                            f"INCLUDE without a filename in {path}"
                        )
                    live = scope.live_directives()
                    if self.timing_package == "tempo2" and live:
                        raise TimIncludeScopeError(
                            f"{path} reaches an INCLUDE with {', '.join(live)} still "
                            f"active. tempo2 scopes these per included file while PINT "
                            f"leaks them into it, and they are emitted verbatim, so "
                            f"flattening would change which TOAs they apply to. "
                            f"Balance the directive around the INCLUDE."
                        )
                    include_path = (path.parent / tokens[1]).resolve()
                    if not include_path.is_file():
                        raise TimCanonicalizationError(
                            f"INCLUDE file not found: {include_path} (from {path})"
                        )
                    self._record_time_scope(
                        path,
                        accum.total,
                        boundary="include_entry",
                        include_path=include_path,
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

            if len(stack) > 1:
                live = scope.live_directives()
                if self.timing_package == "tempo2" and live:
                    raise TimIncludeScopeError(
                        f"Included file {path} ends with {', '.join(live)} still "
                        f"active. These directives are emitted verbatim, so while "
                        f"tempo2 drops this state at the end of the file, flattening "
                        f"would carry it into the parent's following TOAs. Balance "
                        f"the directive inside the included file."
                    )
                self._record_time_scope(
                    path, accum.total - entry_total, boundary="include_eof"
                )
            return False
        finally:
            stack.pop()


def flatten_tim(
    tim_path: Path,
    *,
    timing_package: Literal["pint", "tempo2"] = "tempo2",
    f0: Optional[Fraction] = None,
) -> FlattenTimResult:
    """Return standalone Tempo2 FORMAT 1 text for ``tim_path``.

    INCLUDE directives are inlined in place. Cumulative ``TIME`` offsets are
    baked into TOA MJDs with exact decimal arithmetic and omitted from the
    output; ``MODE`` lines are omitted (their value is returned as
    ``effective_mode``). Every TOA name is rewritten to ``toaNNNNN``.

    A ``TIME`` left live at an INCLUDE boundary is *not* an error: it is baked
    under the leg's own package rule and reported in
    ``include_scope_resolutions``.

    Raises:
        TimLegacyFormatError: A TOA appears without ``FORMAT 1`` in effect, or a
            non-``1`` ``FORMAT`` declaration is present.
        TimIncludeScopeError: An emitted stateful directive (``EFAC`` family,
            ``PROFILE_DIR``, ``SKIP``) is live across an INCLUDE boundary.
        TimCanonicalizationError: Missing or circular INCLUDE, unreadable file,
            or invalid observed ``TIME``/``MODE``/MJD.
    """
    walker = _TimWalker(
        timing_package=timing_package, emit=True, on_legacy_toa="raise", f0=f0
    )
    walker.walk(Path(tim_path).resolve(), [])
    return FlattenTimResult(
        text="\n".join(walker.out) + "\n",
        effective_mode=walker.effective_mode,
        column_dodge_count=walker.column_dodge_counter[0],
        include_scope_resolutions=tuple(walker.include_scope_resolutions),
        dropped_lines=tuple(walker.dropped_lines),
    )


def discover_effective_tim_mode(
    tim_path: Path, *, timing_package: Literal["pint", "tempo2"] = "tempo2"
) -> Optional[int]:
    """Last observed .tim MODE, using the same traversal as canonicalization.

    Runs the same structural directive traversal as :func:`flatten_tim` --
    INCLUDE resolution and circularity, SKIP/END semantics, TIME parsing and
    tempo2 emitted-directive scope guards, FORMAT 1 TOA validation -- and raises
    the same errors for FORMAT 1 input. Legacy input (a non-``1`` ``FORMAT``
    declaration, or TOAs with no ``FORMAT 1`` in effect) is tolerated rather
    than rejected, because
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
    return join_par_lines(out, like=par_text)


# The .tim and .par flag rules are the same tempo2 test (``readTimfile.C`` /
# ``readParfile.C``): a flag starts with ``-`` whose second character is not a
# digit, keeping ``-to -0.897e-6`` and ``-addsat -1`` as values while treating
# ``-pta`` / ``-cycle_post34`` as keys. Keep one definition.
_is_flag_key = is_flag_token


def _iter_flag_pairs(
    tokens: Sequence[str], *, start: int = 5
) -> List[Tuple[int, Optional[int]]]:
    """Return ``(key_index, value_index_or_None)`` for FORMAT 1 flags.

    A key consumes the next token as its value only when that token is not
    itself a flag key. Otherwise the flag is valueless (InPTA ``-cycle_post34``,
    trailing ``-chan``, EPTA ``-gis``). Stray non-key tokens are skipped.
    """
    pairs: List[Tuple[int, Optional[int]]] = []
    index = start
    count = len(tokens)
    while index < count:
        if not _is_flag_key(tokens[index]):
            index += 1
            continue
        key_index = index
        if index + 1 < count and not _is_flag_key(tokens[index + 1]):
            pairs.append((key_index, index + 1))
            index += 2
        else:
            pairs.append((key_index, None))
            index += 1
    return pairs


def _flag_key_indices(tokens: Sequence[str]) -> List[int]:
    """Return token indices holding flag keys (FORMAT 1: flags start at index 5)."""
    return [key_index for key_index, _ in _iter_flag_pairs(tokens)]


def _pair_flag_tokens(flag_tokens: Sequence[str]) -> List[str]:
    """Rewrite a flag-token list to strict key/value pairs.

    Valueless flags receive :data:`VALUELESS_FLAG_VALUE` so Tempo2 cannot steal
    the next ``-pta`` or ``-pn``.
    """
    paired: List[str] = []
    for key_index, value_index in _iter_flag_pairs(flag_tokens, start=0):
        paired.append(flag_tokens[key_index])
        if value_index is None:
            paired.append(VALUELESS_FLAG_VALUE)
        else:
            paired.append(flag_tokens[value_index])
    return paired


def _normalize_toa_flag_pairs(line: str) -> str:
    """Rewrite a FORMAT 1 TOA so every flag is a key/value pair.

    Pair-shaped lines are returned unchanged (spacing preserved). Bare flags
    are filled with :data:`VALUELESS_FLAG_VALUE`.
    """
    spans = list(_TOKEN_RE.finditer(line))
    if len(spans) < 5:
        return line
    tokens = [match.group(0) for match in spans]
    pairs = _iter_flag_pairs(tokens)
    if not pairs or all(value_index is not None for _, value_index in pairs):
        return line
    prefix = line[: spans[4].end()]
    paired = _pair_flag_tokens(tokens[5:])
    rebuilt = prefix if not paired else f"{prefix} {' '.join(paired)}"
    return rebuilt.rstrip("\r")


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
    keeping its raw value. If ``<name>_orig`` is already present, that
    preserved value is left alone and the live ``-<name>`` flag is overwritten
    with the authoritative MetaPulsar value. PINT lowercases flag keys and
    keeps only the last duplicate while tempo2 is case-sensitive and keeps
    them all, so matching case-insensitively is what keeps both engines seeing
    the same thing.

    Raises:
        TimCanonicalizationError: A value is unusable, or stamping would
            exceed tempo2's per-TOA flag limits.
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
        kind, tokens = classify_tim_line(line)
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


def _pn_occurrences(line: str) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Return source spans and values for case-insensitive ``-pn`` pairs."""
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]
    tokens = [line[start:end] for start, end in spans]
    occurrences: List[Tuple[int, int]] = []
    values: List[str] = []
    for key_index, value_index in _iter_flag_pairs(tokens):
        if tokens[key_index].lstrip("-").lower() != "pn" or value_index is None:
            continue
        start = spans[key_index][0]
        while start > 0 and line[start - 1].isspace():
            start -= 1
        occurrences.append((start, spans[value_index][1]))
        values.append(tokens[value_index])
    return occurrences, values


def replace_pn_on_toa_line(line: str, pn_value: str) -> str:
    """Replace or append a single ``-pn`` flag on one TOA line."""
    line = _normalize_toa_flag_pairs(line)
    occurrences, _ = _pn_occurrences(line)
    replaced = line
    for start, end in reversed(occurrences):
        replaced = replaced[:start] + replaced[end:]
    return replaced.rstrip("\r") + f" -pn {pn_value}"


def _validate_injected_tim_text(text: str, affected_names: set[str]) -> None:
    """Recheck Tempo2's hard limits on every TOA changed by PN injection."""
    info_active = False
    skipping = False
    for line in text.splitlines():
        kind, tokens = classify_tim_line(line)
        if kind == "directive":
            directive = tokens[0].upper()
            if directive == "SKIP":
                skipping = True
            elif directive == "NOSKIP":
                skipping = False
            elif directive == "INFO" and not skipping:
                info_active = len(tokens) >= 2 and tokens[1] != "-1"
            continue
        if kind != "data" or len(tokens) < 5 or tokens[0] not in affected_names:
            continue

        _, pn_values = _pn_occurrences(line)
        if len(pn_values) != 1:
            raise TimCanonicalizationError(
                f"Injected TOA must contain exactly one -pn flag: {line.strip()!r}"
            )
        _validate_flag_text("-pn value", pn_values[0])
        n_flags = len(_flag_key_indices(tokens)) + int(info_active)
        if n_flags >= TEMPO2_MAX_FLAGS:
            raise TimCanonicalizationError(
                f"Injecting -pn would give this TOA {n_flags} flags, reaching "
                f"tempo2's fatal MAX_FLAGS={TEMPO2_MAX_FLAGS}: {line.strip()!r}"
            )
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > TEMPO2_MAX_TIM_LINE_BYTES:
            raise TimCanonicalizationError(
                f"Injected TOA line is {line_bytes} bytes; tempo2 can safely read at "
                f"most {TEMPO2_MAX_TIM_LINE_BYTES} bytes before the newline: "
                f"{line.strip()!r}"
            )


def inject_pulse_numbers(canonical_tim: Path, *, derived_tim: Path) -> int:
    """Replace canonical ``-pn`` flags with engine-derived values, joined by name.

    Derived TOAs must have unique names, exactly one integer ``-pn`` pair, and
    names present in the canonical file. Canonical-only names are permitted
    because engine selection directives can omit TOAs from derived output.
    """
    canonical_tim = Path(canonical_tim)
    derived_tim = Path(derived_tim)
    canonical_text = canonical_tim.read_text(encoding="utf-8")
    derived_text = derived_tim.read_text(encoding="utf-8")

    canonical_names: Dict[str, int] = {}
    canonical_lines = canonical_text.splitlines()
    for index, line in enumerate(canonical_lines):
        kind, tokens = classify_tim_line(line)
        if kind != "data" or len(tokens) < 5:
            continue
        name = tokens[0]
        if name in canonical_names:
            raise TimCanonicalizationError(
                f"Duplicate TOA name {name!r} in canonical file {canonical_tim}"
            )
        canonical_names[name] = index

    derived_values: Dict[str, str] = {}
    for line in derived_text.splitlines():
        kind, tokens = classify_tim_line(line)
        if kind != "data" or len(tokens) < 5:
            continue
        name = tokens[0]
        if name in derived_values:
            raise TimCanonicalizationError(
                f"Duplicate TOA name {name!r} in derived file {derived_tim}"
            )
        _, values = _pn_occurrences(line)
        if len(values) != 1:
            raise TimCanonicalizationError(
                f"Derived TOA {name!r} must contain exactly one -pn flag: "
                f"{line.strip()!r}"
            )
        integer_match = _PN_INTEGER_RE.fullmatch(values[0])
        if integer_match is None:
            raise TimCanonicalizationError(
                f"Derived TOA {name!r} has non-integral -pn value {values[0]!r}"
            )
        normalized = str(int(integer_match.group(1)))
        _validate_flag_text("-pn value", normalized)
        derived_values[name] = normalized

    unknown = sorted(set(derived_values) - set(canonical_names))
    if unknown:
        raise TimCanonicalizationError(
            f"Derived TOA names are absent from canonical file {canonical_tim}: "
            + ", ".join(repr(name) for name in unknown[:5])
        )

    for name, pn_value in derived_values.items():
        index = canonical_names[name]
        canonical_lines[index] = replace_pn_on_toa_line(
            canonical_lines[index], pn_value
        )

    trailing_newline = canonical_text.endswith("\n")
    replacement_text = "\n".join(canonical_lines) + ("\n" if trailing_newline else "")
    affected_names = set(derived_values)
    _validate_injected_tim_text(replacement_text, affected_names)

    if replacement_text != canonical_text:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{canonical_tim.name}.", suffix=".tmp", dir=canonical_tim.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(replacement_text)
            os.chmod(temporary_path, canonical_tim.stat().st_mode)
            os.replace(temporary_path, canonical_tim)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    unmatched_count = len(canonical_names) - len(derived_values)
    logger.debug(
        f"Injected pulse numbers into {len(derived_values)} TOAs in {canonical_tim}; "
        f"left {unmatched_count} canonical-only TOAs unchanged"
    )
    return len(derived_values)


def _stamp_toa_line(
    line: str, *, values: Dict[str, str], implicit_flag_count: int = 0
) -> str:
    line = _normalize_toa_flag_pairs(line)
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]
    tokens = [line[start:end] for start, end in spans]

    pairs = _iter_flag_pairs(tokens)
    key_indices = [key_index for key_index, _ in pairs]
    value_at = {key_index: value_index for key_index, value_index in pairs}
    present: Dict[str, List[int]] = {}

    for index in key_indices:
        token = tokens[index]
        if not token.startswith("-"):
            continue
        name = token.lstrip("-").lower()
        present.setdefault(name, []).append(index)

    def _has_exact_pair(name: str, expected: str) -> bool:
        occurrences = present.get(name, [])
        if len(occurrences) != 1:
            return False
        key_index = occurrences[0]
        value_index = value_at.get(key_index)
        return (
            tokens[key_index] == f"-{name}"
            and value_index is not None
            and tokens[value_index] == expected
        )

    # Only the complete, exact lowercase trio identifies one of our artifacts.
    # A release flag that happens to have the same value (or uses different
    # case) is still release metadata and must become *_orig.
    already_canonical = all(
        _has_exact_pair(name, values[name]) for name in CANONICAL_METADATA_FLAGS
    )

    replacements: List[Tuple[int, int, str]] = []
    to_append: List[str] = []
    if not already_canonical:
        for name in CANONICAL_METADATA_FLAGS:
            occurrences = present.get(name, [])
            if occurrences and f"{name}_orig" in present:
                for index in occurrences:
                    value_index = value_at.get(index)
                    start = spans[index][0]
                    end = (
                        spans[value_index][1]
                        if value_index is not None
                        else spans[index][1]
                    )
                    replacements.append((start, end, f"-{name} {values[name]}"))
            else:
                if occurrences:
                    for index in occurrences:
                        start, end = spans[index]
                        replacements.append((start, end, f"-{name}_orig"))
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
    for start, end, replacement in reversed(replacements):
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

    Includes zero-valued windows; call :func:`active_jump_mjd_windows` (or
    :func:`jump_mjd_value_is_empty`) before stamping / conversion.
    """
    out: List[Tuple[Decimal, Decimal, Tuple[str, ...]]] = []
    for _index, raw in iter_active_par_lines(par_text):
        stripped = raw.strip()
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


def jump_mjd_value_is_empty(value_tokens: Sequence[str]) -> bool:
    """True when the JUMP delay is omitted or numerically zero.

    Fit-flag / uncertainty tokens are ignored. Non-numeric value tokens are
    treated as non-empty so the line stays visible to conversion matching.
    """
    if not value_tokens:
        return True
    try:
        return Decimal(value_tokens[0]) == 0
    except InvalidOperation:
        return False


def active_jump_mjd_windows(
    windows: Sequence[Tuple[Decimal, Decimal, Tuple[str, ...]]],
) -> List[Tuple[Decimal, Decimal, Tuple[str, ...]]]:
    """Drop zero-valued ``JUMP MJD`` windows (product policy for flag conversion)."""
    return [w for w in windows if not jump_mjd_value_is_empty(w[2])]


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
    line = _normalize_toa_flag_pairs(line)
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]
    tokens = [line[start:end] for start, end in spans]
    pairs = _iter_flag_pairs(tokens)
    key_indices = [key_index for key_index, _ in pairs]
    value_at = {key_index: value_index for key_index, value_index in pairs}
    present: Dict[str, List[int]] = {}
    for index in key_indices:
        token = tokens[index]
        if not token.startswith("-"):
            continue
        name = token.lstrip("-").lower()
        present.setdefault(name, []).append(index)

    occurrences = present.get(MJD_JUMP_PTA_FLAG, [])
    own_value_index = value_at.get(occurrences[0]) if len(occurrences) == 1 else None
    exact_own = (
        expected_flag_value is not None
        and len(occurrences) == 1
        and tokens[occurrences[0]] == f"-{MJD_JUMP_PTA_FLAG}"
        and own_value_index is not None
        and tokens[own_value_index] == expected_flag_value
    )

    replacements: List[Tuple[int, int, str]] = []
    to_append: List[str] = []
    if exact_own:
        pass
    elif occurrences and f"{MJD_JUMP_PTA_FLAG}_orig" in present:
        if expected_flag_value is not None:
            for index in occurrences:
                value_index = value_at.get(index)
                start = spans[index][0]
                end = (
                    spans[value_index][1]
                    if value_index is not None
                    else spans[index][1]
                )
                replacements.append(
                    (start, end, f"-{MJD_JUMP_PTA_FLAG} {expected_flag_value}")
                )
    else:
        if occurrences:
            for index in occurrences:
                start, end = spans[index]
                replacements.append((start, end, f"-{MJD_JUMP_PTA_FLAG}_orig"))
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
    for start, end, replacement in reversed(replacements):
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
        kind, tokens = classify_tim_line(line)
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
    release par. Zero-valued windows are dropped (omitted from the engine par
    and from ``{pta}_{k}`` numbering). Remaining engine lines are matched by
    equal ``(t1, t2)`` Decimal values, consuming identical **active** release
    windows in document order. Trailing value/fit/err tokens come from the
    **engine** line.
    """
    active_release = active_jump_mjd_windows(release_windows)
    queues: dict[Tuple[Decimal, Decimal], deque[int]] = defaultdict(deque)
    for i, (t1, t2, _vals) in enumerate(active_release):
        queues[(t1, t2)].append(i)

    out_lines: List[str] = []
    for raw in engine_par_text.splitlines():
        stripped = raw.strip()
        tokens = stripped.split() if stripped else []
        is_jump_mjd = (
            len(tokens) >= 4
            and tokens[0].upper() == "JUMP"
            and tokens[1].upper() == "MJD"
            and is_active_par_line(raw)
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
        value_tokens = tuple(tokens[4:])
        if jump_mjd_value_is_empty(value_tokens):
            # Product policy: empty JUMP MJD is a no-op; drop rather than convert.
            continue
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
        :func:`discover_effective_tim_mode`. ``include_scope_resolutions``
        lists every ``TIME`` scoping divergence resolved while flattening.
    """
    tim_path = Path(tim_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    package: Literal["pint", "tempo2"] = (
        "pint" if timing_package == "pint" else "tempo2"
    )
    # F0 lets the flatten bake -padd (a phase offset) into the SAT as padd / F0.
    f0 = _first_par_f0(par_text) if par_text is not None else None
    try:
        flattened = flatten_tim(tim_path, timing_package=package, f0=f0)
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
            flattened = flatten_tim(converted, timing_package=package, f0=f0)

    text = stamp_metadata_flags(
        flattened.text, pta_name=pta_name, timing_package=timing_package
    )
    if par_text is not None:
        active_windows = active_jump_mjd_windows(parse_jump_mjd_windows(par_text))
        if active_windows:
            text = stamp_mjd_jump_pta_flags(
                text,
                pta_name=pta_name,
                windows=[(t1, t2) for t1, t2, _ in active_windows],
                timing_package=timing_package,
            )
    out_path.write_text(text, encoding="utf-8")
    tim_metadata = TimFileAnalyzer().get_tim_metadata(out_path)
    if flattened.column_dodge_count:
        logger.debug(
            f"Shifted {flattened.column_dodge_count} canonical TOA lines clear of "
            "PINT legacy format heuristics"
        )
    return CanonicalTimResult(
        path=out_path,
        effective_mode=flattened.effective_mode,
        tim_metadata=tim_metadata,
        column_dodge_count=flattened.column_dodge_count,
        include_scope_resolutions=flattened.include_scope_resolutions,
        dropped_lines=flattened.dropped_lines,
    )
