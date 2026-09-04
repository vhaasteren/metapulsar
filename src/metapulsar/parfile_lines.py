"""Line-level primitives shared by every MetaPulsar ``.par`` text editor.

MetaPulsar edits par text; it does not re-serialize it. PINT's writer reorders
lines, drops tempo2-only directives (``MODE``, ``TZRSITE``), invents PINT-only
keys (``DMDATA``, ``NE_SWn``) and respells mask parameters (``FDJUMP1`` ->
``FD1JUMP``) -- each of which breaks a tempo2 consumer of the same file. These
helpers are the shared vocabulary for editing by line instead.
"""

from __future__ import annotations

import re
from typing import Final, Iterator, List, Sequence, Tuple

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\S+")


def is_active_par_line(line: str) -> bool:
    """True for a parameter/directive line: not blank, not a comment.

    Both comment spellings are recognized: ``#`` (PINT) and a leading ``C``
    token (tempo2 ``readParfile.C``).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    upper = stripped.upper()
    return not (upper == "C" or upper.startswith("C "))


def iter_active_par_lines(text: str) -> Iterator[Tuple[int, str]]:
    """Yield ``(line_index, raw_line)`` for every active line of ``text``."""
    for index, line in enumerate(text.splitlines()):
        if is_active_par_line(line):
            yield index, line


def par_line_key(line: str) -> str:
    """Upper-cased first token of an active par line."""
    return line.split()[0].upper()


# WN / RN hyperparameter keys. Shared by combination_writer and the Vela
# residual-ingest shim. Not a timing-parameter alias table: these lines are
# never delay columns on MetaPulsar's current (narrowband) path.
#
# ``DMJUMP`` / ``DMEFAC`` / ``DMEQUAD`` apply only to wideband data, where DM
# is a second measurement. MetaPulsar does not ingest wideband TOAs yet, so
# those lines are unused here; they stay in this set so combination products
# and Vela residual ingest drop them the same way until a wideband path exists.
NOISE_PAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "EFAC",
        "TNEFAC",
        "T2EFAC",
        "EQUAD",
        "TNEQUAD",
        "T2EQUAD",
        "ECORR",
        "TNECORR",
        "DMEFAC",
        "DMEQUAD",
        "DMJUMP",
        "RNAMP",
        "RNIDX",
        "TNREDAMP",
        "TNREDGAM",
        "TNREDC",
        "TNREDF",
        "TNREDFC",
        "TNDMAMP",
        "TNDMGAM",
        "TNCHROMAMP",
        "TNCHROMGAM",
        "TNCHROMIDX",
        "TNGAMMA",
        "TNAMP",
    }
)

_NOISE_PREFIXES: Final[tuple[str, ...]] = (
    "EFAC",
    "EQUAD",
    "ECORR",
    "T2EFAC",
    "T2EQUAD",
    "TNEFAC",
    "TNEQUAD",
    "DMEFAC",
    "DMEQUAD",
    "DMJUMP",
)


def is_noise_line(line: str) -> bool:
    """True when ``line`` is unused on MetaPulsar's residual/combination path.

    White/red-noise hyperparameters (EFAC/EQUAD/ECORR, ``TNRED*``, ``RNAMP``,
    ...) are not delay columns. ``DMJUMP`` / ``DMEFAC`` / ``DMEQUAD`` apply
    only to wideband DM *measurements*, which MetaPulsar does not ingest yet.

    Comments and blanks are not noise. ``TRACK`` / ``TIMEEPH`` / ``T2CMETHOD``
    / ``RA`` / ``RAJ`` are not swallowed by the ``TN*`` / ``RN*`` catch-all.
    """
    if not is_active_par_line(line):
        return False
    key = par_line_key(line)
    if key in NOISE_PAR_KEYS:
        return True
    if key.startswith("TN") and key not in {"TRACK", "TIMEEPH", "T2CMETHOD"}:
        return True
    if key.startswith("RN") and key not in {"RA", "RAJ"}:
        return True
    return any(key.startswith(prefix) for prefix in _NOISE_PREFIXES)


def is_flag_token(token: str) -> bool:
    """True if ``token`` is a flag name, not a numeric value.

    Tempo2's test (``readTimfile.C`` / ``readParfile.C``): a flag starts with
    ``-`` whose second character is not a digit. That keeps ``-9.449e-06`` a
    value while treating ``-pta`` / ``-fe`` as keys.
    """
    return len(token) >= 2 and token[0] == "-" and not token[1].isdigit()


def token_spans(line: str) -> List[Tuple[int, int]]:
    """Character ``(start, end)`` of every whitespace-delimited token."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(line)]


def replace_token(line: str, position: int, token: str) -> str:
    """Return ``line`` with token ``position`` replaced, other columns untouched."""
    spans = token_spans(line)
    if position >= len(spans):
        raise IndexError(
            f"token position {position} out of range for par line {line.strip()!r}"
        )
    start, end = spans[position]
    return line[:start] + token + line[end:]


def join_par_lines(lines: Sequence[str], *, like: str) -> str:
    """Join edited lines, preserving ``like``'s trailing-newline policy."""
    result = "\n".join(lines)
    return result + "\n" if like.endswith("\n") else result
