"""Combination par/tim writer for MetaPulsar factory export.

Internal helper used by ``combination_output_dir=`` on
:meth:`~metapulsar.metapulsar_factory.MetaPulsarFactory.create_metapulsar`.
Not a public text-to-par API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

from .pint_helpers import par_text_with_track_minus_2

# Noise / white-noise / red-noise hyperparameter keys removed from the
# combination par (exact key match).
_NOISE_KEYS: Final[frozenset[str]] = frozenset(
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

_JUMP_RE = re.compile(r"^JUMP\b", re.IGNORECASE)

# Tempo2 FDJUMP / flag-JUMP value tokens are read with sscanf("%lf"), which
# stops at a Fortran ``D`` exponent. Rewrite to ``E`` so both engines see the
# same magnitude (PINT accepts either spelling).
FORTRAN_D_EXPONENT_RE: Final[re.Pattern[str]] = re.compile(r"([0-9])[Dd]([+-]?[0-9]+)")
# Keep the private name used in the feature contract / script port.
_FORTRAN_D_EXPONENT_RE = FORTRAN_D_EXPONENT_RE


def fortran_d_to_e(token: str) -> str:
    """Rewrite Fortran ``D`` exponents in a numeric token to ``E``."""
    return _FORTRAN_D_EXPONENT_RE.sub(r"\1E\2", token)


def sanitize_fortran_exponents(text: str) -> str:
    """Rewrite every Fortran ``D`` exponent in ``text`` to ``E``."""
    return _FORTRAN_D_EXPONENT_RE.sub(r"\1E\2", text)


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#") or s.upper().startswith("C "):
        return False
    key = s.split()[0].upper()
    if key in _NOISE_KEYS:
        return True
    # Catch TN* / RN* families without swallowing TRACK / TIMEEPH / etc.
    if key.startswith("TN") and key not in {"TRACK", "TIMEEPH", "T2CMETHOD"}:
        return True
    if key.startswith("RN") and key not in {"RA", "RAJ"}:
        return True
    for prefix in (
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
    ):
        if key.startswith(prefix):
            return True
    return False


def extract_jump_lines(par_text: str) -> list[str]:
    return [ln.rstrip() for ln in par_text.splitlines() if _JUMP_RE.match(ln.strip())]


def extract_fd_terms(par_text: str) -> list[tuple[int, str, str | None]]:
    """Return ``(index, value, uncertainty_or_None)`` for each ``FDx`` line.

    All matching lines are retained in source document order. Repeated ``FDx``
    indices are not de-duplicated. Tempo2/PINT FD lines look like
    ``FD1  value  fitflag  [uncertainty]``. The fit flag on the emitted FDJUMP
    is always ``1``; value/uncertainty are taken from the source with Fortran
    ``D`` exponents rewritten to ``E``.
    """
    terms: list[tuple[int, str, str | None]] = []
    for ln in par_text.splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        m = re.match(r"^FD(\d+)$", parts[0], re.IGNORECASE)
        if not m or len(parts) < 2:
            continue
        idx = int(m.group(1))
        value = fortran_d_to_e(parts[1])
        uncertainty: str | None = None
        if len(parts) >= 4:
            uncertainty = fortran_d_to_e(parts[3])
        elif len(parts) == 2:
            uncertainty = None
        elif len(parts) == 3 and parts[2] not in {"0", "1"}:
            # Bare NAME VALUE UNCERTAINTY (no fit flag).
            uncertainty = fortran_d_to_e(parts[2])
        terms.append((idx, value, uncertainty))
    return terms


def format_fdjump_line(idx: int, pta: str, value: str, uncertainty: str | None) -> str:
    value = fortran_d_to_e(value)
    if uncertainty is None:
        return f"FDJUMP{idx} -pta {pta} {value} 1"
    return f"FDJUMP{idx} -pta {pta} {value} 1 {fortran_d_to_e(uncertainty)}"


@dataclass(frozen=True)
class CombinationParStats:
    n_jumps: int  # copied JUMP lines + JUMP -pta lines
    n_fdjump: int
    n_fdjumpdm: int


@dataclass(frozen=True)
class CombinationWriteResult:
    par_path: Path
    tim_path: Path
    reference_pta: str
    pta_names: tuple[str, ...]
    stats: CombinationParStats


def write_combination_tim(
    *,
    pulsar: str,
    reference_pta: str,
    pta_tim_paths: Mapping[str, Path],
    out_path: Path,
) -> int:
    """FORMAT 1 + relative INCLUDE of each PTA tim (reference first)."""
    if reference_pta not in pta_tim_paths:
        raise KeyError(f"reference PTA {reference_pta!r} missing from PTA tims")
    ordered = [reference_pta] + sorted(p for p in pta_tim_paths if p != reference_pta)
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "FORMAT 1",
        f"C MetaPulsar combination tim for {pulsar} "
        f"(ref={reference_pta}; INCLUDE of all PTA legs)",
    ]
    for pta in ordered:
        target = Path(pta_tim_paths[pta]).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"missing per-PTA tim for INCLUDE: {target}")
        rel = Path(os.path.relpath(target, out_dir.resolve())).as_posix()
        lines.append(f"INCLUDE {rel}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(ordered)


def write_combination_par(
    *,
    reference_pta: str,
    pta_par_texts: Mapping[str, str],
    out_path: Path,
) -> CombinationParStats:
    """Merged combination par; see feature_combination_parfile_writer.md §3.3."""
    if reference_pta not in pta_par_texts:
        raise KeyError(f"reference PTA {reference_pta!r} missing from per-PTA pars")

    ordered = [reference_pta] + sorted(p for p in pta_par_texts if p != reference_pta)
    ref_text = pta_par_texts[reference_pta]
    kept: list[str] = []
    for ln in ref_text.splitlines():
        s = ln.strip()
        if not s:
            kept.append(ln.rstrip())
            continue
        if _is_noise_line(ln):
            continue
        if _JUMP_RE.match(s):
            continue
        if re.match(r"^FD\d+\b", s, re.IGNORECASE):
            continue
        kept.append(ln.rstrip())

    while kept and not kept[-1].strip():
        kept.pop()

    jump_lines: list[str] = []
    for pta in ordered:
        jump_lines.extend(extract_jump_lines(pta_par_texts[pta]))

    pta_jump_lines = [
        f"JUMP -pta {pta} 0.0 1" for pta in ordered if pta != reference_pta
    ]

    fdjump_lines: list[str] = []
    for pta in ordered:
        for idx, value, uncertainty in extract_fd_terms(pta_par_texts[pta]):
            fdjump_lines.append(format_fdjump_line(idx, pta, value, uncertainty))

    fdjumpdm_lines = [
        f"FDJUMPDM -pta {pta} 0.0 1" for pta in ordered if pta != reference_pta
    ]

    fdjump_scale_lines: list[str] = []
    if fdjump_lines or fdjumpdm_lines:
        fdjump_scale_lines = ["FDJUMPLOG Y", "FDJUMP_SCALE LOG"]

    block: list[str] = [
        "",
        (
            f"# MetaPulsar combination: JUMP from all PTAs; "
            f"JUMP -pta for non-reference; "
            f"FDx -> FDJUMPx (value+uncertainty copied); "
            f"FDJUMPDM for non-reference (ref={reference_pta})"
        ),
    ]
    block.extend(jump_lines)
    block.extend(pta_jump_lines)
    block.extend(fdjump_scale_lines)
    block.extend(fdjump_lines)
    block.extend(fdjumpdm_lines)
    block.append("")

    merged = sanitize_fortran_exponents(
        par_text_with_track_minus_2("\n".join(kept + block))
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(merged, encoding="utf-8")
    return CombinationParStats(
        n_jumps=len(jump_lines) + len(pta_jump_lines),
        n_fdjump=len(fdjump_lines),
        n_fdjumpdm=len(fdjumpdm_lines),
    )
