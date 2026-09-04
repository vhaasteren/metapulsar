"""Combination par/tim writer for MetaPulsar factory export.

Internal helper used by ``combination_output_dir=`` on
:meth:`~metapulsar.metapulsar_factory.MetaPulsarFactory.create_metapulsar`.
Not a public text-to-par API.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, Mapping, Sequence

from .parfile_header import (
    combination_options_header_items,
    ensure_metapulsar_par_header,
    strip_metapulsar_par_header,
)
from .parfile_lines import (
    is_active_par_line,
    is_noise_line,
    iter_active_par_lines,
    join_par_lines,
    par_line_key,
)
from .pint_helpers import par_text_with_track_minus_2
from .tim_canonical import (
    _pn_occurrences,
    classify_tim_line,
    replace_pn_on_toa_line,
)

# Back-compat for tests that import the private names.
_is_noise_line = is_noise_line

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


def _first_par_decimal(par_text: str, key: str) -> Decimal:
    """Return the exact value from the first active line whose first token is key."""
    want = key.upper()
    for _index, line in iter_active_par_lines(par_text):
        parts = line.split()
        if parts[0].upper() == want:
            try:
                return Decimal(parts[1].replace("D", "E").replace("d", "E"))
            except (IndexError, InvalidOperation) as exc:
                raise ValueError(f"invalid {key} line: {line!r}") from exc
    raise ValueError(f"required {key} missing from par text")


def _format_dm_delta(delta: Decimal) -> str:
    """Exact ASCII decimal for FDJUMPDM with any exponent normalized to E."""
    return fortran_d_to_e(str(delta))


def _strip_track_lines(text: str) -> str:
    """Drop non-comment TRACK lines; preserve trailing-newline policy."""
    kept: list[str] = []
    for line in text.splitlines():
        if is_active_par_line(line) and par_line_key(line) == "TRACK":
            continue
        kept.append(line)
    return join_par_lines(kept, like=text)


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
class CombinationPnStats:
    n_toas: int
    pn0_abs: int
    tzrmjd: str
    tzrfrq: str
    tzrsite: str
    # Per-leg diagnostics for the modal-offset renumber, keyed by PTA name:
    # the single integer offset added to each leg's own ``-pn``, the leg's TOA
    # count, the fraction of TOAs voting for that offset, and the largest ±turn
    # any TOA departs from it (the shared model's mispredictions, tolerated
    # because we keep the leg's coherent numbering).
    per_pta_offset: Mapping[str, int]
    per_pta_n_toas: Mapping[str, int]
    per_pta_mode_fraction: Mapping[str, float]
    per_pta_max_deviation: Mapping[str, int]


@dataclass(frozen=True)
class CombinationWriteResult:
    par_path: Path
    tim_path: Path
    reference_pta: str
    pta_names: tuple[str, ...]
    stats: CombinationParStats
    pn_stats: CombinationPnStats | None


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
    track_pulse_numbers: bool = True,
    combination_options: Mapping[str, object] | None = None,
) -> CombinationParStats:
    """Merged combination par.

    ``combination_options`` (when provided) are stamped into the MetaPulsar
    comment header — typically factory user-facing knobs and
    :class:`~metapulsar.parameter_manager.AlignmentPolicy` fields via
    :func:`~metapulsar.parfile_header.combination_options_header_items`.
    """
    if reference_pta not in pta_par_texts:
        raise KeyError(f"reference PTA {reference_pta!r} missing from per-PTA pars")

    ordered = [reference_pta] + sorted(p for p in pta_par_texts if p != reference_pta)
    ref_text = strip_metapulsar_par_header(pta_par_texts[reference_pta])
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

    fdjumpdm_lines: list[str] = []
    non_reference_ptas = [pta for pta in ordered if pta != reference_pta]
    if non_reference_ptas:
        ref_dm = _first_par_decimal(pta_par_texts[reference_pta], "DM")
        for pta in non_reference_ptas:
            pta_dm = _first_par_decimal(pta_par_texts[pta], "DM")
            delta = pta_dm - ref_dm
            fdjumpdm_lines.append(f"FDJUMPDM -pta {pta} {_format_dm_delta(delta)} 1")

    fdjump_scale_lines: list[str] = []
    if fdjump_lines or fdjumpdm_lines:
        fdjump_scale_lines = ["FDJUMPLOG Y", "FDJUMP_SCALE LOG"]

    block: list[str] = [
        "",
        (
            f"# MetaPulsar combination: JUMP from all PTAs; "
            f"JUMP -pta for non-reference; "
            f"FDx -> FDJUMPx (value+uncertainty copied); "
            f"FDJUMPDM = DM - DM_ref for non-reference "
            f"(ref={reference_pta})"
        ),
    ]
    block.extend(jump_lines)
    block.extend(pta_jump_lines)
    block.extend(fdjump_scale_lines)
    block.extend(fdjump_lines)
    block.extend(fdjumpdm_lines)
    block.append("")

    body = "\n".join(kept + block)
    if track_pulse_numbers:
        merged = sanitize_fortran_exponents(par_text_with_track_minus_2(body))
    else:
        merged = sanitize_fortran_exponents(_strip_track_lines(body))
    header_extra = combination_options_header_items(reference_pta=reference_pta)
    if combination_options:
        header_extra.update(dict(combination_options))
    # Ensure Product/reference are present even if caller overrode extras.
    header_extra.setdefault("Product", "combination")
    header_extra.setdefault("reference_pta", reference_pta)
    merged = ensure_metapulsar_par_header(merged, extra=header_extra)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(merged, encoding="utf-8")
    return CombinationParStats(
        n_jumps=len(jump_lines) + len(pta_jump_lines),
        n_fdjump=len(fdjump_lines),
        n_fdjumpdm=len(fdjumpdm_lines),
    )


def align_combination_tzr(
    par_path: Path,
    *,
    tzrmjd: str,
    tzrfrq: str,
    tzrsite: str,
) -> None:
    """Rewrite/insert TZRMJD/TZRFRQ/TZRSITE on the combination par."""
    text = Path(par_path).read_text(encoding="utf-8")
    wanted = {
        "TZRMJD": tzrmjd,
        "TZRFRQ": tzrfrq,
        "TZRSITE": tzrsite,
    }
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in text.splitlines():
        if is_active_par_line(line):
            key = par_line_key(line)
            if key in wanted:
                out_lines.append(f"{key} {wanted[key]}")
                seen.add(key)
                continue
        out_lines.append(line)
    for key, value in wanted.items():
        if key not in seen:
            out_lines.append(f"{key} {value}")
    result = join_par_lines(out_lines, like=text)
    if not text:
        result += "\n"
    Path(par_path).write_text(result, encoding="utf-8")


def _first_data_toa_tokens(tim_path: Path) -> tuple[str, str, str, str]:
    """Return exact FORMAT 1 tokens ``(name, freq, mjd, site)`` for the first data TOA."""
    for line in Path(tim_path).read_text(encoding="utf-8").splitlines():
        kind, tokens = classify_tim_line(line)
        if kind != "data" or len(tokens) < 5:
            continue
        return tokens[0], tokens[1], tokens[2], tokens[4]
    raise ValueError(f"no FORMAT 1 data TOA found in {tim_path}")


def _rewrite_tim_pn_sequential(tim_path: Path, relative_pns: list[int]) -> int:
    """Rewrite ``-pn`` flags in document order; return number of data lines rewritten."""
    text = Path(tim_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    consumed = 0
    for line in lines:
        kind, _tokens = classify_tim_line(line)
        if kind == "data":
            if consumed >= len(relative_pns):
                raise ValueError(
                    f"{tim_path}: more data TOAs than pulse numbers provided"
                )
            out.append(replace_pn_on_toa_line(line, str(relative_pns[consumed])))
            consumed += 1
        else:
            out.append(line)
    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    Path(tim_path).write_text(result, encoding="utf-8")
    return consumed


@dataclass(frozen=True)
class _LegOffset:
    """Outcome of aligning one leg's own ``-pn`` onto the global ladder."""

    offset: int
    n_toas: int
    mode_fraction: float
    max_deviation: int


def _read_pn_sequence(tim_path: Path) -> list[int]:
    """Return each data TOA's integer ``-pn`` in document order.

    Every data TOA must carry exactly one integral ``-pn``; the canonical legs
    guarantee this whenever pulse-number tracking is on, which is the only
    condition under which the combination renumber runs.
    """
    values: list[int] = []
    for line in Path(tim_path).read_text(encoding="utf-8").splitlines():
        if classify_tim_line(line)[0] != "data":
            continue
        _spans, pn_values = _pn_occurrences(line)
        if len(pn_values) != 1:
            raise ValueError(
                f"{tim_path}: each data TOA needs exactly one -pn flag "
                f"(found {len(pn_values)}): {line.strip()!r}"
            )
        try:
            # Canonical / release tims sometimes emit ``-pn 123.0``; accept any
            # float string that is an exact integer (float64-safe for |n|<2^53).
            pn_f = float(pn_values[0])
            if not pn_f.is_integer():
                raise ValueError(f"non-integral -pn {pn_values[0]!r}")
            values.append(int(pn_f))
        except ValueError as exc:
            raise ValueError(
                f"{tim_path}: non-integral -pn {pn_values[0]!r}: {line.strip()!r}"
            ) from exc
    return values


def _modal_offset(model_minus_leg: Sequence[int]) -> _LegOffset:
    """Fold one leg's per-TOA ``inferred - source`` into a single integer offset.

    The offset with the largest vote count wins. Exact ties break toward the
    first-seen value in document order (``Counter`` insertion order). Absolute
    offset is gauge — absorbed by that leg's free ``JUMP`` — so plurality is
    enough; a majority gate is not required. ``mode_fraction`` and
    ``max_deviation`` still report how tight the cluster is.
    """
    if not model_minus_leg:
        raise ValueError("model_minus_leg must be non-empty")
    counts = Counter(model_minus_leg)
    top = max(counts.values())
    # Counter preserves first-seen key order; take the first key at the top count.
    offset = next(value for value, count in counts.items() if count == top)
    return _LegOffset(
        offset=offset,
        n_toas=len(model_minus_leg),
        mode_fraction=top / len(model_minus_leg),
        max_deviation=max(abs(value - offset) for value in model_minus_leg),
    )


def _infer_pulse_numbers(par_path: Path, tim_path: Path) -> list[int]:
    """Nearest-integer absolute pulse number per TOA under the shared model.

    Returned in INCLUDE/document order; a PINT re-sort would break the leg-by-leg
    slicing in the caller and is rejected rather than silently tolerated.
    """
    from pint.models import get_model_and_toas

    model, toas = get_model_and_toas(
        str(par_path),
        str(tim_path),
        allow_T2=True,
        allow_tcb=True,
        planets=True,
        include_pn=True,
        ell1h_shapiro="absorbed",
    )
    indices = list(toas.table["index"])
    if indices != list(range(len(toas))):
        raise ValueError(
            "PINT did not preserve INCLUDE/source TOA order "
            f"(index={indices[:10]}{'...' if len(indices) > 10 else ''})"
        )
    toas.compute_pulse_numbers(model)
    return [int(x) for x in toas.table["pulse_number"]]


def renumber_combination_pulse_numbers(
    *,
    combination_par_path: Path,
    combination_tim_path: Path,
    ordered_pta_tims: Sequence[tuple[str, Path]],
) -> CombinationPnStats:
    """Place every leg's own ``-pn`` on one global ladder via a per-PTA constant.

    Each PTA leg already carries coherent pulse numbers (derived from that PTA's
    original model); the only unknown is a single integer offset between its
    origin and the reference's. We keep the leg numbers verbatim and add that
    constant — we never re-derive per-TOA pulse numbers from the shared model,
    which would break within-leg coherence wherever the shared model mispredicts.

    The shared model is used only to *vote* on each leg's offset: infer the
    nearest-integer pulse number under it, then take the plurality
    ``inferred - source`` per leg (ties: first-seen). The offset's absolute
    value is gauge — a uniform per-leg shift is absorbed exactly by that leg's
    free ``JUMP`` (empirically to ~1e-12 turns), so no residual search can or
    need pick it; the plurality mode is the canonical representative.
    """
    if not ordered_pta_tims:
        raise ValueError("ordered_pta_tims must be non-empty")
    ordered = [(name, Path(path)) for name, path in ordered_pta_tims]
    for _name, path in ordered:
        if not path.is_file():
            raise FileNotFoundError(f"missing combination INCLUDE target: {path}")

    # Anchor the global ladder's TZR on the reference leg's first data TOA.
    _name, tzrfrq, tzrmjd, tzrsite = _first_data_toa_tokens(ordered[0][1])
    align_combination_tzr(
        Path(combination_par_path),
        tzrmjd=tzrmjd,
        tzrfrq=tzrfrq,
        tzrsite=tzrsite,
    )

    # Step 1: infer pulse numbers under the shared model (no TRACK needed —
    # compute_pulse_numbers rounds model phase regardless of the residual mode).
    inferred = _infer_pulse_numbers(
        Path(combination_par_path), Path(combination_tim_path)
    )
    leg_pns = {name: _read_pn_sequence(path) for name, path in ordered}
    n_data = sum(len(pns) for pns in leg_pns.values())
    if n_data != len(inferred):
        raise ValueError(
            f"data-line count {n_data} != inferred TOA count {len(inferred)}; "
            "INCLUDE load/order mismatch"
        )

    # Steps 2-3: one plurality integer offset per leg.
    leg_offsets: dict[str, _LegOffset] = {}
    global_pns: dict[str, list[int]] = {}
    cursor = 0
    for name, _path in ordered:
        leg = leg_pns[name]
        window = inferred[cursor : cursor + len(leg)]
        cursor += len(leg)
        if not leg:
            leg_offsets[name] = _LegOffset(0, 0, 1.0, 0)
            global_pns[name] = []
            continue
        result = _modal_offset([inf - pn for inf, pn in zip(window, leg)])
        leg_offsets[name] = result
        global_pns[name] = [pn + result.offset for pn in leg]

    # Steps 4-5: re-origin so the reference leg's first TOA reads 0, then write.
    anchor = global_pns[ordered[0][0]][0]
    for name, path in ordered:
        relative = [g - anchor for g in global_pns[name]]
        written = _rewrite_tim_pn_sequential(path, relative)
        if written != len(relative):
            raise ValueError(
                f"{path}: expected to rewrite {len(relative)} data TOAs, "
                f"rewrote {written}"
            )

    return CombinationPnStats(
        n_toas=n_data,
        pn0_abs=anchor,
        tzrmjd=tzrmjd,
        tzrfrq=tzrfrq,
        tzrsite=tzrsite,
        per_pta_offset={n: o.offset for n, o in leg_offsets.items()},
        per_pta_n_toas={n: o.n_toas for n, o in leg_offsets.items()},
        per_pta_mode_fraction={n: o.mode_fraction for n, o in leg_offsets.items()},
        per_pta_max_deviation={n: o.max_deviation for n, o in leg_offsets.items()},
    )
