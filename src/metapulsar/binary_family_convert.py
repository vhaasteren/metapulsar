"""Gated ELL1-family → DD/DDH conversion for mixed-engine shared combination.

Implements Contracts 1–2 of ``feature_ell1h_truncation_fixw_nltiming.md``:
classification / scale gate (``decide_binary_conversion``) and binary-owned
patch conversion with mandatory delay-fidelity checks (``convert_shared_binary``).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from io import StringIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
from loguru import logger

from metapulsar.parameter_manager import normalize_timing_package
from metapulsar.pint_helpers import (
    Ell1hShapiroMode,
    create_pint_model,
    dict_to_parfile_string,
    get_aliases_for_parameter,
)

if TYPE_CHECKING:  # pragma: no cover
    from metapulsar.parameter_manager import AlignmentPolicy


# ---------------------------------------------------------------------------
# Public types (§4.3)
# ---------------------------------------------------------------------------


class BinaryConversionError(ValueError):
    """Shared binary cannot be converted under the active AlignmentPolicy."""


@dataclass(frozen=True)
class BinaryScaleGate:
    a1_lt_s: float  # |A1| at reference
    a1_max_lt_s: float  # span-aware max |A1|
    e_ref: float  # hypot(EPS1, EPS2) at reference
    e_max: float  # span-aware max e
    scale_s: float  # a1_max*e_max**2 + 0.5*nb*a1_max**2*e_max (§6.3 rev 2)
    threshold_s: float
    span_known: bool  # False iff dots present but START/FINISH absent


@dataclass(frozen=True)
class BinaryConversionDecision:
    outcome: Literal["convert", "skip", "unsupported"]
    reason: str
    source_family: Optional[str]
    target_family: Optional[str]
    scale: Optional[BinaryScaleGate]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BinaryPatch:
    """Binary-owned mutation only; application defined in §8.4."""

    binary_value: str  # "DD" or "DDH"
    removed_keys: tuple[str, ...]
    added_lines: tuple[tuple[str, str], ...]  # (KEY, "value [fit] [unc]")


@dataclass(frozen=True)
class BinaryFidelityReport:
    grid_points_per_orbit: int
    anchor_epochs_mjd: tuple[float, ...]
    total_max_abs_s: float
    shapiro_max_abs_s: Optional[float]
    roemer_max_abs_s: float
    tolerance_total_s: float
    tolerance_shapiro_s: Optional[float]
    tolerance_roemer_s: float


@dataclass(frozen=True)
class BinaryConversionRecord:
    pta_names: tuple[str, ...]
    source_free_params: tuple[str, ...]
    target_free_params: tuple[str, ...]
    fidelity: BinaryFidelityReport
    patch: BinaryPatch
    gauge: Optional[str] = None  # "absorbed" | "full"; None on plain
    required_sampling: tuple[str, ...] = ()
    stigma_provenance: Optional[str] = None
    uncertainty_propagation: str = "diagonal_jacobian"
    """How target uncertainties were obtained (§7.6, always APPROXIMATE).

    ``diagonal_jacobian`` — differenced through the full §7.6 map, but
    diagonal-only: par files carry no covariances, so correlations between the
    source parameters are ignored and the target errors are neither exact nor
    conservative. ``pint_convert_binary`` — PINT's own ELL1→DD propagation
    (plain path), with the same caveat.
    """


@dataclass(frozen=True)
class BinaryConversionReport:
    """Exposed after every shared materialization (§8.5)."""

    decision: BinaryConversionDecision
    record: Optional[BinaryConversionRecord]


@dataclass(frozen=True)
class BinaryConversionMetadata:
    """Typed in-memory channel for Case-D required-sampling (§8.5a)."""

    target_family: str
    gauge: Optional[str]
    required_sampling: tuple[str, ...]
    stigma_central: Optional[float]
    stigma_provenance: Optional[str]


# ---------------------------------------------------------------------------
# Par-dict helpers
# ---------------------------------------------------------------------------

_SECDAY = 86400.0
_DAY_PER_YEAR = 365.25

# Alias closure §7.3: canonical ← accepted spellings
_ALIAS_TO_CANONICAL: dict[str, str] = {
    "E": "ECC",
    "XDOT": "A1DOT",
    "STIG": "STIGMA",
    "VARSIGMA": "STIGMA",
    "NHARM": "NHARMS",
}

_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "ECC": ("ECC", "E"),
    "A1DOT": ("A1DOT", "XDOT"),
    "STIGMA": ("STIGMA", "STIG", "VARSIGMA"),
    "NHARMS": ("NHARMS", "NHARM"),
}

# Binary-owned key universe (§7.3), alias-resolved canonical names
BINARY_OWNED_CANONICAL: frozenset[str] = frozenset(
    {
        "BINARY",
        "PB",
        "PBDOT",
        "A1",
        "A1DOT",
        "ECC",
        "OM",
        "T0",
        "EDOT",
        "OMDOT",
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "TASC",
        "M2",
        "SINI",
        "GAMMA",
        "H3",
        "H4",
        "STIGMA",
        "NHARMS",
    }
)

_ORTHOMETRIC_KEYS = frozenset(
    {"H3", "H4", "STIG", "STIGMA", "VARSIGMA", "NHARM", "NHARMS"}
)

_REMEDIATIONS = (
    'exclude "binary" from combine_components (each PTA keeps its engine-native '
    "binary; no cross-engine binary contract exists, so no parity requirement applies)",
    'use combination_strategy="per_pta"',
    "use a single timing package for the stack",
    'set AlignmentPolicy(unsupported_binary="keep") to proceed with the documented '
    "O(A1*e^2) cross-engine floor",
    "for ELL1H H3-only: supply the orthometric ratio (STIGMA or H4 from another "
    'solution of the same pulsar), or set h3_only="sample_stigma" with '
    "stigma_central/stigma_provenance and commit the analysis to the STIGMA "
    "contract, or use remediation 1–4",
)


def remediation_message() -> str:
    """Return the five remediations from §5.4 as a numbered list."""
    lines = ["Remediations:"]
    for i, text in enumerate(_REMEDIATIONS, start=1):
        lines.append(f"  {i}. {text}")
    return "\n".join(lines)


def _canon_key(key: str) -> str:
    upper = key.upper()
    if upper.startswith("FB"):
        return upper
    return _ALIAS_TO_CANONICAL.get(upper, upper)


def _find_key(par: Mapping[str, Any], *names: str) -> Optional[str]:
    """Return the actual dict key matching any of ``names`` (case-insensitive)."""
    wanted = {n.upper() for n in names}
    for key in par:
        if key.upper() in wanted:
            return key
    return None


def _line_tokens(par: Mapping[str, Any], *names: str) -> Optional[list[str]]:
    key = _find_key(par, *names)
    if key is None:
        return None
    entries = par[key]
    if not entries:
        return None
    first = entries[0] if isinstance(entries, list) else entries
    return str(first).split()


def _param_str(par: Mapping[str, Any], *names: str) -> Optional[str]:
    tokens = _line_tokens(par, *names)
    if tokens is None:
        return None
    return tokens[0]


def _param_float(
    par: Mapping[str, Any], *names: str, default: Optional[float] = None
) -> Optional[float]:
    raw = _param_str(par, *names)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _fit_flag(par: Mapping[str, Any], *names: str) -> bool:
    """True iff the parameter is present and fit flag token is ``1``."""
    tokens = _line_tokens(par, *names)
    if tokens is None:
        return False
    if len(tokens) < 2:
        return False
    return tokens[1] == "1"


def _uncertainty_str(par: Mapping[str, Any], *names: str) -> Optional[str]:
    tokens = _line_tokens(par, *names)
    if tokens is None or len(tokens) < 3:
        return None
    return tokens[2]


def _has_any(par: Mapping[str, Any], names: Iterable[str]) -> bool:
    return any(_find_key(par, n) is not None for n in names)


def _has_fb(par: Mapping[str, Any]) -> bool:
    return any(k.upper().startswith("FB") for k in par)


def _binary_value(par: Mapping[str, Any]) -> Optional[str]:
    raw = _param_str(par, "BINARY")
    return None if raw is None else raw.upper()


def _present_spellings(
    par: Mapping[str, Any], *canonical_or_alias: str
) -> tuple[str, ...]:
    """Return actual keys present for the given names (preserving source spelling)."""
    wanted = {n.upper() for n in canonical_or_alias}
    for canon, aliases in _CANONICAL_ALIASES.items():
        if canon in wanted or wanted.intersection(a.upper() for a in aliases):
            wanted.update(a.upper() for a in aliases)
            wanted.add(canon)
    return tuple(k for k in par if k.upper() in wanted)


def _binary_owned_snapshot(par: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Alias-resolved binary-owned block: canonical → tuple of raw line strings."""
    out: dict[str, tuple[str, ...]] = {}
    for key, entries in par.items():
        canon = _canon_key(key)
        if canon.startswith("FB"):
            out[key.upper()] = tuple(str(e) for e in entries)
            continue
        if canon not in BINARY_OWNED_CANONICAL:
            continue
        # Prefer first-seen spelling; collapse aliases under canonical
        if canon in out:
            continue
        out[canon] = tuple(str(e) for e in entries)
    return out


# Keys whose value is an MJD epoch. At MJD ~5.5e4, 16 significant digits
# resolve only ~2.2e-7 s, and an epoch error delta enters the delay as
# nb*A1*delta -- 2.4e-11 s at J2145 scale, i.e. ~24% of the 1e-10 fidelity
# floor spent on formatting alone. 18 digits drops that to 3.3e-13 s. Other
# keys keep 16 digits: their round-trip error is already orders below the
# floor and the extra digits would only add noise to the written par.
_EPOCH_KEYS = frozenset({"T0", "TASC", "PEPOCH", "POSEPOCH", "DMEPOCH", "T0ASC"})
_EPOCH_PRECISION = 18
_DEFAULT_PRECISION = 16


def _format_line(
    value: Any, fit: bool, unc: Optional[str] = None, *, key: Optional[str] = None
) -> str:
    if isinstance(value, (float, np.floating)):
        precision = (
            _EPOCH_PRECISION
            if key is not None and _canon_key(key) in _EPOCH_KEYS
            else _DEFAULT_PRECISION
        )
        text = f"{np.longdouble(value):.{precision}g}"
    else:
        text = str(value)
    parts = [text, "1" if fit else "0"]
    if unc is not None:
        parts.append(str(unc))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Family classification (§6.2 D4)
# ---------------------------------------------------------------------------


def _classify_family(par: Mapping[str, Any]) -> Optional[str]:
    binary = _binary_value(par)
    if binary is None:
        return None
    has_h = _has_any(par, _ORTHOMETRIC_KEYS)
    if binary == "ELL1H":
        return "ELL1H"
    if binary == "ELL1K":
        return "ELL1k"
    if binary == "ELL1":
        return "ELL1H" if has_h else "ELL1"
    if binary == "T2":
        # §5.1/§5.5: only T2 *with EPS* is ELL1-family. A T2-Kepler par
        # (ECC/OM/T0, no EPS) is not, even when it carries H-terms — the gate
        # must never fire on it, and the orthometric map has no Laplace
        # coordinates to transfer.
        if not _has_any(par, ("EPS1", "EPS2")):
            return None  # T2-Kepler
        return "ELL1H" if has_h else "T2-EPS"
    return None


def _is_plain_family(family: Optional[str]) -> bool:
    return family in {"ELL1", "T2-EPS"}


def _is_orthometric_family(family: Optional[str]) -> bool:
    return family == "ELL1H"


# ---------------------------------------------------------------------------
# Scale gate (§6.3)
# ---------------------------------------------------------------------------


def _compute_scale_gate(par: Mapping[str, Any], threshold_s: float) -> BinaryScaleGate:
    a1 = _param_float(par, "A1")
    if a1 is None:
        raise BinaryConversionError(
            f"missing_a1: BINARY block has no A1\n{remediation_message()}"
        )
    a1_ref = abs(float(a1))

    eps1 = _param_float(par, "EPS1", default=None)
    eps2 = _param_float(par, "EPS2", default=None)
    if eps1 is None and eps2 is None:
        ecc = _param_float(par, "ECC", "E", default=None)
        e_ref = abs(float(ecc)) if ecc is not None else 0.0
        eps1 = 0.0
        eps2 = e_ref  # unused when no EPS; e_ref from ECC
    else:
        eps1 = float(eps1 or 0.0)
        eps2 = float(eps2 or 0.0)
        e_ref = float(math.hypot(eps1, eps2))

    tasc = _param_float(par, "TASC", default=None)
    start = _param_float(par, "START", default=None)
    finish = _param_float(par, "FINISH", default=None)

    has_eps_dots = _has_any(par, ("EPS1DOT", "EPS2DOT"))
    has_a1_dots = _has_any(par, ("A1DOT", "XDOT"))
    dots_present = has_eps_dots or has_a1_dots
    span_known = True
    if dots_present and (start is None or finish is None or tasc is None):
        span_known = False

    def _dt(mjd: float) -> float:
        assert tasc is not None
        return (mjd - tasc) * _SECDAY

    e_max = e_ref
    if has_eps_dots and span_known and tasc is not None:
        e1d = float(_param_float(par, "EPS1DOT", default=0.0) or 0.0)
        e2d = float(_param_float(par, "EPS2DOT", default=0.0) or 0.0)
        for mjd in (start, finish):
            assert mjd is not None
            dt = _dt(mjd)
            e_max = max(e_max, math.hypot(eps1 + e1d * dt, eps2 + e2d * dt))

    a1_max = a1_ref
    if has_a1_dots and span_known and tasc is not None:
        a1dot = float(_param_float(par, "A1DOT", "XDOT", default=0.0) or 0.0)
        a1_val = float(a1)
        for mjd in (start, finish):
            assert mjd is not None
            a1_max = max(a1_max, abs(a1_val + a1dot * _dt(mjd)))

    pb = _param_float(par, "PB", default=None)
    if pb is None or pb == 0.0:
        nb = 0.0
    else:
        nb = 2.0 * math.pi / (float(pb) * _SECDAY)

    scale_s = a1_max * e_max**2 + 0.5 * nb * a1_max**2 * e_max
    return BinaryScaleGate(
        a1_lt_s=a1_ref,
        a1_max_lt_s=a1_max,
        e_ref=e_ref,
        e_max=e_max,
        scale_s=float(scale_s),
        threshold_s=float(threshold_s),
        span_known=span_known,
    )


# ---------------------------------------------------------------------------
# Fit-flag contract (§7.2)
# ---------------------------------------------------------------------------


def _check_fit_pattern(par: Mapping[str, Any], *, orthometric: bool) -> Optional[str]:
    """Return None if OK, else a detail string for unsupported_fit_pattern."""
    free_eps1 = _fit_flag(par, "EPS1")
    free_eps2 = _fit_flag(par, "EPS2")
    free_tasc = _fit_flag(par, "TASC")
    triple_states = {free_eps1, free_eps2, free_tasc}
    # Absent EPS counts as frozen; require all three present for a free triple
    has_triple = all(_find_key(par, n) is not None for n in ("EPS1", "EPS2", "TASC"))
    if not has_triple:
        # Degenerate: treat as frozen triple if none free
        if any((free_eps1, free_eps2, free_tasc)):
            return "partial Laplace triple (missing EPS1/EPS2/TASC key)"
        triple_free = False
    else:
        if len(triple_states) != 1:
            return (
                f"mixed Laplace triple flags "
                f"(EPS1={int(free_eps1)} EPS2={int(free_eps2)} TASC={int(free_tasc)})"
            )
        triple_free = free_eps1

    has_dots = _has_any(par, ("EPS1DOT", "EPS2DOT"))
    if has_dots:
        free_e1d = _fit_flag(par, "EPS1DOT")
        free_e2d = _fit_flag(par, "EPS2DOT")
        if free_e1d != free_e2d:
            return "split EPS1DOT/EPS2DOT fit flags"
        if free_e1d and not triple_free:
            return "free EPS dots require a free Laplace triple"

    if _fit_flag(par, "PB") and not triple_free:
        return "free PB requires a free Laplace triple"

    if orthometric:
        # Coupled sextet: A1, EPS1, EPS2, TASC, H3, STIGMA-or-H4
        stigma_key = _find_key(par, "STIGMA", "STIG", "VARSIGMA")
        h4_key = _find_key(par, "H4")
        ratio_key = stigma_key or h4_key
        names = ["A1", "EPS1", "EPS2", "TASC", "H3"]
        flags = [_fit_flag(par, n) for n in names]
        if ratio_key is not None:
            flags.append(_fit_flag(par, ratio_key))
            names.append(ratio_key)
        if len(set(flags)) > 1:
            detail = " ".join(f"{n}={int(f)}" for n, f in zip(names, flags))
            return f"mixed orthometric sextet flags ({detail})"
    return None


def _source_free_params(par: Mapping[str, Any]) -> tuple[str, ...]:
    names = (
        "A1",
        "EPS1",
        "EPS2",
        "TASC",
        "PB",
        "PBDOT",
        "A1DOT",
        "EPS1DOT",
        "EPS2DOT",
        "H3",
        "H4",
        "STIGMA",
        "M2",
        "SINI",
        "GAMMA",
    )
    free: list[str] = []
    for name in names:
        if name == "A1DOT":
            if _fit_flag(par, "A1DOT", "XDOT"):
                free.append("A1DOT")
        elif name == "STIGMA":
            if _fit_flag(par, "STIGMA", "STIG", "VARSIGMA"):
                free.append("STIGMA")
        elif _fit_flag(par, name):
            free.append(name)
    return tuple(free)


# ---------------------------------------------------------------------------
# Unsupported classification (§5.4 / D8)
# ---------------------------------------------------------------------------


def _ell1h_case(par: Mapping[str, Any]) -> Literal["A", "B", "C", "D", "invalid"]:
    h3 = _param_float(par, "H3", default=None)
    if h3 is None:
        return "invalid"
    if _has_any(par, ("STIGMA", "STIG", "VARSIGMA")):
        return "B"  # gauge decided later; A vs B is gauge, not presence
    if _find_key(par, "H4") is not None:
        return "C"
    return "D"


def _domain_ok(h3: float, stig: float, a1: float) -> bool:
    if not (0.0 < stig <= 1.0):
        return False
    if h3 <= 0.0:
        return False
    if stig == 0.0:
        return False
    return (4.0 * h3 / stig**2) < (0.01 * abs(a1))


def _case_c_tail_bound(h3: float, h4: float, nharms: int) -> float:
    """``4*r*stig**(N+1)/((N+1)*(1-stig))`` with r=H3**4/H4**3, stig=H4/H3."""
    stig = h4 / h3
    if stig <= 0.0 or stig >= 1.0:
        return float("inf")
    r = (h3**4) / (h4**3)
    n = float(nharms)
    return float(4.0 * r * stig ** (n + 1.0) / ((n + 1.0) * (1.0 - stig)))


def _classify_unsupported(
    par: Mapping[str, Any],
    family: str,
    policy: "AlignmentPolicy",
) -> Optional[str]:
    """Return a §5.4 reason code, or None if supported.

    Checked in the §5.4 table order — ELL1H domain, H4 tail, H3-only, ELL1k,
    FB, fit pattern last — so overlapping conditions report the reason the
    design table names first.
    """
    if _is_orthometric_family(family):
        h3 = _param_float(par, "H3", default=None)
        a1 = _param_float(par, "A1", default=None)
        if h3 is None or a1 is None:
            return "ell1h_domain_violation"
        case = _ell1h_case(par)
        if case in ("A", "B"):
            stig = float(_param_float(par, "STIGMA", "STIG", "VARSIGMA") or 0.0)
            if not _domain_ok(float(h3), stig, float(a1)):
                return "ell1h_domain_violation"
        elif case == "C":
            h4 = float(_param_float(par, "H4") or 0.0)
            stig = h4 / float(h3) if h3 else 0.0
            if not _domain_ok(float(h3), stig, float(a1)):
                return "ell1h_domain_violation"
            nharms_raw = _param_float(par, "NHARMS", "NHARM", default=7.0) or 7.0
            nharms = int(nharms_raw)
            bound = _case_c_tail_bound(float(h3), h4, nharms)
            if bound > policy.binary_conversion_threshold_s:
                return "ell1h_h4_tail_exceeds_tolerance"
        else:  # Case D
            if policy.h3_only == "error" or policy.stigma_central is None:
                return "ell1h_h3_only_underdetermined"
            # sample_stigma: domain at stigma_central
            stig = float(policy.stigma_central)  # validated by AlignmentPolicy
            if not _domain_ok(float(h3), stig, float(a1)):
                return "ell1h_domain_violation"
            _require_nltiming_conversion_contract()

    elif _is_plain_family(family):
        if _has_any(par, _ORTHOMETRIC_KEYS):
            # Defensive: _classify_family routes H-terms to ELL1H.
            return "ell1h_h3_only_underdetermined"
    else:  # ELL1k and anything else that reached D8
        return "ell1k_secular_terms_unvalidated"

    if _has_fb(par) or _param_float(par, "PB", default=None) is None:
        return "fb_orbital_series_unsupported"

    if _check_fit_pattern(par, orthometric=_is_orthometric_family(family)) is not None:
        return "unsupported_fit_pattern"
    return None


def _require_nltiming_conversion_contract() -> None:
    """§8.5a landing gate: Case D needs the nltiming metadata contract.

    Raises rather than returning an ``unsupported`` reason code: the gate is a
    hard §18 lock, so ``unsupported_binary="keep"`` must not downgrade it to a
    warning.
    """
    try:
        import nltiming
    except ImportError:  # pragma: no cover - nltiming is a hard dependency
        supported = False
    else:
        supported = bool(getattr(nltiming, "SUPPORTS_CONVERSION_METADATA", False))
    if not supported:
        raise BinaryConversionError(
            "sample_stigma_requires_nltiming_contract: h3_only='sample_stigma' "
            "requires an nltiming build advertising SUPPORTS_CONVERSION_METADATA "
            "(the required_sampling channel of §8.5a)\n"
            f"{remediation_message()}"
        )


def _unsupported_message(
    reason: str,
    scale: Optional[BinaryScaleGate],
    *,
    detail: str = "",
    par: Optional[Mapping[str, Any]] = None,
    policy: Optional["AlignmentPolicy"] = None,
) -> str:
    parts = [reason]
    if scale is not None:
        parts.append(
            f"scale_s={scale.scale_s:.6e} threshold_s={scale.threshold_s:.6e} "
            f"a1_max={scale.a1_max_lt_s:.6g} e_max={scale.e_max:.6g}"
        )
    if reason == "ell1h_h4_tail_exceeds_tolerance" and par is not None:
        h3 = float(_param_float(par, "H3") or 0.0)
        h4 = float(_param_float(par, "H4") or 0.0)
        nharms = int(_param_float(par, "NHARMS", "NHARM", default=7.0) or 7.0)
        bound = _case_c_tail_bound(h3, h4, nharms)
        parts.append(f"tail_bound_s={bound:.6e} NHARMS={nharms}")
    if reason == "unsupported_fit_pattern" and detail:
        parts.append(detail)
    if reason == "ell1h_h3_only_underdetermined":
        parts.append(
            "closure menu: supply STIGMA or H4; or set "
            'h3_only="sample_stigma" with stigma_central/stigma_provenance; '
            "or use remediations 1–4"
        )
    parts.append(remediation_message())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Contract 1 — decide_binary_conversion (§6)
# ---------------------------------------------------------------------------


def decide_binary_conversion(
    parfile_dicts: Mapping[str, dict],
    *,
    reference_pta: str,
    timing_packages: Mapping[str, Optional[str]],
    combine_components: Sequence[str],
    policy: "AlignmentPolicy",
) -> BinaryConversionDecision:
    """Pure classification + scale gate. No mutation, no I/O, no model building."""
    # D1
    if policy.binary_conversion == "off":
        return BinaryConversionDecision(
            outcome="skip",
            reason="policy_off",
            source_family=None,
            target_family=None,
            scale=None,
        )

    # D2
    if "binary" not in combine_components:
        return BinaryConversionDecision(
            outcome="skip",
            reason="binary_not_shared",
            source_family=None,
            target_family=None,
            scale=None,
        )

    # D3
    packages = {normalize_timing_package(p) for p in timing_packages.values()}
    if not {"pint", "tempo2"}.issubset(packages):
        return BinaryConversionDecision(
            outcome="skip",
            reason="single_engine_stack",
            source_family=None,
            target_family=None,
            scale=None,
        )

    if reference_pta not in parfile_dicts:
        raise BinaryConversionError(
            f"reference_pta={reference_pta!r} missing from parfile_dicts"
        )
    ref = parfile_dicts[reference_pta]

    # D4
    if _binary_value(ref) is None:
        return BinaryConversionDecision(
            outcome="skip",
            reason="no_binary",
            source_family=None,
            target_family=None,
            scale=None,
        )
    family = _classify_family(ref)
    if family is None:
        return BinaryConversionDecision(
            outcome="skip",
            reason="not_ell1_family",
            source_family=None,
            target_family=None,
            scale=None,
        )

    # D5 — binary-owned blocks must be string-identical across PTAs
    ref_block = _binary_owned_snapshot(ref)
    for pta, par in parfile_dicts.items():
        if pta == reference_pta:
            continue
        other = _binary_owned_snapshot(par)
        if other != ref_block:
            raise BinaryConversionError(
                "shared_binary_blocks_diverge: binary-owned keys differ across "
                f"PTAs after shared merge (reference={reference_pta}, other={pta})"
            )

    # D6
    scale = _compute_scale_gate(ref, policy.binary_conversion_threshold_s)

    # D7 / D7b (auto mode only; "always" skips D7)
    if policy.binary_conversion == "auto":
        if scale.scale_s <= scale.threshold_s:
            if scale.span_known:
                return BinaryConversionDecision(
                    outcome="skip",
                    reason="below_threshold",
                    source_family=family,
                    target_family=None,
                    scale=scale,
                )
            # D7b: sub-threshold with unknown span
            return BinaryConversionDecision(
                outcome="unsupported",
                reason="gate_span_unknown",
                source_family=family,
                target_family=None,
                scale=scale,
            )

    # D8
    unsup = _classify_unsupported(ref, family, policy)
    if unsup is not None:
        fit_detail = ""
        if unsup == "unsupported_fit_pattern":
            fit_detail = (
                _check_fit_pattern(ref, orthometric=_is_orthometric_family(family))
                or ""
            )
        return BinaryConversionDecision(
            outcome="unsupported",
            reason=unsup,
            source_family=family,
            target_family=None,
            scale=scale,
            warnings=(fit_detail,) if fit_detail else (),
        )

    # D9
    target = "DDH" if _is_orthometric_family(family) else "DD"
    return BinaryConversionDecision(
        outcome="convert",
        reason="gate_fired",
        source_family=family,
        target_family=target,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# §7.6 maps (normative; np.longdouble)
# ---------------------------------------------------------------------------


def _absorbed_to_intrinsic(
    x_p: Any, e1p: Any, e2p: Any, h3: Any, stig: Any, nb: Any
) -> tuple[Any, Any, Any]:
    """Undo the FW10 absorbed gauge (writeup Theorem 1 + Sec. 5.3)."""
    x_p = np.longdouble(x_p)
    e1p = np.longdouble(e1p)
    e2p = np.longdouble(e2p)
    h3 = np.longdouble(h3)
    stig = np.longdouble(stig)
    nb = np.longdouble(nb)
    x_i = x_p - 4.0 * h3 / stig**2
    p1 = x_p * e1p - 4.0 * h3 / stig
    p2 = x_p * e2p - 8.0 * nb * (h3 / stig**2) * x_i
    return x_i, p1 / x_i, p2 / x_i


def _kepler_map(
    eps1: Any, eps2: Any, tasc_mjd: Any, pb_days: Any
) -> tuple[Any, Any, Any]:
    eps1 = np.longdouble(eps1)
    eps2 = np.longdouble(eps2)
    ecc = np.hypot(eps1, eps2)
    om = np.arctan2(eps1, eps2) % (2.0 * np.pi)  # eps==0 -> om = 0
    t0_mjd = np.longdouble(tasc_mjd) + np.longdouble(pb_days) * om / (2.0 * np.pi)
    return ecc, om, t0_mjd


def _delta_t0_seconds(x: Any, eps1: Any, h3: Any = 0.0, stig: Any = None) -> Any:
    """(3/2)x*eps1 series-constant leak + optional h3/stig absorbed term."""
    dt0 = np.longdouble(1.5) * np.longdouble(x) * np.longdouble(eps1)
    if stig is not None:
        dt0 = dt0 + np.longdouble(h3) / np.longdouble(stig)
    return dt0


def _intrinsic_dots(
    x_p: Any,
    e1p: Any,
    e2p: Any,
    e1i: Any,
    e2i: Any,
    x_i: Any,
    xdot: Any,
    e1dot_p: Any,
    e2dot_p: Any,
    h3: Any = 0.0,
    stig: Any = None,
    nb: Any = 0.0,
) -> tuple[Any, Any]:
    """Exact time derivative of the transferred coordinates."""
    x_p = np.longdouble(x_p)
    e1p = np.longdouble(e1p)
    e2p = np.longdouble(e2p)
    e1i = np.longdouble(e1i)
    e2i = np.longdouble(e2i)
    x_i = np.longdouble(x_i)
    xdot = np.longdouble(xdot)
    e1dot_p = np.longdouble(e1dot_p)
    e2dot_p = np.longdouble(e2dot_p)
    e1dot_i = (xdot * e1p + x_p * e1dot_p - xdot * e1i) / x_i
    corr2 = (
        8.0 * np.longdouble(nb) * (np.longdouble(h3) / np.longdouble(stig) ** 2) * xdot
        if stig is not None
        else np.longdouble(0.0)
    )
    e2dot_i = (xdot * e2p + x_p * e2dot_p - corr2 - xdot * e2i) / x_i
    return e1dot_i, e2dot_i


def _rereference_dots(params: Any, tau_s: float) -> None:
    """TASC→T0 epoch re-referencing of every t-linear parameter.

    ``params`` may be a :class:`SimpleNamespace` (OM in radians, OMDOT in rad/s)
    or a PINT TimingModel (OM in degrees, OMDOT in deg/yr).
    """
    tau_s = float(np.longdouble(tau_s))
    if hasattr(params, "params"):  # TimingModel duck
        pb = np.longdouble(params.PB.value)
        pbdot = np.longdouble(_model_value(params, "PBDOT"))
        params.PB.value = float(pb + pbdot * tau_s / _SECDAY)
        a1 = np.longdouble(params.A1.value)
        a1dot = np.longdouble(_model_value(params, "A1DOT"))
        params.A1.value = float(a1 + a1dot * tau_s)
        if hasattr(params, "ECC") and params.ECC.value is not None:
            edot = np.longdouble(_model_value(params, "EDOT"))
            params.ECC.value = float(np.longdouble(params.ECC.value) + edot * tau_s)
        if hasattr(params, "OM") and params.OM.value is not None:
            # PINT OMDOT is deg/yr
            omdot = np.longdouble(_model_value(params, "OMDOT"))
            params.OM.value = float(
                np.longdouble(params.OM.value)
                + omdot * tau_s / (_SECDAY * _DAY_PER_YEAR)
            )
        return

    params.PB = np.longdouble(params.PB) + np.longdouble(params.PBDOT) * tau_s / _SECDAY
    params.A1 = np.longdouble(params.A1) + np.longdouble(params.A1DOT) * tau_s
    params.ECC = np.longdouble(params.ECC) + np.longdouble(params.EDOT) * tau_s
    params.OM = np.longdouble(params.OM) + np.longdouble(params.OMDOT) * tau_s


def _model_value(model: Any, name: str, default: float = 0.0) -> float:
    if not hasattr(model, name):
        return default
    param = getattr(model, name)
    if param is None or getattr(param, "value", None) is None:
        return default
    try:
        quantity = param.quantity
        if quantity is not None:
            # Prefer SI-ish bases where unambiguous
            unit = getattr(param, "units", None)
            if unit is not None and str(unit) in {"", "1", "dimensionless_unscaled"}:
                return float(np.longdouble(param.value))
            return float(np.longdouble(param.value))
    except Exception:
        pass
    return float(np.longdouble(param.value))


def _check_domain(h3: float, stig: float, a1: float) -> None:
    if not _domain_ok(float(h3), float(stig), float(a1)):
        raise BinaryConversionError(
            "ell1h_domain_violation: require 0 < stig <= 1, H3 > 0, "
            f"and 4*H3/stig**2 < 0.01*A1 (H3={h3}, stig={stig}, A1={a1})\n"
            f"{remediation_message()}"
        )


def apply_conversion_corrections(dd_model: Any, source_model: Any) -> None:
    """Plain-path post-processing after ``pint.binaryconvert`` (C2b)."""
    a1 = _model_value(source_model, "A1")
    eps1 = _model_value(source_model, "EPS1")
    dt0 = _delta_t0_seconds(a1, eps1)
    dd_model.T0.value = float(np.longdouble(dd_model.T0.value) + dt0 / _SECDAY)
    tasc = _model_value(source_model, "TASC")
    tau = (np.longdouble(dd_model.T0.value) - np.longdouble(tasc)) * _SECDAY
    _rereference_dots(dd_model, float(tau))


def _ddh_map(
    *,
    a1_p: Any,
    e1p: Any,
    e2p: Any,
    tasc: Any,
    pb: Any,
    h3: Any,
    stig: Any,
    xdot: Any,
    e1dot_p: Any,
    e2dot_p: Any,
    pbdot: Any,
    absorbed: bool,
) -> SimpleNamespace:
    """Pure §7.6 ELL1H → DDH map (no model objects, no I/O).

    Inputs are printed-par units (A1 lt-s, TASC/PB days, H3 s); outputs carry
    ``OM`` in radians and ``OMDOT`` in rad/s. Isolated from
    :func:`_convert_ell1h_block` so the uncertainty Jacobian (§7.6) can
    difference the exact same map that produces the values.
    """
    a1_p = np.longdouble(a1_p)
    e1p = np.longdouble(e1p)
    e2p = np.longdouble(e2p)
    tasc = np.longdouble(tasc)
    pb = np.longdouble(pb)
    h3 = np.longdouble(h3)
    stig = np.longdouble(stig)
    xdot = np.longdouble(xdot)
    e1dot_p = np.longdouble(e1dot_p)
    e2dot_p = np.longdouble(e2dot_p)
    pbdot = np.longdouble(pbdot)

    pb_s = pb * _SECDAY
    nb = 2.0 * np.pi / pb_s if pb_s else np.longdouble(0.0)

    if absorbed:
        x_i, e1i, e2i = _absorbed_to_intrinsic(a1_p, e1p, e2p, h3, stig, nb)
        dt0 = _delta_t0_seconds(x_i, e1i, h3, stig)
        stig_for_dots: Optional[float] = float(stig)
    else:
        x_i, e1i, e2i = a1_p, e1p, e2p
        dt0 = _delta_t0_seconds(x_i, e1i)
        stig_for_dots = None

    ecc, om_rad, t0 = _kepler_map(e1i, e2i, tasc, pb)
    t0 = t0 + dt0 / _SECDAY
    tau = (t0 - tasc) * _SECDAY
    e1dot_i, e2dot_i = _intrinsic_dots(
        a1_p,
        e1p,
        e2p,
        e1i,
        e2i,
        x_i,
        xdot,
        e1dot_p,
        e2dot_p,
        h3=float(h3),
        stig=stig_for_dots,
        nb=float(nb),
    )
    if ecc:
        edot = (e1i * e1dot_i + e2i * e2dot_i) / ecc
        omdot_rad_s = (e1dot_i * e2i - e2dot_i * e1i) / ecc**2
    else:
        edot = np.longdouble(0.0)
        omdot_rad_s = np.longdouble(0.0)

    out = SimpleNamespace(
        A1=x_i,
        PB=pb,
        ECC=ecc,
        OM=om_rad,  # radians
        T0=t0,
        H3=h3,
        STIGMA=stig,
        EDOT=edot,
        OMDOT=omdot_rad_s,  # rad/s
        PBDOT=pbdot,
        A1DOT=xdot,
    )
    _rereference_dots(out, float(tau))
    return out


# Outputs the §7.6 uncertainty Jacobian reports, and the unit conversion from
# the `_ddh_map` frame (OM rad, OMDOT rad/s) to the par frame (deg, deg/yr).
_DDH_UNC_OUTPUT_SCALE: dict[str, float] = {
    "A1": 1.0,
    "PB": 1.0,
    "ECC": 1.0,
    "OM": 180.0 / math.pi,
    "T0": 1.0,
    "H3": 1.0,
    "STIGMA": 1.0,
    "EDOT": 1.0,
    "OMDOT": (180.0 / math.pi) * _SECDAY * _DAY_PER_YEAR,
}


def _propagate_ddh_uncertainties(
    base: Mapping[str, Any], sigmas: Mapping[str, float], *, absorbed: bool
) -> dict[str, float]:
    """Diagonal-only uncertainty propagation through the full §7.6 Jacobian.

    Par files carry no covariances, so this is the diagonal approximation —
    documented as approximate on :class:`BinaryConversionRecord`. In the
    absorbed gauge H3/STIGMA genuinely feed A1/ECC/OM/T0, so dropping those
    partials (as a source-uncertainty passthrough would) understates the output
    errors. Partials come from central differences on :func:`_ddh_map` itself,
    so they can never drift from the values that map produces.
    """
    variance = {name: 0.0 for name in _DDH_UNC_OUTPUT_SCALE}
    for key, sigma in sigmas.items():
        if key not in base:
            continue
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            continue
        x0 = np.longdouble(base[key])
        step = np.longdouble(1e-6) * max(
            abs(np.longdouble(x0)), abs(np.longdouble(sigma)), np.longdouble(1e-30)
        )
        plus = _ddh_map(**{**dict(base), key: x0 + step}, absorbed=absorbed)
        minus = _ddh_map(**{**dict(base), key: x0 - step}, absorbed=absorbed)
        for name, scale in _DDH_UNC_OUTPUT_SCALE.items():
            deriv = (
                np.longdouble(getattr(plus, name)) - np.longdouble(getattr(minus, name))
            ) / (2.0 * step)
            variance[name] += float(deriv * scale * sigma) ** 2
    return {name: math.sqrt(value) for name, value in variance.items()}


def _source_uncertainties(par: Mapping[str, Any]) -> dict[str, float]:
    """Read per-parameter 1-sigma values off the source par lines (token 2)."""
    wanted = {
        "a1_p": ("A1",),
        "e1p": ("EPS1",),
        "e2p": ("EPS2",),
        "tasc": ("TASC",),
        "pb": ("PB",),
        "h3": ("H3",),
        "stig": ("STIGMA", "STIG", "VARSIGMA"),
        "xdot": ("A1DOT", "XDOT"),
        "e1dot_p": ("EPS1DOT",),
        "e2dot_p": ("EPS2DOT",),
        "pbdot": ("PBDOT",),
    }
    out: dict[str, float] = {}
    for key, names in wanted.items():
        raw = _uncertainty_str(par, *names)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            out[key] = value
    return out


def _convert_ell1h_block(
    model: Any,
    gauge: Ell1hShapiroMode,
    policy: "AlignmentPolicy",
    *,
    source_uncertainties: Optional[Mapping[str, float]] = None,
) -> tuple[Any, dict[str, float]]:
    """Cases A/B/C/D → (DDH TimingModel, propagated uncertainties).

    First-party map; ``convert_binary`` is never used here.
    """
    h3 = np.longdouble(_model_value(model, "H3"))
    a1_p = np.longdouble(_model_value(model, "A1"))

    has_stig = (
        hasattr(model, "STIGMA")
        and model.STIGMA is not None
        and model.STIGMA.value is not None
    )
    has_h4 = (
        hasattr(model, "H4") and model.H4 is not None and model.H4.value is not None
    )

    if has_stig:
        stig = np.longdouble(_model_value(model, "STIGMA"))
        case = "B" if gauge == "absorbed" else "A"
    elif has_h4:
        stig = np.longdouble(_model_value(model, "H4")) / h3
        case = "C"
    else:
        if policy.stigma_central is None:
            raise BinaryConversionError(
                "ell1h_h3_only_underdetermined: no stigma_central for Case D\n"
                f"{remediation_message()}"
            )
        stig = np.longdouble(policy.stigma_central)
        case = "D"

    _check_domain(float(h3), float(stig), float(a1_p))

    base = {
        "a1_p": a1_p,
        "e1p": np.longdouble(_model_value(model, "EPS1")),
        "e2p": np.longdouble(_model_value(model, "EPS2")),
        "tasc": np.longdouble(_model_value(model, "TASC")),
        "pb": np.longdouble(_model_value(model, "PB")),
        "h3": h3,
        "stig": stig,
        "xdot": np.longdouble(_model_value(model, "A1DOT")),
        "e1dot_p": np.longdouble(_model_value(model, "EPS1DOT")),
        "e2dot_p": np.longdouble(_model_value(model, "EPS2DOT")),
        "pbdot": np.longdouble(_model_value(model, "PBDOT")),
    }
    absorbed = case in ("B", "C", "D")
    out = _ddh_map(**base, absorbed=absorbed)
    uncertainties = _propagate_ddh_uncertainties(
        base, source_uncertainties or {}, absorbed=absorbed
    )

    # Build a TimingModel by rewriting the source par text onto BINARY DDH.
    model_out = _timing_model_from_ddh_params(
        model, out, case=case, policy=policy, uncertainties=uncertainties
    )
    return model_out, uncertainties


def _timing_model_from_ddh_params(
    source_model: Any,
    params: SimpleNamespace,
    *,
    case: str,
    policy: "AlignmentPolicy",
    uncertainties: Optional[Mapping[str, float]] = None,
) -> Any:
    """Construct a BinaryDDH TimingModel carrying ``params``."""
    from pint.models.model_builder import parse_parfile

    # Start from the source as_parfile, then splice binary keys.
    text = source_model.as_parfile()
    par = parse_parfile(StringIO(text))

    # Remove ELL1-family keys
    for name in (
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "TASC",
        "NHARM",
        "NHARMS",
        "H4",
        "STIG",
        "STIGMA",
        "VARSIGMA",
        "ECC",
        "OM",
        "T0",
        "EDOT",
        "OMDOT",
        "E",
    ):
        key = _find_key(par, name)
        if key is not None:
            del par[key]

    bin_key = _find_key(par, "BINARY")
    if bin_key is None:
        raise BinaryConversionError("unexpected_converter_output: missing BINARY")
    par[bin_key] = ["DDH"]

    om_deg = float(np.degrees(np.longdouble(params.OM)))
    omdot_deg_yr = float(
        np.degrees(np.longdouble(params.OMDOT)) * _SECDAY * _DAY_PER_YEAR
    )

    unc_map = dict(uncertainties or {})

    def _unc(name: str) -> Optional[str]:
        """Propagated 1-sigma for an output key, or None when unavailable."""
        value = unc_map.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            return None
        return f"{float(value):.16g}"

    def _set(
        name: str, value: Any, fit: bool = False, unc: Optional[str] = None
    ) -> None:
        key = _find_key(par, name) or name
        par[key] = [_format_line(value, fit, unc, key=name)]

    # Values and uncertainties both come from the §7.6 map (the absorbed gauge
    # mixes H3/STIGMA into A1/ECC/OM/T0, so source errors cannot pass through).
    a1_fit = not getattr(source_model.A1, "frozen", True)
    _set("A1", float(params.A1), a1_fit, _unc("A1"))

    pb_fit = not getattr(source_model.PB, "frozen", True)
    _set("PB", float(params.PB), pb_fit, _unc("PB"))

    triple_free = (
        not getattr(source_model.EPS1, "frozen", True)
        if hasattr(source_model, "EPS1")
        else False
    )
    _set("ECC", float(params.ECC), triple_free, _unc("ECC"))
    _set("OM", om_deg, triple_free, _unc("OM"))
    _set("T0", float(params.T0), triple_free, _unc("T0"))

    # Emit EDOT/OMDOT when source had EPS dots
    if hasattr(source_model, "EPS1DOT") and source_model.EPS1DOT.value is not None:
        dots_free = not source_model.EPS1DOT.frozen
        _set("EDOT", float(params.EDOT), dots_free, _unc("EDOT"))
        _set("OMDOT", omdot_deg_yr, dots_free, _unc("OMDOT"))

    if _model_value(source_model, "PBDOT") != 0.0 or (
        hasattr(source_model, "PBDOT") and source_model.PBDOT.value is not None
    ):
        pbdot_fit = not getattr(source_model.PBDOT, "frozen", True)
        _set("PBDOT", float(params.PBDOT), pbdot_fit)

    if _model_value(source_model, "A1DOT") != 0.0 or (
        hasattr(source_model, "A1DOT") and source_model.A1DOT.value is not None
    ):
        a1dot_fit = not getattr(source_model.A1DOT, "frozen", True)
        _set("A1DOT", float(params.A1DOT), a1dot_fit)

    h3_fit = not getattr(source_model.H3, "frozen", True)
    _set("H3", float(params.H3), h3_fit, _unc("H3"))

    stig_fit = False
    if hasattr(source_model, "STIGMA") and source_model.STIGMA.value is not None:
        stig_fit = not source_model.STIGMA.frozen
    elif hasattr(source_model, "H4") and source_model.H4.value is not None:
        stig_fit = not source_model.H4.frozen
    elif case == "D":
        stig_fit = True  # sampling axis
    _set("STIGMA", float(params.STIGMA), stig_fit, _unc("STIGMA"))

    # Fallback only: a source with no uncertainties at all propagates none, and
    # some downstream fitters reject a free parameter with no error column.
    for name in ("ECC", "OM", "T0"):
        key = _find_key(par, name)
        if key is None:
            continue
        tokens = str(par[key][0]).split()
        if len(tokens) == 2 and tokens[1] == "1":
            par[key] = [f"{tokens[0]} 1 1e-12"]

    # DDH has no gauge freedom, so ell1h_shapiro is irrelevant on reload.
    text_out = dict_to_parfile_string(par, format="pint")
    return create_pint_model(text_out, ell1h_shapiro="full")


# ---------------------------------------------------------------------------
# Patch construction (§7.3 / C4)
# ---------------------------------------------------------------------------


def _ensure_uncertainties_for_convert(model: Any) -> None:
    """``convert_binary`` can fail without usable EPS uncertainties; set tiny unc.

    PINT reports missing uncertainties as ``0.0`` (not ``None``); treat zero /
    non-finite the same as absent so the converter's OM/ECC error propagation
    does not see ``None`` Quantity objects.
    """
    import astropy.units as u

    for name in ("EPS1", "EPS2", "EPS1DOT", "EPS2DOT", "TASC", "PB", "A1"):
        if not hasattr(model, name):
            continue
        param = getattr(model, name)
        if param is None or param.value is None:
            continue
        try:
            unc = param.uncertainty_value
        except Exception:
            unc = None
        needs = unc is None or (
            isinstance(unc, (int, float, np.floating))
            and (not math.isfinite(float(unc)) or float(unc) == 0.0)
        )
        if not needs:
            continue
        try:
            unit = param.units if param.units is not None else u.dimensionless_unscaled
            scale = abs(float(param.value)) if float(param.value) != 0.0 else 1.0
            param.uncertainty = scale * 1e-12 * unit
        except Exception:
            pass


def _build_patch_plain(
    source_dict: Mapping[str, Any],
    converted_model: Any,
    *,
    triple_free: bool,
    dots_free: bool,
    had_dots: bool,
) -> BinaryPatch:
    removed: list[str] = []
    for name in ("EPS1", "EPS2", "EPS1DOT", "EPS2DOT", "TASC"):
        removed.extend(_present_spellings(source_dict, name))

    reemit_pb = _has_any(source_dict, ("PBDOT",))
    reemit_a1 = _has_any(source_dict, ("A1DOT", "XDOT"))
    if reemit_pb:
        removed.extend(_present_spellings(source_dict, "PB"))
    if reemit_a1:
        removed.extend(_present_spellings(source_dict, "A1"))

    added: list[tuple[str, str]] = []
    ecc_unc = _model_unc_str(converted_model, "ECC")
    om_unc = _model_unc_str(converted_model, "OM")
    t0_unc = _model_unc_str(converted_model, "T0")
    added.append(("ECC", _format_line(converted_model.ECC.value, triple_free, ecc_unc)))
    added.append(("OM", _format_line(converted_model.OM.value, triple_free, om_unc)))
    added.append(
        ("T0", _format_line(converted_model.T0.value, triple_free, t0_unc, key="T0"))
    )
    if had_dots:
        added.append(
            ("EDOT", _format_line(_model_value(converted_model, "EDOT"), dots_free))
        )
        added.append(
            ("OMDOT", _format_line(_model_value(converted_model, "OMDOT"), dots_free))
        )
    if reemit_pb:
        pb_fit = _fit_flag(source_dict, "PB")
        pb_unc = _uncertainty_str(source_dict, "PB")
        added.append(("PB", _format_line(converted_model.PB.value, pb_fit, pb_unc)))
    if reemit_a1:
        a1_key = _find_key(source_dict, "A1") or "A1"
        a1_fit = _fit_flag(source_dict, "A1")
        a1_unc = _uncertainty_str(source_dict, "A1")
        added.append((a1_key, _format_line(converted_model.A1.value, a1_fit, a1_unc)))

    return BinaryPatch(
        binary_value="DD",
        removed_keys=tuple(dict.fromkeys(removed)),
        added_lines=tuple(added),
    )


def _build_patch_orthometric(
    source_dict: Mapping[str, Any],
    converted_model: Any,
    *,
    triple_free: bool,
    dots_free: bool,
    had_dots: bool,
    case: str,
    policy: "AlignmentPolicy",
    uncertainties: Mapping[str, float],
) -> BinaryPatch:
    def _unc(name: str) -> Optional[str]:
        value = uncertainties.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            return None
        return f"{float(value):.16g}"

    removed: list[str] = []
    for name in (
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "TASC",
        "NHARM",
        "NHARMS",
        "H4",
        "STIG",
        "STIGMA",
        "VARSIGMA",
        "A1",
    ):
        removed.extend(_present_spellings(source_dict, name))
    reemit_pb = _has_any(source_dict, ("PBDOT",))
    if reemit_pb:
        removed.extend(_present_spellings(source_dict, "PB"))

    added: list[tuple[str, str]] = []
    a1_fit = _fit_flag(source_dict, "A1")
    added.append(("A1", _format_line(converted_model.A1.value, a1_fit, _unc("A1"))))
    added.append(
        ("ECC", _format_line(converted_model.ECC.value, triple_free, _unc("ECC")))
    )
    added.append(
        ("OM", _format_line(converted_model.OM.value, triple_free, _unc("OM")))
    )
    added.append(
        (
            "T0",
            _format_line(converted_model.T0.value, triple_free, _unc("T0"), key="T0"),
        )
    )
    if had_dots:
        added.append(
            (
                "EDOT",
                _format_line(
                    _model_value(converted_model, "EDOT"), dots_free, _unc("EDOT")
                ),
            )
        )
        added.append(
            (
                "OMDOT",
                _format_line(
                    _model_value(converted_model, "OMDOT"), dots_free, _unc("OMDOT")
                ),
            )
        )
    if reemit_pb:
        pb_fit = _fit_flag(source_dict, "PB")
        added.append(("PB", _format_line(converted_model.PB.value, pb_fit, _unc("PB"))))

    stig_fit = False
    if _has_any(source_dict, ("STIGMA", "STIG", "VARSIGMA")):
        stig_fit = _fit_flag(source_dict, "STIGMA", "STIG", "VARSIGMA")
    elif _has_any(source_dict, ("H4",)):
        stig_fit = _fit_flag(source_dict, "H4")
    elif case == "D":
        stig_fit = True
    added.append(
        (
            "STIGMA",
            _format_line(
                _model_value(converted_model, "STIGMA"), stig_fit, _unc("STIGMA")
            ),
        )
    )
    if case == "D" and policy.stigma_provenance:
        # Informational comment line — never load-bearing (§8.5a)
        added.append(
            (
                "C",
                f"MetaPulsar Case-D STIGMA prior center; provenance: "
                f"{policy.stigma_provenance}; not a standalone artifact",
            )
        )

    return BinaryPatch(
        binary_value="DDH",
        removed_keys=tuple(dict.fromkeys(removed)),
        added_lines=tuple(added),
    )


def _model_unc_str(model: Any, name: str) -> Optional[str]:
    if not hasattr(model, name):
        return None
    param = getattr(model, name)
    try:
        unc = param.uncertainty_value
    except Exception:
        return None
    if unc is None:
        return None
    return f"{float(unc):.16g}"


# ---------------------------------------------------------------------------
# C5 converter-output audit
# ---------------------------------------------------------------------------

_CONVERTER_DEFAULT_ZERO_KEYS = frozenset(
    {"EDOT", "OMDOT", "PBDOT", "A1DOT", "GAMMA", "XDOT"}
)


def _audit_converter_output(
    source_dict: Mapping[str, Any],
    converted_dict: Mapping[str, Any],
    patch: BinaryPatch,
    corrected_model: Any,
    source_model: Any,
) -> None:
    """C5 audit of binary-owned keys after conversion.

    Pass-through equality is alias-symmetric (PINT aliases via
    ``get_aliases_for_parameter``) and, for numeric axes present on
    ``source_model``, compares physical ``_model_value``s so Tempo-style
    ``unit_scale`` (e.g. XDOT −0.009436 → A1DOT −9.436e−15) does not false-fail.

    Frozen-zero converter defaults (EDOT/OMDOT/PBDOT/A1DOT/GAMMA/…) that are
    absent from both the source and the patch are ignorable extras — convert_binary
    emits them even when the source never had those axes. Documented here so the
    audit stays fail-closed on genuine unexpected keys without inventing a second
    plain-path converter.
    """
    src_keys = set(_binary_owned_snapshot(source_dict))
    # Also track FB*
    src_keys.update(k.upper() for k in source_dict if k.upper().startswith("FB"))

    removed_canon = {_canon_key(k) for k in patch.removed_keys}
    added_canon = {_canon_key(k) for k, _ in patch.added_lines if k.upper() != "C"}

    expected = (src_keys - removed_canon) | added_canon | {"BINARY"}
    # Target always has ECC/OM/T0 (or A1/STIGMA for DDH) via added_lines
    conv_keys = set(_binary_owned_snapshot(converted_dict))
    conv_keys.update(k.upper() for k in converted_dict if k.upper().startswith("FB"))

    extras = conv_keys - expected
    missing = expected - conv_keys
    # Ignorable frozen-zero defaults
    ignorable = set()
    for key in list(extras):
        if (
            key in _CONVERTER_DEFAULT_ZERO_KEYS
            or _canon_key(key) in _CONVERTER_DEFAULT_ZERO_KEYS
        ):
            val = _param_float(converted_dict, key, default=None)
            if val is None or abs(float(val)) == 0.0:
                if not _fit_flag(converted_dict, key):
                    ignorable.add(key)
    extras -= ignorable
    # M2/SINI may appear as None defaults on DD — ignore if absent from source
    for key in ("M2", "SINI"):
        if key in extras and key not in src_keys:
            val = _param_float(converted_dict, key, default=None)
            if val is None:
                extras.discard(key)

    if extras or missing:
        raise BinaryConversionError(
            "unexpected_converter_output: "
            f"extras={sorted(extras)} missing={sorted(missing)}"
        )

    reemitted = set()
    if any(_canon_key(k) == "PB" for k in patch.removed_keys):
        reemitted.add("PB")
    if any(_canon_key(k) == "A1" for k in patch.removed_keys):
        reemitted.add("A1")

    for canon in sorted(src_keys - removed_canon):
        if canon in reemitted or canon == "BINARY":
            continue
        if canon.startswith("FB"):
            continue

        aliases = tuple(get_aliases_for_parameter(canon))
        if not aliases:
            aliases = (canon,)

        # Prefer physical model values when the source TimingModel owns the axis.
        # source_model was loaded with create_pint_model (PINT unit_scale applied);
        # corrected_model is the post-§7.6 converted TimingModel. Canonical PINT
        # names (A1DOT, not XDOT) are the model attribute names.
        source_has_param = (
            hasattr(source_model, canon)
            and getattr(getattr(source_model, canon), "value", None) is not None
        )
        if source_has_param:
            src_phys = _model_value(source_model, canon)
            conv_phys = _model_value(corrected_model, canon)
            if canon in {"TASC", "T0", "PEPOCH", "POSEPOCH", "DMEPOCH"}:
                if not np.isclose(
                    np.longdouble(src_phys),
                    np.longdouble(conv_phys),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise BinaryConversionError(
                        f"converter_modified_passthrough_key: {canon} "
                        f"src_model={src_phys} conv_model={conv_phys}"
                    )
            else:
                if not np.isclose(
                    float(src_phys), float(conv_phys), rtol=1e-12, atol=0.0
                ):
                    if float(src_phys) == float(int(src_phys)) and float(
                        src_phys
                    ) != float(conv_phys):
                        raise BinaryConversionError(
                            f"converter_modified_passthrough_key: {canon} "
                            f"src_model={src_phys} conv_model={conv_phys}"
                        )
                    if abs(float(src_phys)) > 0 and (
                        abs(float(src_phys) - float(conv_phys)) / abs(float(src_phys))
                        > 1e-12
                    ):
                        raise BinaryConversionError(
                            f"converter_modified_passthrough_key: {canon} "
                            f"src_model={src_phys} conv_model={conv_phys}"
                        )
            continue

        # Fallback: alias-aware dict-token compare (string / non-model keys).
        src_val = _param_float(source_dict, *aliases, default=None)
        conv_val = _param_float(converted_dict, *aliases, default=None)
        if src_val is None and conv_val is None:
            continue
        if src_val is None or conv_val is None:
            src_s = _param_str(source_dict, *aliases)
            conv_s = _param_str(converted_dict, *aliases)
            if src_s != conv_s:
                raise BinaryConversionError(
                    f"converter_modified_passthrough_key: {canon} "
                    f"src={src_s!r} conv={conv_s!r}"
                )
            continue
        if canon in {"TASC", "T0", "PEPOCH", "POSEPOCH", "DMEPOCH"}:
            if not np.isclose(
                np.longdouble(src_val), np.longdouble(conv_val), rtol=0.0, atol=1e-12
            ):
                raise BinaryConversionError(
                    f"converter_modified_passthrough_key: {canon}"
                )
        else:
            if not np.isclose(float(src_val), float(conv_val), rtol=1e-12, atol=0.0):
                if float(src_val) == float(int(src_val)) and float(src_val) != float(
                    conv_val
                ):
                    raise BinaryConversionError(
                        f"converter_modified_passthrough_key: {canon}"
                    )
                if (
                    abs(float(src_val)) > 0
                    and abs(float(src_val) - float(conv_val)) / abs(float(src_val))
                    > 1e-12
                ):
                    raise BinaryConversionError(
                        f"converter_modified_passthrough_key: {canon} "
                        f"src={src_val} conv={conv_val}"
                    )

    # Re-emitted PB/A1 audited against the corrected model. The comparison must
    # be against the PATCH — the artifact that actually lands in the par files —
    # not against `converted_dict`, which is parsed straight back out of
    # `corrected_model.as_parfile()` and so agrees with the model by
    # construction. Auditing model-vs-model can never fail (§7.3/§C5).
    patch_values = {
        _canon_key(key): line.split()[0]
        for key, line in patch.added_lines
        if key.upper() != "C" and line.split()
    }
    for name in sorted(reemitted):
        corrected = _model_value(corrected_model, name)
        emitted = patch_values.get(name)
        if emitted is None:
            raise BinaryConversionError(
                f"correction_not_applied: {name} is marked re-emitted (§7.3) but "
                "the patch carries no line for it, so the §7.6 correction would "
                "be dropped from every written par"
            )
        if not np.isclose(float(emitted), float(corrected), rtol=1e-12, atol=0.0):
            raise BinaryConversionError(
                f"correction_not_applied: {name} patch={emitted} "
                f"corrected_model={corrected}"
            )


# ---------------------------------------------------------------------------
# Fidelity harness (§7.5 F1–F7)
# ---------------------------------------------------------------------------


def _shapiro_present(model: Any) -> bool:
    for name in ("M2", "SINI", "H3", "STIGMA"):
        if hasattr(model, name) and getattr(model, name).value is not None:
            if name in ("M2", "SINI", "H3") and _model_value(model, name) != 0.0:
                return True
            if name == "STIGMA":
                return True
    return False


def _stand_alone_delay_s(model: Any, toas: Any) -> Optional[np.ndarray]:
    """Return the stand-alone binary Shapiro delay, or None when there is none.

    Fail closed: a *present* Shapiro sector whose evaluator cannot be reached is
    an error, not a reason to silently drop the F5/F7 component assertion. Only
    a genuinely Shapiro-free model returns None.
    """
    binary_comp = None
    for name, comp in model.components.items():
        if name.startswith("Binary"):
            binary_comp = comp
            break
    if binary_comp is None:
        if _shapiro_present(model):
            raise BinaryConversionError(
                "fidelity_check_failed: Shapiro terms present but no binary "
                "component exposes a stand-alone evaluator"
            )
        return None
    try:
        binary_comp.update_binary_object(toas)
        bm = binary_comp.binary_instance
        delay = bm.delayS()
    except Exception as exc:
        if _shapiro_present(model):
            raise BinaryConversionError(
                f"fidelity_check_failed: stand-alone delayS unavailable for a "
                f"model carrying Shapiro terms: {exc}"
            ) from exc
        logger.debug(f"stand-alone delayS unavailable on a Shapiro-free model: {exc}")
        return None
    return np.asarray(
        delay.to_value("s") if hasattr(delay, "to_value") else delay, dtype=float
    )


def _orbital_phase(model: Any, mjds: np.ndarray) -> np.ndarray:
    tasc = _model_value(model, "TASC", default=_model_value(model, "T0"))
    pb = _model_value(model, "PB")
    return 2.0 * np.pi * (np.asarray(mjds, dtype=np.longdouble) - tasc) / pb


def _tail_pred_case_c(
    model: Any, mjds: np.ndarray, h3: float, stig: float, nharms: int
) -> np.ndarray:
    """Δ_S^(28) − truncated NHARMS series."""
    from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
    import astropy.units as u

    bm = ELL1Hmodel()
    bm.fit_params = ["H3", "STIGMA"]
    pb = _model_value(model, "PB")
    bm.update_input(
        barycentric_toa=np.asarray(mjds, dtype=np.longdouble),
        PB=pb * u.day,
        A1=_model_value(model, "A1") * u.lsec,
        TASC=np.longdouble(_model_value(model, "TASC")) * u.day,
        EPS1=_model_value(model, "EPS1") * u.Unit(""),
        EPS2=_model_value(model, "EPS2") * u.Unit(""),
        H3=h3 * u.s,
        STIGMA=stig * u.Unit(""),
    )
    s28 = bm.delayS3p_H3_STIGMA_exact(h3 * u.s, stig).to_value(u.s)
    s_tr = bm.delayS3p_H3_STIGMA_approximate(
        h3 * u.s, stig, end_harm=int(nharms)
    ).to_value(u.s)
    return np.asarray(s28 - s_tr, dtype=float)


def _tail_pred_case_d(
    model: Any, mjds: np.ndarray, h3: float, stig: float
) -> np.ndarray:
    """Δ_S^(28) − H3-only (−(4/3)H3 sin3Φ)."""
    from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
    import astropy.units as u

    bm = ELL1Hmodel()
    bm.fit_params = ["H3", "STIGMA"]
    pb = _model_value(model, "PB")
    bm.update_input(
        barycentric_toa=np.asarray(mjds, dtype=np.longdouble),
        PB=pb * u.day,
        A1=_model_value(model, "A1") * u.lsec,
        TASC=np.longdouble(_model_value(model, "TASC")) * u.day,
        EPS1=_model_value(model, "EPS1") * u.Unit(""),
        EPS2=_model_value(model, "EPS2") * u.Unit(""),
        H3=h3 * u.s,
        STIGMA=stig * u.Unit(""),
    )
    s28 = bm.delayS3p_H3_STIGMA_exact(h3 * u.s, stig).to_value(u.s)
    # H3-only evaluator: stigma=0 → −(4/3) H3 sin3Φ
    bm.fit_params = ["H3"]
    bm.update_input(
        barycentric_toa=np.asarray(mjds, dtype=np.longdouble),
        PB=pb * u.day,
        A1=_model_value(model, "A1") * u.lsec,
        TASC=np.longdouble(_model_value(model, "TASC")) * u.day,
        EPS1=_model_value(model, "EPS1") * u.Unit(""),
        EPS2=_model_value(model, "EPS2") * u.Unit(""),
        H3=h3 * u.s,
        STIGMA=0.0 * u.Unit(""),
    )
    s_h3 = bm.delayS3p_H3_STIGMA_exact(h3 * u.s, 0.0).to_value(u.s)
    # Prefer the H3-only closed form if available
    try:
        phi = bm.Phi()
        phi = phi.to_value(u.rad) if hasattr(phi, "to_value") else np.asarray(phi)
        s_h3 = -(4.0 / 3.0) * h3 * np.sin(3.0 * phi)
    except Exception:
        pass
    return np.asarray(s28 - s_h3, dtype=float)


def _fidelity_tolerances(
    *,
    plain: bool,
    a1_max: float,
    e_max: float,
    nb: float,
    h3: float,
    stig: float,
    peak_delay_s: float,
    floor: float,
) -> tuple[float, float, Optional[float]]:
    if plain:
        tol_roemer = 3.0 * (0.015 * nb * a1_max**2 * e_max + a1_max * e_max**4) + floor
        if peak_delay_s > 0.0:
            tol_shapiro = 3.0 * e_max * peak_delay_s + floor
        else:
            tol_shapiro = None
    else:
        r = h3 / stig**3 if stig else 0.0
        tol_roemer = (
            3.0 * (3.0 * nb * a1_max * h3 / stig + 4.0 * (h3 / stig) * e_max) + floor
            if stig
            else floor
        )
        tol_shapiro = 3.0 * 2.0 * r * e_max * (4.0 + 20.0 * stig**4) + floor
    tol_total = tol_roemer + (tol_shapiro or 0.0)
    return (
        float(tol_roemer),
        float(tol_total),
        (float(tol_shapiro) if tol_shapiro is not None else None),
    )


def _absorbed_shapiro_to_full(
    d_s_absorbed: np.ndarray, mjds: np.ndarray, model: Any, h3: float, stig: float
) -> np.ndarray:
    """Lift Eq.28 absorbed Shapiro onto the Eq.29 full footing.

    Identity: ``Δ_S^(28) = Δ_S^(29) − 4 r ς sinΦ + 2 r ς² cos2Φ``, so
    ``Δ_S^(29) = Δ_S^(28) + 4 r ς sinΦ − 2 r ς² cos2Φ``. Comparing DDH's full
    Shapiro against the absorbed evaluator without this lift would re-introduce
    the 3.6 μs gauge term into the F5 component split even when the total delay
    identity holds.
    """
    if stig <= 0.0:
        return d_s_absorbed
    r = h3 / stig**3
    tasc = _model_value(model, "TASC", default=_model_value(model, "T0"))
    pb = _model_value(model, "PB")
    phi = 2.0 * np.pi * (np.asarray(mjds, dtype=np.longdouble) - tasc) / pb
    return np.asarray(
        d_s_absorbed
        + 4.0 * r * stig * np.sin(phi)
        - 2.0 * r * stig**2 * np.cos(2.0 * phi),
        dtype=float,
    )


def run_fidelity_check(
    source_model: Any,
    converted_model: Any,
    *,
    policy: "AlignmentPolicy",
    plain: bool,
    case: Optional[str] = None,
    nharms: int = 7,
    stigma_central: Optional[float] = None,
    grid_points: int = 1024,
) -> BinaryFidelityReport:
    """§7.5 delay-fidelity invariant (mean-removed; Case C/D F4b tail)."""
    from pint.simulation import make_fake_toas_fromMJDs
    import astropy.units as u

    tasc = _model_value(source_model, "TASC", default=_model_value(source_model, "T0"))
    pb = _model_value(source_model, "PB")
    anchors = [tasc]
    has_dots = any(
        _model_value(source_model, n) != 0.0
        for n in ("EPS1DOT", "EPS2DOT", "A1DOT", "PBDOT")
    )
    start = _model_value(source_model, "START", default=float("nan"))
    finish = _model_value(source_model, "FINISH", default=float("nan"))
    if has_dots and math.isfinite(start) and math.isfinite(finish):
        anchors.extend([start, finish])

    a1_max = abs(_model_value(source_model, "A1"))
    e_max = math.hypot(
        _model_value(source_model, "EPS1"), _model_value(source_model, "EPS2")
    )
    if e_max == 0.0 and hasattr(converted_model, "ECC"):
        e_max = abs(_model_value(converted_model, "ECC"))
    nb = 2.0 * math.pi / (pb * _SECDAY) if pb else 0.0
    h3 = _model_value(source_model, "H3")
    if case == "C":
        stig = _model_value(source_model, "H4") / h3 if h3 else 0.0
    elif case == "D":
        stig = float(stigma_central or 0.0)
    elif case in ("A", "B"):
        stig = _model_value(source_model, "STIGMA")
    else:
        stig = _model_value(converted_model, "STIGMA") if not plain else 0.0

    floor = float(policy.binary_fidelity_floor_s)
    d_total_parts: list[np.ndarray] = []
    d_shap_parts: list[np.ndarray] = []
    src_shapiro_peaks: list[float] = []
    absorbed_cases = case in ("B", "C", "D")

    for t_a in anchors:
        mjds = np.asarray(
            t_a + (np.arange(grid_points) / float(grid_points)) * pb,
            dtype=float,
        )
        toas = make_fake_toas_fromMJDs(
            MJDs=mjds * u.d,
            model=source_model,
            obs="@",
            freq=np.inf * u.MHz,
        )
        d_src = np.asarray(source_model.delay(toas).to_value(u.s), dtype=float)
        d_conv = np.asarray(converted_model.delay(toas).to_value(u.s), dtype=float)
        d_total = d_src - d_conv

        tail = np.zeros_like(d_total)
        if case == "C":
            tail = _tail_pred_case_c(source_model, mjds, h3, stig, nharms)
            d_total = d_total + tail
        elif case == "D":
            tail = _tail_pred_case_d(source_model, mjds, h3, stig)
            d_total = d_total + tail

        d_total_parts.append(d_total)

        d_s_src = _stand_alone_delay_s(source_model, toas)
        d_s_conv = _stand_alone_delay_s(converted_model, toas)
        if d_s_src is not None and d_s_conv is not None:
            # F6 uses peak|delayS_source| — the SOURCE Shapiro amplitude, never
            # the source-minus-converted difference (that would make the
            # tolerance self-referential and ~5x too tight).
            src_shapiro_peaks.append(float(np.max(np.abs(d_s_src))))
            if absorbed_cases and stig:
                d_s_src = _absorbed_shapiro_to_full(
                    d_s_src, mjds, source_model, h3, stig
                )
            d_shap = d_s_src - d_s_conv
            if case in ("C", "D"):
                d_shap = d_shap + tail
            d_shap_parts.append(d_shap)

    d_total_all = np.concatenate(d_total_parts)
    # F4c centering
    c_measured = float(np.mean(d_total_all))
    # The ELL1-series constant is -(3/2)*A1*eps1 in the *intrinsic* pair, i.e.
    # the converted model's own (A1, ECC·sin OM) — not the printed absorbed
    # pair. They differ by the 4*H3/stig transfer, which is negligible at
    # stig ~ 0.5 but 20x larger at stig ~ 0.05, where the printed-pair form is
    # 22% wrong and would trip the 10% assertion on a correct conversion.
    a1_i = abs(_model_value(converted_model, "A1"))
    ecc_i = _model_value(converted_model, "ECC")
    om_i_rad = math.radians(_model_value(converted_model, "OM"))
    eps1_i = ecc_i * math.sin(om_i_rad)
    c_pred = 1.5 * a1_i * eps1_i
    if not plain and stig:
        r = h3 / stig**3
        c_pred -= 2.0 * r * math.log1p(stig**2)
    c_pred = float(c_pred)
    # F4c sanity: the two families differ by a predicted, unobservable constant.
    # A measured mean far from the prediction means the map itself is wrong, so
    # this is an assertion, not a log line.
    if abs(c_pred) > 0 and abs(c_measured - c_pred) > 0.10 * abs(c_pred) + 1e-10:
        raise BinaryConversionError(
            "fidelity_check_failed: F4c mean constant off prediction — "
            f"measured={c_measured:.6e} s predicted={c_pred:.6e} s "
            "(tolerance 10%); the source/target parameter map is inconsistent"
        )
    d_total_all = d_total_all - c_measured

    d_shap_all = None
    if d_shap_parts:
        d_shap_all = np.concatenate(d_shap_parts)

    if d_shap_all is not None and len(d_shap_all) == len(d_total_all):
        d_shap_all = d_shap_all - np.mean(d_shap_all)
        d_roemer = d_total_all - d_shap_all
        d_roemer = d_roemer - np.mean(d_roemer)
    else:
        d_roemer = d_total_all
        d_shap_all = None

    peak_delay_s = 0.0
    if plain and _shapiro_present(source_model):
        peak_delay_s = max(src_shapiro_peaks) if src_shapiro_peaks else 0.0

    tol_roemer, tol_total, tol_shapiro = _fidelity_tolerances(
        plain=plain,
        a1_max=a1_max,
        e_max=e_max,
        nb=nb,
        h3=h3,
        stig=stig if stig else 1.0,
        peak_delay_s=peak_delay_s,
        floor=floor,
    )
    # User-facing scale on the published §7.5 budget (AlignmentPolicy).
    # Applied after derivation so factor=1 is bit-identical to the prior default.
    factor = float(policy.binary_fidelity_tolerance_factor)
    tol_roemer *= factor
    tol_total *= factor
    if tol_shapiro is not None:
        tol_shapiro *= factor

    total_max = float(np.max(np.abs(d_total_all)))
    roemer_max = float(np.max(np.abs(d_roemer)))
    shapiro_max = float(np.max(np.abs(d_shap_all))) if d_shap_all is not None else None

    if (
        total_max > tol_total
        or roemer_max > tol_roemer
        or (
            shapiro_max is not None
            and tol_shapiro is not None
            and shapiro_max > tol_shapiro
        )
    ):
        raise BinaryConversionError(
            "fidelity_check_failed: "
            f"total_max={total_max:.6e} tol_total={tol_total:.6e} "
            f"roemer_max={roemer_max:.6e} tol_roemer={tol_roemer:.6e} "
            f"shapiro_max={shapiro_max} tol_shapiro={tol_shapiro}"
        )

    return BinaryFidelityReport(
        grid_points_per_orbit=grid_points,
        anchor_epochs_mjd=tuple(float(a) for a in anchors),
        total_max_abs_s=total_max,
        shapiro_max_abs_s=shapiro_max,
        roemer_max_abs_s=roemer_max,
        tolerance_total_s=tol_total,
        tolerance_shapiro_s=tol_shapiro,
        tolerance_roemer_s=tol_roemer,
    )


# ---------------------------------------------------------------------------
# Patch application (§8.4)
# ---------------------------------------------------------------------------


def engine_native_binary_key(key: str, timing_package: Optional[str]) -> str:
    """Map a canonical patch key onto the target engine's native spelling.

    Tempo2 reads the DDH orthometric ratio as ``STIG`` only — ``readParfile.C``
    has no ``STIGMA``/``VARSIGMA`` branch, and ``DDHmodel.C`` *exits the
    process* when ``stig`` is unset. PINT accepts ``STIG`` as an alias of
    ``STIGMA``, so the tempo2-native spelling is the one both engines read.
    Analogous to portable ``CLK`` for the clock pin: prefer the spelling every
    target engine actually reads.
    """
    if _canon_key(key) == "STIGMA" and normalize_timing_package(timing_package) == (
        "tempo2"
    ):
        return "STIG"
    return key


def apply_binary_patch(
    parfile_dict: dict,
    patch: BinaryPatch,
    *,
    timing_package: Optional[str] = None,
) -> None:
    """Apply a binary-owned patch in place (§8.4).

    ``timing_package`` selects the engine-native spelling for keys that differ
    between PINT and tempo2 (see :func:`engine_native_binary_key`); the
    alias-resolved postcondition 2 identity across PTAs is unaffected.
    """
    for key in patch.removed_keys:
        # Match case-insensitively
        actual = _find_key(parfile_dict, key)
        if actual is not None:
            del parfile_dict[actual]
        elif key in parfile_dict:
            del parfile_dict[key]

    bin_key = _find_key(parfile_dict, "BINARY")
    if bin_key is None:
        raise BinaryConversionError(
            "unexpected_converter_output: BINARY missing at patch application"
        )
    # Preserve list-of-strings shape; value only
    parfile_dict[bin_key] = [patch.binary_value]

    for key, line in patch.added_lines:
        if key.upper() == "C":
            # Comment lines: accumulate under 'C' if present
            if "C" in parfile_dict:
                parfile_dict["C"] = list(parfile_dict["C"]) + [line]
            else:
                parfile_dict["C"] = [line]
            continue
        parfile_dict[engine_native_binary_key(key, timing_package)] = [line]


def assert_postconditions(
    parfile_dicts: Mapping[str, dict],
    *,
    target_family: str,
    pre_nonbinary: Mapping[str, dict],
) -> None:
    """§8.4 postconditions; raise BinaryConversionError on violation."""
    forbidden = {
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "TASC",
        "NHARM",
        "NHARMS",
        "H4",
        "VARSIGMA",
    }
    for pta, par in parfile_dicts.items():
        for key in list(par):
            if key.upper() in forbidden:
                raise BinaryConversionError(
                    f"postcondition: {pta} still has forbidden key {key}"
                )
        binary = _binary_value(par)
        if binary != target_family:
            raise BinaryConversionError(
                f"postcondition: {pta} BINARY={binary} != {target_family}"
            )
        has_h3 = _find_key(par, "H3") is not None
        stig_spellings = _present_spellings(par, "STIGMA")
        if target_family == "DDH":
            if not (has_h3 and stig_spellings):
                raise BinaryConversionError(
                    f"postcondition: {pta} DDH missing H3/STIGMA"
                )
            if len(stig_spellings) != 1:
                raise BinaryConversionError(
                    f"postcondition: {pta} carries {len(stig_spellings)} STIGMA "
                    f"spellings {stig_spellings}; exactly one is required"
                )
        else:
            if has_h3 or stig_spellings:
                raise BinaryConversionError(
                    f"postcondition: {pta} DD must not carry H3/STIGMA"
                )

    # Binary-owned identity across PTAs
    blocks = {pta: _binary_owned_snapshot(par) for pta, par in parfile_dicts.items()}
    ref_pta = next(iter(blocks))
    for pta, block in blocks.items():
        if block != blocks[ref_pta]:
            raise BinaryConversionError(
                f"postcondition: binary-owned blocks diverge ({ref_pta} vs {pta})"
            )

    # Non-binary keys preserved exactly
    for pta, par in parfile_dicts.items():
        pre = pre_nonbinary[pta]
        for key, value in pre.items():
            if _canon_key(key) in BINARY_OWNED_CANONICAL or key.upper().startswith(
                "FB"
            ):
                continue
            if key.upper() == "BINARY":
                continue
            if key not in par or par[key] != value:
                raise BinaryConversionError(
                    f"postcondition: non-binary key {key} changed on {pta}"
                )


def _nonbinary_snapshot(par: Mapping[str, Any]) -> dict:
    """Snapshot of non-binary-owned key→value mappings."""
    out = {}
    for key, value in par.items():
        if key.upper() == "BINARY":
            continue
        if _canon_key(key) in BINARY_OWNED_CANONICAL or key.upper().startswith("FB"):
            continue
        out[key] = copy.deepcopy(value)
    return out


# ---------------------------------------------------------------------------
# Contract 2 — convert_shared_binary (§7)
# ---------------------------------------------------------------------------


def convert_shared_binary(
    reference_dict: dict,
    decision: BinaryConversionDecision,
    *,
    pta_names: tuple[str, ...],
    policy: "AlignmentPolicy",
    ell1h_shapiro: Ell1hShapiroMode,
) -> tuple[BinaryPatch, BinaryConversionRecord]:
    """Convert a shared ELL1-family binary to DD/DDH; return patch + record.

    ``ell1h_shapiro`` is the *stack's own* resolved evaluator mode
    (``ParameterManager.ell1h_shapiro``) and is required, never defaulted: it
    selects the gauge of the §7.6 map, and because the source model is loaded
    in the same mode a wrong value is self-consistent and would slip past the
    fidelity check (§18, "gauge source" lock).

    Never mutates ``reference_dict``. Fail closed on any internal failure.
    """
    from pint.models.model_builder import parse_parfile
    from pint.binaryconvert import convert_binary

    if decision.outcome != "convert":
        raise BinaryConversionError(
            f"convert_shared_binary requires outcome='convert', got {decision.outcome}"
        )

    plain = decision.target_family == "DD"
    gauge: Optional[str] = None if plain else ell1h_shapiro
    case: Optional[str] = None
    required_sampling: tuple[str, ...] = ()
    stigma_provenance: Optional[str] = None

    # C1
    try:
        source_model = create_pint_model(
            dict(reference_dict), ell1h_shapiro=ell1h_shapiro
        )
    except Exception as exc:
        raise BinaryConversionError(
            f"model_build_failed: could not load source binary: {exc}"
        ) from exc

    triple_free = _fit_flag(reference_dict, "EPS1")
    had_dots = _has_any(reference_dict, ("EPS1DOT", "EPS2DOT"))
    dots_free = _fit_flag(reference_dict, "EPS1DOT") if had_dots else False
    ddh_uncertainties: dict[str, float] = {}

    # C2 / C2b
    try:
        if plain:
            # convert_binary needs usable EPS uncertainties; only the plain path
            # feeds it, and the orthometric map reads its sigmas off the source
            # par instead, so the placeholders stay scoped here.
            _ensure_uncertainties_for_convert(source_model)
            converted = convert_binary(source_model, "DD")
            apply_conversion_corrections(converted, source_model)
            case = None
        else:
            # Determine case for record / fidelity
            if _has_any(reference_dict, ("STIGMA", "STIG", "VARSIGMA")):
                case = "B" if ell1h_shapiro == "absorbed" else "A"
            elif _find_key(reference_dict, "H4") is not None:
                case = "C"
            else:
                case = "D"
                required_sampling = ("STIGMA",)
                stigma_provenance = policy.stigma_provenance
            converted, ddh_uncertainties = _convert_ell1h_block(
                source_model,
                ell1h_shapiro,
                policy,
                source_uncertainties=_source_uncertainties(reference_dict),
            )
    except BinaryConversionError:
        raise
    except Exception as exc:
        raise BinaryConversionError(
            f"conversion_failed: {exc}\n{remediation_message()}"
        ) from exc

    # C3
    converted_dict = parse_parfile(StringIO(converted.as_parfile()))

    # C4
    if plain:
        patch = _build_patch_plain(
            reference_dict,
            converted,
            triple_free=triple_free,
            dots_free=dots_free,
            had_dots=had_dots,
        )
    else:
        patch = _build_patch_orthometric(
            reference_dict,
            converted,
            triple_free=triple_free,
            dots_free=dots_free,
            had_dots=had_dots,
            case=case or "B",
            policy=policy,
            uncertainties=ddh_uncertainties,
        )

    # C5
    _audit_converter_output(
        reference_dict, converted_dict, patch, converted, source_model
    )

    # C6 fidelity
    nharms = int(_param_float(reference_dict, "NHARMS", "NHARM", default=7.0) or 7.0)
    try:
        fidelity = run_fidelity_check(
            source_model,
            converted,
            policy=policy,
            plain=plain,
            case=case,
            nharms=nharms,
            stigma_central=policy.stigma_central,
        )
    except BinaryConversionError:
        raise
    except Exception as exc:
        raise BinaryConversionError(
            f"fidelity_check_failed: harness error: {exc}"
        ) from exc

    # C6b serialization fidelity
    serialized = copy.deepcopy(dict(reference_dict))
    apply_binary_patch(serialized, patch)
    try:
        reloaded = create_pint_model(serialized, ell1h_shapiro=ell1h_shapiro)
        run_fidelity_check(
            source_model,
            reloaded,
            policy=policy,
            plain=plain,
            case=case,
            nharms=nharms,
            stigma_central=policy.stigma_central,
        )
    except BinaryConversionError as exc:
        raise BinaryConversionError(
            str(exc).replace(
                "fidelity_check_failed", "fidelity_check_failed_serialized"
            )
        ) from exc
    except Exception as exc:
        raise BinaryConversionError(f"fidelity_check_failed_serialized: {exc}") from exc

    source_free = _source_free_params(reference_dict)
    target_free: list[str] = []
    if triple_free:
        target_free.extend(["ECC", "OM", "T0"])
    if had_dots and dots_free:
        target_free.extend(["EDOT", "OMDOT"])
    for name in ("A1", "PB", "PBDOT", "A1DOT", "H3", "STIGMA", "M2", "SINI", "GAMMA"):
        if name == "A1DOT":
            if _fit_flag(reference_dict, "A1DOT", "XDOT"):
                target_free.append("A1DOT")
        elif name == "STIGMA":
            if decision.target_family == "DDH" and (
                _fit_flag(reference_dict, "STIGMA", "STIG", "VARSIGMA", "H4")
                or case == "D"
            ):
                target_free.append("STIGMA")
        elif name == "H3":
            if _fit_flag(reference_dict, "H3"):
                target_free.append("H3")
        elif _fit_flag(reference_dict, name):
            target_free.append(name)

    record = BinaryConversionRecord(
        pta_names=pta_names,
        source_free_params=source_free,
        target_free_params=tuple(target_free),
        fidelity=fidelity,
        patch=patch,
        gauge=gauge,
        required_sampling=required_sampling,
        stigma_provenance=stigma_provenance,
        uncertainty_propagation=(
            "pint_convert_binary" if plain else "diagonal_jacobian"
        ),
    )
    return patch, record


def metadata_from_report(
    report: Optional[BinaryConversionReport],
) -> Optional[BinaryConversionMetadata]:
    """Derive :class:`BinaryConversionMetadata` from a conversion report."""
    if report is None or report.record is None:
        return None
    rec = report.record
    return BinaryConversionMetadata(
        target_family=rec.patch.binary_value,
        gauge=rec.gauge,
        required_sampling=rec.required_sampling,
        stigma_central=(
            None
            if not rec.required_sampling
            else (
                # Recover from patch STIGMA line when Case D
                float(
                    next(
                        (
                            line.split()[0]
                            for key, line in rec.patch.added_lines
                            if key.upper() == "STIGMA"
                        ),
                        "nan",
                    )
                )
            )
        ),
        stigma_provenance=rec.stigma_provenance,
    )
