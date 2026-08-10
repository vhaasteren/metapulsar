"""Tests for the par-value unit boundary (`feature_par_units.md` §8).

Covers the `metapulsar.pint_compat` accessors re-exported through
``metapulsar.pint_helpers``: par-vs-model equality, Tempo-convention
spellings, the emission portability rule against a model of tempo2's
parser, epoch accessors, alias closure, long-double preservation, prototype
hygiene, and the §7.3 literal ban on hand-written unit conversions.
"""

from __future__ import annotations

import inspect
import math
import time
from io import StringIO

import numpy as np
import pytest

from metapulsar.pint_helpers import (
    CANONICAL_SI,
    ParUnitError,
    has_canonical_unit,
    mjd_from_model,
    mjd_from_par,
    pint_parameter_name,
    si_from_model,
    si_from_par,
    si_quantity_from_token,
    token_from_si,
)

# ---------------------------------------------------------------------------
# Par templates (§8 test 1)
# ---------------------------------------------------------------------------

_BASE = """PSR J0000+0000
RAJ 00:00:00
DECJ 00:00:00
F0 100.0
PEPOCH 55000
DM 10.0
"""

_DD_BASE = _BASE + ("BINARY DD\nPB 6.5\nA1 10.0\nECC 1e-5\nOM 45.0\nT0 55000.5\n")

_ELL1_BASE = _BASE + (
    "BINARY ELL1\nPB 6.5\nA1 10.0\nTASC 55000.5\nEPS1 7e-6\nEPS2 -1.8e-5\n"
)

_ELL1H_BASE = _BASE + (
    "BINARY ELL1H\nPB 6.5\nA1 10.0\nTASC 55000.5\nEPS1 7e-6\nEPS2 -1.8e-5\n"
    "H3 1.8e-7\nSTIGMA 0.5\n"
)

#: Per-key par text carrying the key with a Tempo-convention-exercising token.
_PAR_TEXT: dict[str, str] = {
    "A1": _DD_BASE,
    "PB": _DD_BASE,
    "ECC": _DD_BASE,
    "OM": _DD_BASE,
    "EDOT": _DD_BASE + "EDOT 0.0031\n",  # scaled spelling -> 3.1e-15 /s
    "OMDOT": _DD_BASE + "OMDOT 0.05\n",  # deg/yr
    "PBDOT": _DD_BASE + "PBDOT 1.2\n",  # scaled spelling -> 1.2e-12
    "A1DOT": _DD_BASE + "A1DOT 0.005485\n",  # scaled spelling -> 5.485e-15
    "M2": _DD_BASE + "M2 0.25\nSINI 0.9\n",
    "SINI": _DD_BASE + "M2 0.25\nSINI 0.9\n",
    "EPS1": _ELL1_BASE,
    "EPS2": _ELL1_BASE,
    "EPS1DOT": _ELL1_BASE + "EPS1DOT 0.005038\n",  # 1e-12/s -> 5.038e-15 /s
    "EPS2DOT": _ELL1_BASE + "EPS2DOT -0.000397\n",
    "H3": _ELL1H_BASE,
    "STIGMA": _ELL1H_BASE,
    "H4": _ELL1_BASE.replace("BINARY ELL1", "BINARY ELL1H") + "H3 1.8e-7\nH4 9e-8\n",
    "NE_SW": _BASE + "NE_SW 5.0\n",
}


def _parse(text: str):
    from pint.models.model_builder import parse_parfile

    return parse_parfile(StringIO(text))


def _model(text: str):
    from pint.models import get_model

    return get_model(StringIO(text))


# ---------------------------------------------------------------------------
# 1. Par path equals model path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CANONICAL_SI))
def test_par_path_equals_model_path(name):
    text = _PAR_TEXT[name]
    par_si = si_from_par(_parse(text), name)
    model_si = si_from_model(_model(text), name)
    assert par_si is not None
    assert par_si == pytest.approx(model_si, rel=1e-12)


@pytest.mark.parametrize("name", ["TASC", "T0", "PEPOCH", "START", "FINISH"])
def test_epoch_accessors_and_si_refusal(name):
    text = (_ELL1_BASE if name == "TASC" else _DD_BASE) + (
        "START 54000.5\nFINISH 56000.5\n"
    )
    par = _parse(text)
    model = _model(text)
    par_mjd = mjd_from_par(par, name)
    model_mjd = mjd_from_model(model, name)
    assert par_mjd is not None and model_mjd is not None
    assert float(par_mjd) == pytest.approx(float(model_mjd), abs=1e-9)
    with pytest.raises(ParUnitError):
        si_from_par(par, name)
    with pytest.raises(ParUnitError):
        si_from_model(model, name)


# ---------------------------------------------------------------------------
# 2. Both spellings, one physics (row B)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,scaled,plain",
    [
        ("XDOT", "0.005485", "5.485e-15"),
        ("A1DOT", "-0.009436", "-9.436e-15"),
        ("PBDOT", "7.2", "7.2e-12"),
        ("EDOT", "0.0031", "3.1e-15"),
    ],
)
def test_row_b_both_spellings(name, scaled, plain):
    a = si_quantity_from_token(name, scaled)
    b = si_quantity_from_token(name, plain)
    assert float(a.value) == pytest.approx(float(b.value), rel=1e-12)
    assert float(b.value) == pytest.approx(float(plain), rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Row A is unconditional
# ---------------------------------------------------------------------------


def test_row_a_unconditional_and_emission_refusal():
    # PINT applies the 1e-12/s unit to EPS dots unconditionally; the boundary
    # reproduces PINT exactly rather than second-guessing it.
    q = si_quantity_from_token("EPS1DOT", "5.038e-15")
    assert float(q.value) == pytest.approx(5.038e-27, rel=1e-12)
    q = si_quantity_from_token("EPS1DOT", "0.005038")
    assert float(q.value) == pytest.approx(5.038e-15, rel=1e-12)
    with pytest.raises(ParUnitError, match="readParfile"):
        token_from_si("EPS1DOT", 5.038e-27)


# ---------------------------------------------------------------------------
# 4./5. Emission inverse + portability against a model of tempo2's parser
# ---------------------------------------------------------------------------

#: Parameters tempo2 rescales with the `|token| > 1e-7 -> x1e-12` heuristic
#: (readParfile.C: EDOT 1973, PBDOT 2082, A1DOT/XDOT 2106, EPS dots 2134).
_TEMPO2_HEURISTIC = {"A1DOT", "PBDOT", "EDOT", "EPS1DOT", "EPS2DOT"}


def _tempo2_reads_si(name: str, token: str) -> float:
    """readParfile.C as a function: token -> tempo2's value on the SI axis."""
    canonical = pint_parameter_name(name)
    v = float(token)
    if canonical in _TEMPO2_HEURISTIC and abs(v) > 1e-7:
        v *= 1e-12
    if canonical == "OMDOT":  # tempo2 keeps deg/yr (readParfile.C:2100)
        return math.radians(v) / (365.25 * 86400.0)
    if canonical == "OM":  # deg
        return math.radians(v)
    if canonical == "PB":  # days
        return v * 86400.0
    return v


def _grid(name: str):
    """Log-spaced |values| in canonical units spanning every threshold."""
    magnitudes = [1e-27, 1e-21, 3.0e-19, 1e-16, 1e-12, 1e-9, 2e-7, 1e-3, 1.0, 2e5]
    return [0.0] + [s * m for m in magnitudes for s in (1.0, -1.0)]


@pytest.mark.parametrize("name", sorted(CANONICAL_SI))
def test_emission_inverse_and_portability(name):
    for value in _grid(name):
        try:
            token = token_from_si(name, value)
        except ParUnitError:
            # Only the EPS dots have a non-portable window, and only inside
            # (0, 1e-19] /s.
            assert pint_parameter_name(name) in {"EPS1DOT", "EPS2DOT"}
            assert 0.0 < abs(value) <= 1e-19
            continue
        pint_reads = float(si_quantity_from_token(name, token).value)
        assert pint_reads == pytest.approx(value, rel=1e-12, abs=0.0), token
        tempo2_reads = _tempo2_reads_si(name, token)
        assert tempo2_reads == pytest.approx(value, rel=1e-9, abs=0.0), token


def test_row_b_above_threshold_emits_scaled_spelling():
    # Emitted verbatim, `2e-07` would be re-read as 2e-19 by BOTH engines.
    token = token_from_si("A1DOT", 2e-7)
    assert float(token) == pytest.approx(2e5, rel=1e-12)
    assert float(si_quantity_from_token("A1DOT", token).value) == pytest.approx(
        2e-7, rel=1e-12
    )


# ---------------------------------------------------------------------------
# 6. Alias closure
# ---------------------------------------------------------------------------


def test_alias_closure():
    assert si_from_par({"XDOT": ["0.005485 1"]}, "A1DOT") == pytest.approx(
        5.485e-15, rel=1e-12
    )
    assert si_from_par({"E": ["1e-5 1"]}, "ECC") == pytest.approx(1e-5, rel=1e-12)
    assert si_from_par({"STIG": ["0.5 1"]}, "STIGMA") == pytest.approx(0.5)
    assert si_from_par({"VARSIGMA": ["0.5 1"]}, "STIGMA") == pytest.approx(0.5)
    assert si_from_par({"SOLARN0": ["5.0 1"]}, "NE_SW") == pytest.approx(5.0)
    assert has_canonical_unit("XDOT") and has_canonical_unit("E")
    assert not has_canonical_unit("DM") and not has_canonical_unit("TASC")


# ---------------------------------------------------------------------------
# 7. Long-double preservation (string-not-float rule, §5.3.1)
# ---------------------------------------------------------------------------


def test_long_double_preservation():
    token = "2.4593314651519201"
    q = si_quantity_from_token("PB", token)
    days = np.longdouble(q.value) / np.longdouble(86400.0)
    exact = np.longdouble(token)
    via_float64 = np.longdouble(float(token))
    assert abs(float(days - exact)) < abs(float(days - via_float64)) or (days == exact)
    assert abs(float((days - exact) / exact)) < 1e-17


# ---------------------------------------------------------------------------
# 8. Prototypes are cheap and inert
# ---------------------------------------------------------------------------


def test_prototypes_cheap_and_inert():
    from metapulsar.pint_compat import _prototype

    p1 = _prototype("EPS1DOT")
    p2 = _prototype("EPS1DOT")
    assert p1 is not p2
    p1.value = "1.0"
    assert p2.value is None  # mutation does not leak between prototypes
    assert getattr(p1, "_parent", None) is None  # no component graph attached

    _prototype("A1DOT")  # warm caches before timing
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        _prototype("A1DOT")
    per_call = (time.perf_counter() - t0) / n
    # Loose guard so a refactor cannot silently reintroduce the ~1.2 ms
    # deepcopy of the whole component graph.
    assert per_call < 100e-6, f"prototype construction {per_call*1e6:.1f} us/call"


def test_canonical_si_entries_are_convertible():
    """Import-time invariant: a PINT unit change fails here, not in a run."""
    from metapulsar.pint_compat import _descriptor

    for name, canonical_unit in CANONICAL_SI.items():
        d = _descriptor(name)
        (1.0 * d.units).to(canonical_unit)  # raises on incompatibility


# ---------------------------------------------------------------------------
# §7.3 literal ban
# ---------------------------------------------------------------------------


def test_no_hand_written_unit_conversions_in_binary_family_convert():
    """No 1e-12 / 1e12 / 180-over-pi / days-per-year applied to par values.

    The allowlist is an explicit line-content match: comparison tolerances
    and placeholder sigmas are legitimate uses of the literal, and any new
    occurrence has to be added here deliberately.
    """
    import metapulsar.binary_family_convert as bfc

    source_path = inspect.getsourcefile(bfc)
    assert source_path is not None
    banned = (
        "1e-12",
        "1e12",
        "180.0 / math.pi",
        "math.pi / 180",
        "_DAY_PER_YEAR",
        "365.25",
    )
    allowed = (
        "rtol=1e-12",
        "atol=1e-12",
        "> 1e-12",  # audit token-branch relative tolerance
        '1 1e-12"',  # ECC/OM/T0 placeholder sigma line
        "* 1e-12 * unit",  # _ensure_uncertainties_for_convert placeholder
    )
    offenders = []
    with open(source_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            code = line.split("#", 1)[0]
            if any(token in code for token in banned) and not any(
                token in code for token in allowed
            ):
                offenders.append(f"{lineno}: {line.rstrip()}")
    assert not offenders, "hand-written unit conversion(s):\n" + "\n".join(offenders)
