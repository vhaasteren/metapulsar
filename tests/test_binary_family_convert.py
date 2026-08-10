"""Unit tests for gated ELL1-family → DD/DDH conversion (Contracts 1–2).

Table-driven acceptance tests T1–T21 and rev-2 T30–T45 from
``feature_ell1h_truncation_fixw_nltiming.md`` §12.1. Synthetic pars only.
"""

from __future__ import annotations

import copy
import logging
import math
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from metapulsar.binary_family_convert import (
    BinaryConversionError,
    BinaryConversionReport,
    _absorbed_shapiro_to_full,
    _absorbed_to_intrinsic,
    _convert_ell1h_block,
    _fit_flag,
    _intrinsic_dots,
    _mixed_orthometric_sextet_detail,
    _model_value,
    _nonbinary_snapshot,
    _shapiro_present,
    _stand_alone_delay_s,
    apply_binary_patch,
    apply_conversion_corrections,
    assert_postconditions,
    convert_shared_binary,
    decide_binary_conversion,
    metadata_from_report,
    prepare_mixed_orthometric_sextet,
    remediation_message,
    run_fidelity_check,
)
from metapulsar.parameter_manager import AlignmentPolicy, ParameterManager
from metapulsar.pint_helpers import (
    create_pint_model,
    dict_to_parfile_string,
    mjd_from_model,
)
from tests.helpers import make_tim_metadata

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

J2145 = dict(
    A1=10.1641056,
    PB=6.83890261,
    EPS1=7.0e-6,
    EPS2=-1.8e-5,
    TASC=55000.0,
    H3=1.8e-7,
)

COMMON_HEAD = """PSR J2145-0750
RAJ 21:45:50.4
DECJ -07:50:18
F0 62.295
PEPOCH 55000
DM 9.0
EPHEM DE440
CLOCK TT(BIPM2021)
UNITS TDB
"""


def _ell1_dict(
    *,
    a1=J2145["A1"],
    pb=J2145["PB"],
    eps1=J2145["EPS1"],
    eps2=J2145["EPS2"],
    tasc=J2145["TASC"],
    free=True,
    binary="ELL1",
    extra=None,
    start=None,
    finish=None,
):
    flag = "1" if free else "0"
    d = {
        "PSR": ["J2145-0750"],
        "RAJ": ["21:45:50.4"],
        "DECJ": ["-07:50:18"],
        "F0": ["62.295"],
        "PEPOCH": ["55000"],
        "DM": ["9.0"],
        "BINARY": [binary],
        "PB": [f"{pb} {flag} 1e-10"],
        "A1": [f"{a1} {flag} 1e-8"],
        "TASC": [f"{tasc} {flag} 1e-6"],
        "EPS1": [f"{eps1} {flag} 1e-9"],
        "EPS2": [f"{eps2} {flag} 1e-9"],
    }
    if start is not None:
        d["START"] = [str(start)]
    if finish is not None:
        d["FINISH"] = [str(finish)]
    if extra:
        d.update(extra)
    return d


def _two_pta(par: dict, *, packages=("pint", "tempo2")):
    return {
        "PINT": copy.deepcopy(par),
        "T2": copy.deepcopy(par),
    }, {"PINT": packages[0], "T2": packages[1]}


def _decide(par, **kw):
    dicts, pkgs = _two_pta(par)
    policy = kw.pop("policy", AlignmentPolicy())
    combine = kw.pop("combine_components", ["binary"])
    timing_packages = kw.pop("timing_packages", pkgs)
    reference_pta = kw.pop("reference_pta", "PINT")
    if "parfile_dicts" in kw:
        dicts = kw.pop("parfile_dicts")
    return decide_binary_conversion(
        dicts,
        reference_pta=reference_pta,
        timing_packages=timing_packages,
        combine_components=combine,
        policy=policy,
        **kw,
    )


def _par_text_from_dict(d: dict) -> str:
    return dict_to_parfile_string(d, format="pint")


def _raw_difference_series(source, converted, *, grid=256, case="B"):
    """Uncentered (source − converted) delay and Shapiro series on one orbit.

    Mirrors §7.5 F1–F5 up to but NOT including the F4c/F5 centering, so tests
    can assert what centering is worth.
    """
    import astropy.units as u
    from pint.simulation import make_fake_toas_fromMJDs

    tasc_ld = mjd_from_model(source, "TASC")
    if tasc_ld is None:
        tasc_ld = mjd_from_model(source, "T0")
    tasc = float(tasc_ld) if tasc_ld is not None else 0.0
    pb = _model_value(source, "PB") / 86400.0  # canonical PB is seconds
    mjds = np.asarray(tasc + (np.arange(grid) / float(grid)) * pb, dtype=float)
    toas = make_fake_toas_fromMJDs(
        MJDs=mjds * u.d, model=source, obs="@", freq=np.inf * u.MHz
    )
    d_total = np.asarray(source.delay(toas).to_value(u.s), dtype=float) - np.asarray(
        converted.delay(toas).to_value(u.s), dtype=float
    )
    d_s_src = _stand_alone_delay_s(source, toas)
    d_s_conv = _stand_alone_delay_s(converted, toas)
    h3 = _model_value(source, "H3")
    stig = _model_value(converted, "STIGMA")
    if case in ("B", "C", "D") and stig:
        d_s_src = _absorbed_shapiro_to_full(d_s_src, mjds, source, h3, stig)
    return d_total, d_s_src - d_s_conv, mjds


# ---------------------------------------------------------------------------
# Runtime split
# ---------------------------------------------------------------------------
#
# The decision matrix (T1-T17, policy validation, domain refusals, patch and
# postcondition shape) is pure dict work and stays in ``make fast`` -- that is
# the layer you iterate on.
#
# The physics tests drive the §7.5 harness, which builds PINT models and
# evaluates 1024-point delay grids at up to three anchors, twice per conversion
# (C6 in memory, C6b on the reloaded par). Left unmarked they were ~5 min and
# dominated ``make fast``. They are marked ``slow`` instead, so they still run
# under ``make test`` and in CI but do not blunt the fast target.
#
# This is a deliberate, narrow deviation from §12.4 ("MetaPulsar unit tests run
# under make fast"): the *coverage* is unchanged, only which target pays for it.
slow = pytest.mark.slow

# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_policy_validation_binary_fields():
    with pytest.raises(ValueError, match="binary_conversion"):
        AlignmentPolicy(binary_conversion="maybe")
    with pytest.raises(ValueError, match="binary_conversion_threshold_s"):
        AlignmentPolicy(binary_conversion_threshold_s=0.0)
    with pytest.raises(ValueError, match="binary_conversion_threshold_s"):
        AlignmentPolicy(binary_conversion_threshold_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="binary_fidelity_floor_s"):
        AlignmentPolicy(binary_fidelity_floor_s=-1.0)
    with pytest.raises(ValueError, match="binary_fidelity_tolerance_factor"):
        AlignmentPolicy(binary_fidelity_tolerance_factor=0.0)
    with pytest.raises(ValueError, match="binary_fidelity_tolerance_factor"):
        AlignmentPolicy(binary_fidelity_tolerance_factor=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported_binary"):
        AlignmentPolicy(unsupported_binary="strip")  # type: ignore[arg-type]
    # Existing checks retained
    with pytest.raises(ValueError, match="bipm_version"):
        AlignmentPolicy(bipm_version=99)
    with pytest.raises(ValueError, match="ne_sw"):
        AlignmentPolicy(ne_sw=-1.0)


def test_policy_validation_h3_only_t41():
    with pytest.raises(ValueError, match="stigma_central"):
        AlignmentPolicy(h3_only="sample_stigma")
    with pytest.raises(ValueError, match="stigma_provenance"):
        AlignmentPolicy(h3_only="sample_stigma", stigma_central=0.37)
    with pytest.raises(ValueError, match="stigma_central"):
        AlignmentPolicy(
            h3_only="sample_stigma",
            stigma_central=1.5,
            stigma_provenance="x",
        )
    with pytest.raises(ValueError, match="require h3_only"):
        AlignmentPolicy(stigma_central=0.37)


# ---------------------------------------------------------------------------
# Decision matrix T1–T17
# ---------------------------------------------------------------------------


def test_t1_convert_gate_fired():
    # A1=10, e such that two-term scale ~ 2e-9
    d = _decide(_ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0))
    assert d.outcome == "convert"
    assert d.reason == "gate_fired"
    assert d.target_family == "DD"
    assert d.scale is not None and d.scale.scale_s > 1e-9


def test_t2_skip_below_threshold():
    d = _decide(_ell1_dict(a1=0.1, eps1=1e-5, eps2=0.0))
    assert d.outcome == "skip"
    assert d.reason == "below_threshold"


def test_t3_always_bypasses_threshold():
    d = _decide(
        _ell1_dict(a1=0.1, eps1=1e-5, eps2=0.0),
        policy=AlignmentPolicy(binary_conversion="always"),
    )
    assert d.outcome == "convert"
    assert d.reason == "gate_fired"


def test_t4_policy_off():
    d = _decide(
        _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0),
        policy=AlignmentPolicy(binary_conversion="off"),
    )
    assert d.outcome == "skip"
    assert d.reason == "policy_off"


def test_t5_binary_not_shared():
    d = _decide(
        _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0),
        combine_components=["astrometry"],
    )
    assert d.outcome == "skip"
    assert d.reason == "binary_not_shared"


@pytest.mark.parametrize("pkgs", [("pint", "pint"), ("tempo2", "tempo2")])
def test_t6_single_engine(pkgs):
    d = _decide(
        _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0),
        timing_packages={"PINT": pkgs[0], "T2": pkgs[1]},
    )
    assert d.outcome == "skip"
    assert d.reason == "single_engine_stack"


@pytest.mark.parametrize("pkgs", [("pint", "pint"), ("tempo2", "tempo2")])
def test_t6_force_mixed_engine_bypasses_single_engine_skip(pkgs):
    d = _decide(
        _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0),
        timing_packages={"PINT": pkgs[0], "T2": pkgs[1]},
        force_mixed_engine=True,
    )
    assert d.reason != "single_engine_stack"


def test_t7_not_ell1_family():
    par = _ell1_dict()
    par["BINARY"] = ["DD"]
    for k in ("EPS1", "EPS2", "TASC"):
        par.pop(k, None)
    par["ECC"] = ["1e-5 1"]
    par["OM"] = ["10 1"]
    par["T0"] = ["55000 1"]
    d = _decide(par)
    assert d.outcome == "skip"
    assert d.reason == "not_ell1_family"

    # T2-Kepler (no EPS)
    par2 = _ell1_dict(binary="T2")
    for k in ("EPS1", "EPS2", "TASC"):
        par2.pop(k, None)
    par2["ECC"] = ["1e-5 1"]
    par2["OM"] = ["10 1"]
    par2["T0"] = ["55000 1"]
    d2 = _decide(par2)
    assert d2.outcome == "skip"
    assert d2.reason == "not_ell1_family"


def test_t8_t2_eps_converts():
    d = _decide(_ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="T2"))
    assert d.outcome == "convert"
    assert d.source_family == "T2-EPS"
    assert d.target_family == "DD"


def test_t9_ell1h_h3_only_unsupported():
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="ELL1H")
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    d = _decide(par)
    assert d.outcome == "unsupported"
    assert d.reason == "ell1h_h3_only_underdetermined"


def test_t10_ell1h_h3_stigma_converts():
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="ELL1H")
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    d = _decide(par)
    assert d.outcome == "convert"
    assert d.target_family == "DDH"
    assert d.reason == "gate_fired"


def test_t11_ell1k_unsupported():
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="ELL1K")
    d = _decide(par)
    assert d.outcome == "unsupported"
    assert d.reason == "ell1k_secular_terms_unvalidated"


def test_t12_ell1_plus_h3_classified_ell1h():
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="ELL1")
    par["H3"] = [f"{J2145['H3']} 1"]
    d = _decide(par)
    assert d.source_family == "ELL1H"
    assert d.outcome == "unsupported"
    assert d.reason == "ell1h_h3_only_underdetermined"


def test_t2_wrapper_resolved_through_pint_model_builder():
    """``BINARY T2`` is classified by the component PINT actually builds.

    MetaPulsar loads every model with ``allow_T2=True``, so the family has to
    follow ``guess_binary_model``, not a local key heuristic.
    """
    kepler = {"ECC": ["1e-5 1"], "OM": ["10 1"], "T0": ["55000 1"]}

    def _t2(extra, *, keplerian=False):
        par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="T2")
        if keplerian:
            for key in ("EPS1", "EPS2", "TASC"):
                par.pop(key, None)
            par.update(kepler)
        par.update(extra)
        return par

    # Laplace-Lagrange T2: ELL1, and convertible.
    plain = _decide(_t2({}))
    assert plain.resolved_binary_model == "ELL1"
    assert plain.source_family == "T2-EPS"
    assert plain.outcome == "convert"

    # T2 carrying the orthometric pair is ELL1H, so it targets DDH.
    orthometric = _decide(
        _t2({"H3": [f"{J2145['H3']} 1 1e-10"], "STIGMA": ["0.5 1 0.01"]})
    )
    assert orthometric.resolved_binary_model == "ELL1H"
    assert orthometric.source_family == "ELL1H"
    assert orthometric.target_family == "DDH"

    # Keplerian T2 variants are DD-family: never ELL1-family, gate never fires,
    # but the resolved component is recorded so the skip can be explained.
    for extra, expected in (
        ({}, "DD"),
        ({"KIN": ["80 1"], "KOM": ["60 1"]}, "DDK"),
        ({"H3": ["1e-8 1"], "STIGMA": ["0.9 1"]}, "DDH"),
    ):
        decision = _decide(_t2(extra, keplerian=True))
        assert decision.resolved_binary_model == expected
        assert decision.outcome == "skip"
        assert decision.reason == "not_ell1_family"
        assert decision.source_family is None


def test_declared_binary_model_is_never_overridden_by_the_guess():
    """PINT only reinterprets ``BINARY T2``; a declared model stands as written.

    ``guess_binary_model`` is parameter-driven, so it answers ``ELL1`` for an
    ``ELL1K`` par (``ELL1`` accepts ``OMDOT``). Trusting it there would silently
    downgrade a periastron-advance model to the plain family.
    """
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="ELL1K")
    par["OMDOT"] = ["0.001 1"]
    decision = _decide(par)
    assert decision.resolved_binary_model == "ELL1K"
    assert decision.source_family == "ELL1k"
    assert decision.outcome == "unsupported"
    assert decision.reason == "ell1k_secular_terms_unvalidated"


def test_t2_without_laplace_coordinates_is_not_ell1_family():
    """A circular T2 par resolves to ELL1 but has no EPS for the gate to use."""
    par = _ell1_dict(a1=10.0, binary="T2")
    for key in ("EPS1", "EPS2"):
        par.pop(key, None)
    decision = _decide(par)
    assert decision.resolved_binary_model == "ELL1"
    assert decision.outcome == "skip"
    assert decision.reason == "not_ell1_family"


def test_t2_uncoverable_parameter_set_is_not_ell1_family():
    """No single PINT component covers EPS *and* Keplerian coordinates.

    ``guess_binary_model`` returns an empty list, which is the case PINT itself
    refuses to build, so the gate must not fire on it either.
    """
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="T2")
    par["ECC"] = ["1e-5 1"]
    par["OM"] = ["10 1"]
    par["T0"] = ["55000 1"]
    decision = _decide(par)
    assert decision.resolved_binary_model is None
    assert decision.outcome == "skip"
    assert decision.reason == "not_ell1_family"


def test_t13_fb_unsupported():
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0)
    par["FB0"] = ["1e-6 1"]
    d = _decide(par)
    assert d.outcome == "unsupported"
    assert d.reason == "fb_orbital_series_unsupported"


def test_t14_always_does_not_widen_supported_set():
    par = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, binary="ELL1H")
    par["H3"] = [f"{J2145['H3']} 1"]
    d = _decide(par, policy=AlignmentPolicy(binary_conversion="always"))
    assert d.outcome == "unsupported"
    assert d.reason == "ell1h_h3_only_underdetermined"


def test_t15_divergent_blocks_raise():
    dicts, pkgs = _two_pta(_ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0))
    dicts["T2"]["A1"] = ["9.0 1 1e-8"]
    with pytest.raises(BinaryConversionError, match="shared_binary_blocks_diverge"):
        decide_binary_conversion(
            dicts,
            reference_pta="PINT",
            timing_packages=pkgs,
            combine_components=["binary"],
            policy=AlignmentPolicy(),
        )


def test_t16_numeric_gate_and_span():
    # Fire / skip around 1 ns with two-term formula
    # Solve roughly: a1=10, pb=6.84 → nb≈1.06e-5; scale≈10 e^2 + 0.5*nb*100*e
    # At e=1.0e-5: 1e-9 + 5.3e-9 = 6.3e-9
    # Need scale ~1.1e-9: try smaller e
    pb = 6.83890261
    nb = 2 * math.pi / (pb * 86400)
    a1 = 10.0

    def scale(e):
        return a1 * e**2 + 0.5 * nb * a1**2 * e

    # Binary-search the eccentricity that puts the two-term scale on
    # either side of the 1 ns gate.
    lo, hi = 1e-7, 5e-5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if scale(mid) < 1.1e-9:
            lo = mid
        else:
            hi = mid
    e_fire = hi
    lo, hi = 1e-7, 5e-5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if scale(mid) < 0.9e-9:
            lo = mid
        else:
            hi = mid
    e_skip = lo

    d_fire = _decide(_ell1_dict(a1=a1, pb=pb, eps1=e_fire, eps2=0.0))
    assert d_fire.outcome == "convert"
    assert d_fire.scale is not None
    assert d_fire.scale.scale_s == pytest.approx(scale(e_fire), rel=1e-6)

    d_skip = _decide(_ell1_dict(a1=a1, pb=pb, eps1=e_skip, eps2=0.0))
    assert d_skip.outcome == "skip"
    assert d_skip.reason == "below_threshold"

    # EPS-dots: e(START) pushes over threshold. Token is the Tempo scaled
    # spelling: "1.2" means 1.2e-12 /s (PINT's 1e-12/s unit, applied by the
    # SI boundary), i.e. ~1e-4 of eccentricity drift over the ±1000 d span.
    e_ref = e_skip
    par = _ell1_dict(
        a1=a1,
        pb=pb,
        eps1=e_ref,
        eps2=0.0,
        start=54000.0,
        finish=56000.0,
        extra={"EPS1DOT": ["1.2 0"], "EPS2DOT": ["0 0"]},
    )
    d_dots = _decide(par)
    assert d_dots.outcome == "convert"
    assert d_dots.scale is not None and d_dots.scale.span_known
    assert d_dots.scale.span_provenance == "par"

    # dots without START/FINISH, reference gate NOT firing → gate_span_unknown
    par2 = _ell1_dict(
        a1=a1,
        pb=pb,
        eps1=e_skip,
        eps2=0.0,
        extra={"EPS1DOT": ["1e-14 0"], "EPS2DOT": ["0 0"]},
    )
    d_unk = _decide(par2)
    assert d_unk.outcome == "unsupported"
    assert d_unk.reason == "gate_span_unknown"
    assert d_unk.scale is not None and d_unk.scale.span_provenance is None

    # dots without START/FINISH, reference gate firing → convert, span_known=False
    par3 = _ell1_dict(
        a1=a1,
        pb=pb,
        eps1=e_fire,
        eps2=0.0,
        extra={"EPS1DOT": ["1e-14 0"], "EPS2DOT": ["0 0"]},
    )
    d_fire2 = _decide(par3)
    assert d_fire2.outcome == "convert"
    assert d_fire2.scale is not None and d_fire2.scale.span_known is False
    assert d_fire2.scale.span_provenance is None


def _subthreshold_eps1(*, a1=10.0, pb=6.83890261) -> float:
    """EPS1 putting the two-term reference scale just under 1 ns (T16 helper)."""
    nb = 2.0 * math.pi / (pb * 86400.0)

    def scale(e: float) -> float:
        return a1 * e**2 + 0.5 * nb * a1**2 * e

    lo, hi = 1e-7, 5e-5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if scale(mid) < 0.9e-9:
            lo = mid
        else:
            hi = mid
    return lo


def test_gate_span_from_tim_metadata():
    """Sub-threshold + dots + no START/FINISH: tim span certifies below_threshold."""
    a1, pb = 10.0, 6.83890261
    e_skip = _subthreshold_eps1(a1=a1, pb=pb)
    # Tiny dots: present (so D7b would fire without a span) but negligible over
    # a ~2000-day baseline, matching the AEI-DR2 sub-threshold refusals.
    tiny_dots = {"EPS1DOT": ["1e-20 0"], "EPS2DOT": ["0 0"]}

    par = _ell1_dict(
        a1=a1,
        pb=pb,
        eps1=e_skip,
        eps2=0.0,
        extra=tiny_dots,
    )

    d_tim = _decide(par, span_mjd=(54000.0, 56000.0))
    assert d_tim.outcome == "skip"
    assert d_tim.reason == "below_threshold"
    assert d_tim.scale is not None
    assert d_tim.scale.span_known is True
    assert d_tim.scale.span_provenance == "tim"

    d_unk = _decide(par)
    assert d_unk.outcome == "unsupported"
    assert d_unk.reason == "gate_span_unknown"
    assert d_unk.scale is not None and d_unk.scale.span_provenance is None

    # Par START/FINISH win over an explicit tim span.
    par_par = _ell1_dict(
        a1=a1,
        pb=pb,
        eps1=e_skip,
        eps2=0.0,
        start=54000.0,
        finish=56000.0,
        extra=tiny_dots,
    )
    d_par = _decide(par_par, span_mjd=(53000.0, 57000.0))
    assert d_par.scale is not None
    assert d_par.scale.span_provenance == "par"


def test_parameter_manager_feeds_tim_span_to_gate(tmp_path):
    """Factory-style file_data with tim_metadata certifies sub-threshold dots."""
    a1, pb = 10.0, 6.83890261
    e_skip = _subthreshold_eps1(a1=a1, pb=pb)
    par_text = COMMON_HEAD + (
        f"BINARY ELL1\nPB {pb} 1 1e-10\n"
        f"A1 {a1} 1 1e-8\nTASC {J2145['TASC']} 1 1e-6\n"
        f"EPS1 {e_skip} 1 1e-9\nEPS2 0.0 1 1e-9\n"
        "EPS1DOT 1e-20 0\nEPS2DOT 0 0\n"
    )
    tim_meta = make_tim_metadata(mjd_min=54000.0, mjd_max=56000.0)

    pm = ParameterManager(
        file_data={
            "PINT": {
                "par_content": par_text,
                "timing_package": "pint",
                "tim_metadata": tim_meta,
            },
            "T2": {
                "par_content": par_text,
                "timing_package": "tempo2",
                "tim_metadata": tim_meta,
            },
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
        output_dir=tmp_path / "out_tim_span",
        pulsar_name="J2145",
        alignment_policy=AlignmentPolicy(ne_sw=0.0),
    )
    assert pm._tim_span_mjd() == (54000.0, 56000.0)
    pm.make_parfiles_shared()
    report = pm.last_binary_conversion_report
    assert report is not None
    assert report.decision.outcome == "skip"
    assert report.decision.reason == "below_threshold"
    assert report.decision.scale is not None
    assert report.decision.scale.span_provenance == "tim"

    pm_no_tim = ParameterManager(
        file_data={
            "PINT": {"par_content": par_text, "timing_package": "pint"},
            "T2": {"par_content": par_text, "timing_package": "tempo2"},
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
        output_dir=tmp_path / "out_no_tim_span",
        pulsar_name="J2145",
        alignment_policy=AlignmentPolicy(ne_sw=0.0),
    )
    with pytest.raises(BinaryConversionError, match="gate_span_unknown"):
        pm_no_tim.make_parfiles_shared()


def test_t17_fit_patterns():
    base = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=True)
    assert _decide(base).outcome == "convert"

    frozen = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=False)
    assert _decide(frozen).outcome == "convert"

    mixed = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=True)
    mixed["EPS2"] = [f"{1.4e-5} 0 1e-9"]
    d = _decide(mixed)
    assert d.outcome == "unsupported"
    assert d.reason == "unsupported_fit_pattern"

    tasc_only = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=False)
    tasc_only["TASC"] = ["55000.0 1 1e-6"]
    assert _decide(tasc_only).reason == "unsupported_fit_pattern"

    dots_split = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=True)
    dots_split["EPS1DOT"] = ["1e-14 1"]
    dots_split["EPS2DOT"] = ["1e-14 0"]
    assert _decide(dots_split).reason == "unsupported_fit_pattern"

    free_pb_frozen_triple = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=False)
    free_pb_frozen_triple["PB"] = ["6.83890261 1 1e-10"]
    assert _decide(free_pb_frozen_triple).reason == "unsupported_fit_pattern"

    free_dots = _ell1_dict(a1=10.0, eps1=1.4e-5, eps2=0.0, free=True)
    free_dots["EPS1DOT"] = ["1e-14 1"]
    free_dots["EPS2DOT"] = ["-2e-14 1"]
    d = _decide(free_dots)
    assert d.outcome == "convert"


def _ppta_mixed_sextet_par():
    """Kepler free, Shapiro frozen — PPTA DR3 J1545-4550 house style."""
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        free=True,
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 0 1e-10"]
    par["STIG"] = ["0.5 0 0.01"]
    return par


def test_mixed_orthometric_sextet_error_policy_refuses():
    par = _ppta_mixed_sextet_par()
    dicts, pkgs = _two_pta(par)
    policy = AlignmentPolicy(
        binary_conversion="always", mixed_orthometric_sextet="error"
    )
    d = decide_binary_conversion(
        dicts,
        reference_pta="PINT",
        timing_packages=pkgs,
        combine_components=["binary"],
        policy=policy,
    )
    assert d.outcome == "unsupported"
    assert d.reason == "unsupported_fit_pattern"
    assert "mixed orthometric sextet" in (d.warnings[0] if d.warnings else "")
    assert prepare_mixed_orthometric_sextet(dicts, policy=policy, decision=d) == ()
    assert _mixed_orthometric_sextet_detail(dicts["PINT"]) is not None


def test_mixed_orthometric_sextet_warn_unfreeze_then_converts():
    par = _ppta_mixed_sextet_par()
    dicts, pkgs = _two_pta(par)
    policy = AlignmentPolicy(binary_conversion="always")
    d = decide_binary_conversion(
        dicts,
        reference_pta="PINT",
        timing_packages=pkgs,
        combine_components=["binary"],
        policy=policy,
    )
    assert d.outcome == "unsupported"
    unfrozen = prepare_mixed_orthometric_sextet(dicts, policy=policy, decision=d)
    assert unfrozen == ("H3", "STIGMA")
    for pta_par in dicts.values():
        assert _fit_flag(pta_par, "H3")
        assert _fit_flag(pta_par, "STIG", "STIGMA")
        assert _mixed_orthometric_sextet_detail(pta_par) is None
    d = decide_binary_conversion(
        dicts,
        reference_pta="PINT",
        timing_packages=pkgs,
        combine_components=["binary"],
        policy=policy,
    )
    assert d.outcome == "convert"
    assert d.target_family == "DDH"


@slow
def test_mixed_orthometric_sextet_unfreeze_converts_with_fidelity():
    par = _ppta_mixed_sextet_par()
    dicts, _pkgs = _two_pta(par)
    policy = AlignmentPolicy(binary_conversion="always")
    decision = decide_binary_conversion(
        dicts,
        reference_pta="PINT",
        timing_packages={"PINT": "pint", "T2": "tempo2"},
        combine_components=["binary"],
        policy=policy,
    )
    assert decision.outcome == "unsupported"
    prepare_mixed_orthometric_sextet(dicts, policy=policy, decision=decision)
    decision = decide_binary_conversion(
        dicts,
        reference_pta="PINT",
        timing_packages={"PINT": "pint", "T2": "tempo2"},
        combine_components=["binary"],
        policy=policy,
    )
    assert decision.outcome == "convert"
    patch, record = convert_shared_binary(
        dicts["PINT"],
        decision,
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )
    assert patch.binary_value == "DDH"
    assert "H3" in record.target_free_params
    assert "STIGMA" in record.target_free_params
    assert record.fidelity.total_max_abs_s < 2e-9


def test_mixed_orthometric_sextet_unfreeze_preserves_bare_uncertainties():
    par = _ppta_mixed_sextet_par()
    # Valid PINT syntax: VALUE UNCERTAINTY, with no explicit fit flag.
    par["H3"] = [f"{J2145['H3']} 1e-10"]
    par["STIG"] = ["0.5 0.01"]
    dicts, pkgs = _two_pta(par)
    policy = AlignmentPolicy(binary_conversion="always")
    decision = decide_binary_conversion(
        dicts,
        reference_pta="PINT",
        timing_packages=pkgs,
        combine_components=["binary"],
        policy=policy,
    )
    prepare_mixed_orthometric_sextet(dicts, policy=policy, decision=decision)
    assert dicts["PINT"]["H3"] == [f"{J2145['H3']} 1 1e-10"]
    assert dicts["PINT"]["STIG"] == ["0.5 1 0.01"]


@pytest.mark.parametrize(
    ("policy", "a1", "expected_reason"),
    [
        (AlignmentPolicy(binary_conversion="off"), J2145["A1"], "policy_off"),
        (AlignmentPolicy(), 0.1, "below_threshold"),
    ],
)
def test_manager_skip_does_not_unfreeze_mixed_sextet(
    policy, a1, expected_reason, caplog
):
    par = _ppta_mixed_sextet_par()
    par["A1"] = [f"{a1} 1 1e-8"]
    dicts, _ = _two_pta(par)
    before = copy.deepcopy(dicts)
    pm = ParameterManager(
        file_data={
            "PINT": {"timing_package": "pint"},
            "T2": {"timing_package": "tempo2"},
        },
        combine_components=["binary"],
        pulsar_name="J1545-4550",
        alignment_policy=policy,
    )

    with caplog.at_level(logging.WARNING):
        pm._maybe_convert_shared_binary(dicts)

    assert dicts == before
    assert pm.last_binary_conversion_report.decision.reason == expected_reason
    assert not any("mixed orthometric sextet" in r.message for r in caplog.records)


def test_manager_mixed_sextet_error_is_not_downgraded_by_keep():
    par = _ppta_mixed_sextet_par()
    dicts, _ = _two_pta(par)
    before = copy.deepcopy(dicts)
    pm = ParameterManager(
        file_data={
            "PINT": {"timing_package": "pint"},
            "T2": {"timing_package": "tempo2"},
        },
        combine_components=["binary"],
        pulsar_name="J1545-4550",
        alignment_policy=AlignmentPolicy(
            binary_conversion="always",
            mixed_orthometric_sextet="error",
            unsupported_binary="keep",
        ),
    )

    with pytest.raises(BinaryConversionError, match="mixed orthometric sextet"):
        pm._maybe_convert_shared_binary(dicts)
    assert dicts == before


@slow
def test_manager_warns_with_pulsar_and_converts_mixed_sextet(caplog):
    par = _ppta_mixed_sextet_par()
    dicts, _ = _two_pta(par)
    pm = ParameterManager(
        file_data={
            "PINT": {"timing_package": "pint"},
            "T2": {"timing_package": "tempo2"},
        },
        combine_components=["binary"],
        pulsar_name="J1545-4550",
        alignment_policy=AlignmentPolicy(binary_conversion="always"),
    )

    with caplog.at_level(logging.WARNING):
        pm._maybe_convert_shared_binary(dicts)

    messages = [
        r.message for r in caplog.records if "mixed orthometric sextet" in r.message
    ]
    assert len(messages) == 1
    assert "J1545-4550" in messages[0]
    assert "PINT" in messages[0]
    assert "H3, STIGMA" in messages[0]
    assert all(par["BINARY"][0].split()[0] == "DDH" for par in dicts.values())


# ---------------------------------------------------------------------------
# Conversion / patch / fidelity T18–T21
# ---------------------------------------------------------------------------


@slow
def test_t18_plain_conversion_values():
    from pint.binaryconvert import convert_binary

    par = _ell1_dict(a1=10.0, eps1=7e-6, eps2=-1.8e-5, free=True)
    par["PBDOT"] = ["1e-12 0 1e-14"]
    par["A1DOT"] = ["1e-14 0 1e-16"]
    decision = _decide(par, policy=AlignmentPolicy(binary_conversion="always"))
    assert decision.outcome == "convert"
    patch, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(binary_conversion="always"),
        ell1h_shapiro="full",
    )
    assert patch.binary_value == "DD"
    assert "EPS1" in patch.removed_keys
    assert "TASC" in patch.removed_keys
    keys_added = {k for k, _ in patch.added_lines}
    assert {"ECC", "OM", "T0"} <= keys_added
    assert "PB" in keys_added  # PBDOT present
    assert "A1" in keys_added  # A1DOT present

    # Raw convert_binary must NOT match corrected T0 when eps1 nonzero
    model = create_pint_model(par)
    raw = convert_binary(model, "DD")
    t0_line = dict(patch.added_lines)["T0"].split()[0]
    assert not np.isclose(float(t0_line), float(raw.T0.value), rtol=0, atol=1e-12)

    assert "EPS1" in record.source_free_params
    assert "ECC" in record.target_free_params


def test_t19_patch_application_preserves_nonbinary():
    par_a = _ell1_dict(a1=10.0, eps1=7e-6, eps2=-1.8e-5)
    par_a["JUMP"] = ["-fe L_wide 0 1"]
    par_a["FD1"] = ["1e-5 1"]
    par_a["EFAC"] = ["-f L_wide 1.1"]
    par_a["TNRedAmp"] = ["-13.0"]  # non-binary noise hyperparam
    par_b = copy.deepcopy(par_a)
    par_b["JUMP"] = ["-fe S_band 0.5 1"]
    par_b["EFAC"] = ["-f S_band 1.2"]

    decision = _decide(par_a, policy=AlignmentPolicy(binary_conversion="always"))
    patch, _ = convert_shared_binary(
        par_a,
        decision,
        pta_names=("A", "B"),
        policy=AlignmentPolicy(binary_conversion="always"),
        ell1h_shapiro="full",
    )
    # Use library snapshot helper for postconditions
    from metapulsar.binary_family_convert import _nonbinary_snapshot

    pre_nb = {"A": _nonbinary_snapshot(par_a), "B": _nonbinary_snapshot(par_b)}
    apply_binary_patch(par_a, patch)
    apply_binary_patch(par_b, patch)
    assert_postconditions(
        {"A": par_a, "B": par_b}, target_family="DD", pre_nonbinary=pre_nb
    )
    assert par_a["JUMP"] == ["-fe L_wide 0 1"]
    assert par_b["JUMP"] == ["-fe S_band 0.5 1"]
    assert par_a["TNRedAmp"] == ["-13.0"]
    assert par_b["TNRedAmp"] == ["-13.0"]


@slow
def test_t20_fidelity_sentinels():
    from pint.binaryconvert import convert_binary

    par = _ell1_dict(a1=10.0, eps1=7e-6, eps2=-1.8e-5)
    policy = AlignmentPolicy(binary_conversion="always")
    source = create_pint_model(par)
    good = convert_binary(source, "DD")
    apply_conversion_corrections(good, source)

    # OM perturbed
    bad = convert_binary(source, "DD")
    apply_conversion_corrections(bad, source)
    bad.OM.value = float(bad.OM.value) + np.degrees(1e-6)
    with pytest.raises(BinaryConversionError, match="fidelity_check_failed"):
        run_fidelity_check(source, bad, policy=policy, plain=True)

    # Gauge bug: naive ELL1H→DDH via convert_binary
    par_h = _ell1_dict(
        a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"], binary="ELL1H"
    )
    par_h["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par_h["STIGMA"] = ["0.5 1 0.01"]
    src_h = create_pint_model(par_h, ell1h_shapiro="absorbed")
    try:
        naive = convert_binary(src_h, "DDH")
    except Exception as exc:
        pytest.skip(f"convert_binary ELL1H→DDH unavailable: {exc}")
    with pytest.raises(BinaryConversionError, match="fidelity_check_failed"):
        run_fidelity_check(src_h, naive, policy=policy, plain=False, case="B")

    # H3 deletion (Appendix C / AC#2): the third sentinel. Both engines would
    # agree perfectly on the stripped model, so only a source-vs-target check
    # within one engine can see it. Strip H3 by converting the matching plain
    # ELL1 -- the O(10^2) ns Shapiro term simply vanishes.
    par_plain = _ell1_dict(
        a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"], binary="ELL1"
    )
    stripped = convert_binary(create_pint_model(par_plain), "DD")
    apply_conversion_corrections(stripped, create_pint_model(par_plain))
    assert not _shapiro_present(stripped)
    with pytest.raises(BinaryConversionError, match="fidelity_check_failed") as excinfo:
        run_fidelity_check(src_h, stripped, policy=policy, plain=False, case="B")
    # It fails by the deleted Shapiro amplitude, ~(4/3)*H3 ~ 10^2 ns, not by
    # some incidental sub-ns bookkeeping difference.
    message = str(excinfo.value)
    total_max = float(message.split("total_max=")[1].split()[0])
    assert total_max > 1e-8, message


def test_t21_report_lifecycle(tmp_path):
    # unsupported keep
    par_h = COMMON_HEAD + (
        f"BINARY ELL1H\nPB {J2145['PB']} 1 1e-10\n"
        f"A1 {J2145['A1']} 1 1e-8\nTASC {J2145['TASC']} 1 1e-6\n"
        f"EPS1 {J2145['EPS1']} 1 1e-9\nEPS2 {J2145['EPS2']} 1 1e-9\n"
        f"H3 {J2145['H3']} 1 1e-10\n"
    )
    pm = ParameterManager(
        file_data={
            "PINT": {"par_content": par_h, "timing_package": "pint"},
            "T2": {"par_content": par_h, "timing_package": "tempo2"},
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
        output_dir=tmp_path / "out1",
        pulsar_name="J2145",
        alignment_policy=AlignmentPolicy(unsupported_binary="keep", ne_sw=0.0),
    )
    written = pm.make_parfiles_shared()
    assert pm.last_binary_conversion_report is not None
    assert (
        pm.last_binary_conversion_report.decision.reason
        == "ell1h_h3_only_underdetermined"
    )
    assert "ELL1H" in Path(written["PINT"]).read_text(encoding="utf-8")

    # error raises
    pm_err = ParameterManager(
        file_data={
            "PINT": {"par_content": par_h, "timing_package": "pint"},
            "T2": {"par_content": par_h, "timing_package": "tempo2"},
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
        output_dir=tmp_path / "out2",
        pulsar_name="J2145",
        alignment_policy=AlignmentPolicy(ne_sw=0.0),
    )
    with pytest.raises(BinaryConversionError, match="ell1h_h3_only_underdetermined"):
        pm_err.make_parfiles_shared()

    # reset on reuse: first converts plain ELL1, second is DD → skip
    par_ell1 = COMMON_HEAD + (
        f"BINARY ELL1\nPB {J2145['PB']} 1 1e-10\n"
        f"A1 {J2145['A1']} 1 1e-8\nTASC {J2145['TASC']} 1 1e-6\n"
        f"EPS1 {J2145['EPS1']} 1 1e-9\nEPS2 {J2145['EPS2']} 1 1e-9\n"
    )
    pm2 = ParameterManager(
        file_data={
            "PINT": {"par_content": par_ell1, "timing_package": "pint"},
            "T2": {"par_content": par_ell1, "timing_package": "tempo2"},
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
        output_dir=tmp_path / "out3",
        pulsar_name="J2145",
        alignment_policy=AlignmentPolicy(ne_sw=0.0),
    )
    pm2.make_parfiles_shared()
    assert pm2.last_binary_conversion_report.decision.outcome == "convert"

    # Swap file_data content to DD by rewriting (manager holds file_data)
    par_dd = COMMON_HEAD + (
        f"BINARY DD\nPB {J2145['PB']} 1 1e-10\n"
        f"A1 {J2145['A1']} 1 1e-8\nT0 55003.0 1 1e-6\n"
        f"ECC 1.9e-5 1 1e-9\nOM 158.7 1 0.01\n"
    )
    pm2.file_data["PINT"]["par_content"] = par_dd
    pm2.file_data["T2"]["par_content"] = par_dd
    pm2.make_parfiles_shared()
    assert pm2.last_binary_conversion_report.decision.reason == "not_ell1_family"

    # MetaPulsar.conversion_metadata mirrors the factory-propagated report
    from metapulsar.metapulsar import MetaPulsar

    # Use a lightweight stand-in: conversion_metadata only reads the report attr.
    class _PulsarLike:
        binary_conversion_report = pm2.last_binary_conversion_report

        conversion_metadata = MetaPulsar.conversion_metadata

    assert _PulsarLike().conversion_metadata() is None  # skip / no convert record


# ---------------------------------------------------------------------------
# Rev-2 maps / fidelity T30–T45 (selected load-bearing rows)
# ---------------------------------------------------------------------------


@slow
def test_t30_harmonic_identity():
    from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
    import astropy.units as u

    stig = 0.5
    h3 = J2145["H3"]
    r = h3 / stig**3
    n = 4096
    t = J2145["TASC"] + np.linspace(0.0, J2145["PB"], n, endpoint=False)
    bm = ELL1Hmodel()
    bm.fit_params = ["H3", "STIGMA"]
    bm.update_input(
        barycentric_toa=np.asarray(t, dtype=np.longdouble),
        PB=J2145["PB"] * u.day,
        A1=J2145["A1"] * u.lsec,
        TASC=np.longdouble(J2145["TASC"]) * u.day,
        EPS1=J2145["EPS1"] * u.Unit(""),
        EPS2=J2145["EPS2"] * u.Unit(""),
        H3=h3 * u.s,
        STIGMA=stig * u.Unit(""),
    )
    phi = bm.Phi()
    phi = phi.to_value(u.rad) if hasattr(phi, "to_value") else np.asarray(phi)
    s28 = bm.delayS3p_H3_STIGMA_exact(h3 * u.s, stig).to_value(u.s)
    s29 = bm.delayS_H3_STIGMA_exact(h3 * u.s, stig).to_value(u.s)
    ident = s28 - (s29 - 4 * r * stig * np.sin(phi) + 2 * r * stig**2 * np.cos(2 * phi))
    assert np.max(np.abs(ident)) <= 1e-25


@slow
def test_t31_naive_map_counterexample():
    from pint.binaryconvert import convert_binary

    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    src = create_pint_model(par, ell1h_shapiro="absorbed")
    try:
        naive = convert_binary(src, "DDH")
    except Exception as exc:
        pytest.skip(f"convert_binary ELL1H→DDH unavailable: {exc}")
    with pytest.raises(BinaryConversionError):
        run_fidelity_check(
            src,
            naive,
            policy=AlignmentPolicy(),
            plain=False,
            case="B",
        )


@slow
@pytest.mark.parametrize("stig", [0.5, 0.9])
def test_t32_case_b_identity(stig):
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = [f"{stig} 1 0.01"]
    decision = _decide(par, policy=AlignmentPolicy(binary_conversion="always"))
    assert decision.outcome == "convert"
    assert decision.target_family == "DDH"
    patch, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(binary_conversion="always"),
        ell1h_shapiro="absorbed",
    )
    assert record.gauge == "absorbed"
    assert record.fidelity.total_max_abs_s <= record.fidelity.tolerance_total_s
    assert record.fidelity.total_max_abs_s < 1.5e-9


@slow
def test_t33_case_a_delta_t0_sentinel():
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    decision = _decide(par, policy=AlignmentPolicy(binary_conversion="always"))
    _, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(binary_conversion="always"),
        ell1h_shapiro="full",
    )
    assert record.gauge == "full"
    assert record.fidelity.total_max_abs_s < 1.0e-9


@slow
def test_t34_plain_path_delta_t0_load_bearing():
    from pint.binaryconvert import convert_binary

    par = _ell1_dict(a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"])
    policy = AlignmentPolicy(binary_conversion="always")
    source = create_pint_model(par)
    raw = convert_binary(source, "DD")
    with pytest.raises(BinaryConversionError, match="fidelity_check_failed"):
        run_fidelity_check(source, raw, policy=policy, plain=True)
    apply_conversion_corrections(raw, source)
    rep = run_fidelity_check(source, raw, policy=policy, plain=True)
    assert rep.total_max_abs_s <= rep.tolerance_total_s


@slow
def test_t35_dots_rereferencing():
    from pint.binaryconvert import convert_binary

    par = _ell1_dict(a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"])
    par["PBDOT"] = ["1e-12 0"]
    par["A1DOT"] = ["1e-14 0"]
    par["START"] = ["51350"]
    par["FINISH"] = ["58650"]
    policy = AlignmentPolicy(binary_conversion="always")
    source = create_pint_model(par)
    naive = convert_binary(source, "DD")
    with pytest.raises(BinaryConversionError):
        run_fidelity_check(source, naive, policy=policy, plain=True)
    apply_conversion_corrections(naive, source)
    rep = run_fidelity_check(source, naive, policy=policy, plain=True)
    assert rep.total_max_abs_s <= rep.tolerance_total_s


@slow
def test_t36_case_c_tail_gate():
    # stig=0.5, NHARMS=4 → bound ~72 ns
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1"]
    par["H4"] = [f"{0.5 * J2145['H3']} 1"]
    par["NHARMS"] = ["4"]
    d = _decide(par)
    assert d.outcome == "unsupported"
    assert d.reason == "ell1h_h4_tail_exceeds_tolerance"

    # small stig=0.05 → converts
    h3 = J2145["H3"]
    stig = 0.05
    par2 = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par2["H3"] = [f"{h3} 1 1e-10"]
    par2["H4"] = [f"{stig * h3} 1 1e-10"]
    par2["NHARMS"] = ["7"]
    d2 = _decide(par2, policy=AlignmentPolicy(binary_conversion="always"))
    assert d2.outcome == "convert"
    patch, record = convert_shared_binary(
        par2,
        d2,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(binary_conversion="always"),
        ell1h_shapiro="absorbed",
    )
    assert record.fidelity.total_max_abs_s <= record.fidelity.tolerance_total_s


def test_t37_case_d_default_message():
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1"]
    d = _decide(par)
    assert d.reason == "ell1h_h3_only_underdetermined"
    msg = remediation_message()
    assert "sample_stigma" in msg or "STIGMA" in msg


@slow
def test_t38_case_d_sample_stigma():
    policy = AlignmentPolicy(
        binary_conversion="always",
        h3_only="sample_stigma",
        stigma_central=0.37,
        stigma_provenance="mass-function closure, m_p=1.4",
    )
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    d = _decide(par, policy=policy)
    assert d.outcome == "convert"
    patch, record = convert_shared_binary(
        par,
        d,
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )
    assert record.required_sampling == ("STIGMA",)
    assert record.stigma_provenance == "mass-function closure, m_p=1.4"
    assert any(k == "C" for k, _ in patch.added_lines)
    assert record.fidelity.total_max_abs_s <= record.fidelity.tolerance_total_s


@slow
def test_t39_case_d_scan_sentinel():
    """Map at ς'=0.1 reproduces M2≈36.5 M☉ and residual ≈25 ns order."""
    tsun = 4.925490947e-6
    stig = 0.1
    h3 = J2145["H3"]
    m2 = h3 / (tsun * stig**3)
    assert m2 == pytest.approx(36.5, rel=0.05)
    # residual scale ~ h3*stig for the missing h4
    assert h3 * stig == pytest.approx(1.8e-8, rel=0.05)


def test_t40_domain_refusals():
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = ["-1e-7 1"]
    par["STIGMA"] = ["0.5 1"]
    assert _decide(par, policy=AlignmentPolicy(binary_conversion="always")).reason == (
        "ell1h_domain_violation"
    )

    par2 = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par2["H3"] = [f"{J2145['H3']} 1"]
    par2["STIGMA"] = ["1.2 1"]
    assert _decide(par2, policy=AlignmentPolicy(binary_conversion="always")).reason == (
        "ell1h_domain_violation"
    )

    # 4H3/stig^2 >= 0.01*A1
    par3 = _ell1_dict(a1=1.0, eps1=1.4e-5, eps2=0.0, binary="ELL1H")
    par3["H3"] = ["1e-3 1"]
    par3["STIGMA"] = ["0.5 1"]
    assert _decide(par3, policy=AlignmentPolicy(binary_conversion="always")).reason == (
        "ell1h_domain_violation"
    )


def test_t41_map_roundtrip():
    x_p, e1p, e2p = J2145["A1"], J2145["EPS1"], J2145["EPS2"]
    h3, stig = J2145["H3"], 0.5
    nb = 2 * math.pi / (J2145["PB"] * 86400)
    x_i, e1i, e2i = _absorbed_to_intrinsic(x_p, e1p, e2p, h3, stig, nb)
    # inverse
    x_p2 = x_i + 4 * h3 / stig**2
    e1p2 = (x_i * e1i + 4 * h3 / stig) / x_p2
    e2p2 = (x_i * e2i + 8 * nb * (h3 / stig**2) * x_i) / x_p2
    assert x_p2 / x_p == pytest.approx(1.0, rel=1e-12)
    assert e1p2 == pytest.approx(e1p, rel=1e-12)
    assert e2p2 == pytest.approx(e2p, rel=1e-12)


@slow
def test_t42_fidelity_mean_handling():
    """F4c/F5 centering sentinels: the harness must break without mean removal.

    Encodes review finding 4 — a harness that skipped F4c or left the Shapiro
    component uncentered would pass the same fixtures for the wrong reason.
    """
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    _, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )
    assert record.fidelity.total_max_abs_s < 1e-9

    # (a) The F4c constant matches its prediction to 10%. Rebuild the same
    # source/converted pair and measure the raw (uncentered) mean directly.
    source = create_pint_model(par, ell1h_shapiro="absorbed")
    converted, _ = _convert_ell1h_block(source, "absorbed", policy)
    d_total, d_shapiro, _ = _raw_difference_series(source, converted)
    c_measured = float(np.mean(d_total))
    a1_i = abs(converted.A1.value)
    eps1_i = converted.ECC.value * math.sin(math.radians(converted.OM.value))
    stig = 0.5
    r = J2145["H3"] / stig**3
    c_pred = 1.5 * a1_i * eps1_i - 2.0 * r * math.log1p(stig**2)
    assert c_measured == pytest.approx(c_pred, rel=0.10)

    # (b) Without mean removal the check fails trivially: the raw constant is
    # ~1e-4 s, six orders above the tolerance the centered series passes at.
    assert abs(c_measured) > 1e-6
    assert abs(c_measured) > 1e4 * record.fidelity.tolerance_total_s

    # (c) An UNCENTERED Shapiro component breaks the F5 split: d_roemer picks
    # up the whole Shapiro normalization constant even though d_total is fine.
    centered_total = d_total - np.mean(d_total)
    good_roemer = centered_total - (d_shapiro - np.mean(d_shapiro))
    good_roemer = good_roemer - np.mean(good_roemer)
    bad_roemer = centered_total - d_shapiro
    assert np.max(np.abs(good_roemer)) <= record.fidelity.tolerance_roemer_s
    assert np.max(np.abs(bad_roemer)) > record.fidelity.tolerance_roemer_s


@slow
def test_t42b_f4c_assertion_catches_inconsistent_map():
    """A perturbed A1 shifts the F4c constant → fidelity_check_failed."""
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    policy = AlignmentPolicy(binary_conversion="always")
    source = create_pint_model(par, ell1h_shapiro="absorbed")
    converted, _ = _convert_ell1h_block(source, "absorbed", policy)
    # A 30% A1 error moves the (3/2)*A1*eps1 constant well outside the 10% band.
    converted.A1.value = float(converted.A1.value) * 1.3
    with pytest.raises(BinaryConversionError, match="fidelity_check_failed"):
        run_fidelity_check(source, converted, policy=policy, plain=False, case="B")


@slow
def test_t43_serialization_fidelity():
    # START/FINISH give the harness its span anchors (F1). Without them the
    # grid is one orbit at TASC, where a dropped PB re-referencing accumulates
    # only ~3e-11 s -- below the floor. The re-referencing exists precisely
    # because the error grows over the span, so the sentinel must be measured
    # there.
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        start=53000.0,
        finish=56650.0,
    )
    par["PBDOT"] = ["1e-12 0"]
    par["A1DOT"] = ["1e-14 0"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    patch, record = convert_shared_binary(
        par, decision, pta_names=("PINT", "T2"), policy=policy, ell1h_shapiro="full"
    )
    assert "PB" in {k for k, _ in patch.added_lines}
    assert record.fidelity.total_max_abs_s <= record.fidelity.tolerance_total_s

    # C6b is only load-bearing if it catches a patch that loses a correction.
    # Drop the PB re-emission: the in-memory model still carries PBDOT*tau, so
    # C6 (source vs corrected model) passes -- but the delivered par text keeps
    # the ELL1 PB, and only the reloaded-par check sees that.
    from metapulsar.binary_family_convert import BinaryPatch, run_fidelity_check
    from pint.binaryconvert import convert_binary

    corrupted = BinaryPatch(
        binary_value=patch.binary_value,
        removed_keys=tuple(k for k in patch.removed_keys if k.upper() != "PB"),
        added_lines=tuple((k, v) for k, v in patch.added_lines if k.upper() != "PB"),
    )
    source = create_pint_model(par, ell1h_shapiro="full")
    corrected = convert_binary(create_pint_model(par), "DD")
    apply_conversion_corrections(corrected, source)
    # C6 (in-memory) still passes: the corruption is in the patch, not the model.
    run_fidelity_check(source, corrected, policy=policy, plain=True)

    serialized = copy.deepcopy(par)
    apply_binary_patch(serialized, corrupted)
    reloaded = create_pint_model(serialized, ell1h_shapiro="full")
    with pytest.raises(BinaryConversionError, match="fidelity_check_failed"):
        run_fidelity_check(source, reloaded, policy=policy, plain=True)


@slow
def test_c5_audit_catches_a_patch_that_drops_a_correction():
    """C5's `correction_not_applied` must be reachable.

    It audited the corrected model against a dict parsed straight back out of
    that same model's `as_parfile()`, and its fallback branch compared
    `_model_value(corrected_model, name)` to itself -- so the check could never
    fire whatever the patch contained. The patch is the artifact that lands in
    the pars, so that is what has to be audited.
    """
    from io import StringIO

    from metapulsar.binary_family_convert import (
        BinaryPatch,
        _audit_converter_output,
    )
    from pint.binaryconvert import convert_binary
    from pint.models.model_builder import parse_parfile

    # A1DOT, not PBDOT: at this fixture the PB re-referencing is 4.4e-13 in
    # relative terms, i.e. under the audit's own 1e-12 rtol, so a stale PB is
    # genuinely indistinguishable. The A1 re-referencing is 2.6e-10 relative
    # at a realistic XDOT, which the audit can and must resolve.
    par = _ell1_dict(a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"])
    par["A1DOT"] = ["1e-14 0"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    patch, _ = convert_shared_binary(
        par, decision, pta_names=("PINT", "T2"), policy=policy, ell1h_shapiro="full"
    )
    assert "A1" in {k.upper() for k, _ in patch.added_lines}
    source = create_pint_model(par, ell1h_shapiro="full")
    corrected = convert_binary(create_pint_model(par), "DD")
    apply_conversion_corrections(corrected, source)
    converted_dict = parse_parfile(StringIO(corrected.as_parfile()))

    # The real patch audits clean.
    _audit_converter_output(par, converted_dict, patch, corrected, source)

    # A patch that re-emits A1 with the UNCORRECTED (source) value is caught.
    stale = BinaryPatch(
        binary_value=patch.binary_value,
        removed_keys=patch.removed_keys,
        added_lines=tuple(
            (k, f"{J2145['A1']} 1 1e-8") if k.upper() == "A1" else (k, v)
            for k, v in patch.added_lines
        ),
    )
    with pytest.raises(BinaryConversionError, match="correction_not_applied"):
        _audit_converter_output(par, converted_dict, stale, corrected, source)

    # A patch that drops the re-emission entirely is caught too -- here by the
    # key-universe half of C5, which fires first: A1 is still in removed_keys,
    # so the expected key set no longer contains it while the converted model
    # does. Either reason is a correct refusal; what matters is that it names
    # A1 and does not pass.
    dropped = BinaryPatch(
        binary_value=patch.binary_value,
        removed_keys=patch.removed_keys,
        added_lines=tuple((k, v) for k, v in patch.added_lines if k.upper() != "A1"),
    )
    with pytest.raises(BinaryConversionError, match="A1") as excinfo:
        _audit_converter_output(par, converted_dict, dropped, corrected, source)
    assert "unexpected_converter_output" in str(
        excinfo.value
    ) or "correction_not_applied" in str(excinfo.value)


@slow
def test_c5_passthrough_accepts_xdot_spelling_physical():
    """C5 must accept Tempo2 XDOT spelling when the token is already physical."""
    par = _ell1_dict(a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"])
    par["XDOT"] = ["8.410070170088968e-15 1 8.75e-16"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    assert decision.outcome == "convert"
    patch, _ = convert_shared_binary(
        par, decision, pta_names=("PINT", "T2"), policy=policy, ell1h_shapiro="full"
    )
    assert patch.binary_value == "DD"
    serialized = dict(par)
    apply_binary_patch(serialized, patch)
    assert "XDOT" in serialized or any(k.upper() == "XDOT" for k in serialized)


@slow
def test_c5_passthrough_accepts_xdot_unit_scaled_token():
    """C5 must compare physical A1DOT when XDOT uses Tempo unit_scale tokens."""
    par = _ell1_dict(
        a1=11.003316789,
        pb=16.33534782659533,
        eps1=-4.0581e-6,
        eps2=-9.1166e-6,
        tasc=55819.254684930,
    )
    par["XDOT"] = ["-0.009436 1 0.001891"]  # → physical -9.436e-15 after unit_scale
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    assert decision.outcome == "convert"
    patch, _ = convert_shared_binary(
        par, decision, pta_names=("PINT", "T2"), policy=policy, ell1h_shapiro="full"
    )
    assert patch.binary_value == "DD"


@slow
def test_c5_still_catches_true_a1dot_drift():
    """C5 must still refuse when the converted model's A1DOT physically drifts."""
    from io import StringIO

    from metapulsar.binary_family_convert import _audit_converter_output
    from pint.binaryconvert import convert_binary
    from pint.models.model_builder import parse_parfile

    par = _ell1_dict()
    par["A1DOT"] = ["1e-14 0"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    patch, _ = convert_shared_binary(
        par, decision, pta_names=("PINT", "T2"), policy=policy, ell1h_shapiro="full"
    )
    source = create_pint_model(par, ell1h_shapiro="full")
    corrected = convert_binary(create_pint_model(par), "DD")
    apply_conversion_corrections(corrected, source)
    # Drift before serializing so model and converted_dict stay synchronized.
    corrected.A1DOT.value = float(corrected.A1DOT.value) * 2.0
    converted_dict = parse_parfile(StringIO(corrected.as_parfile()))
    with pytest.raises(
        BinaryConversionError, match="converter_modified_passthrough_key"
    ):
        _audit_converter_output(par, converted_dict, patch, corrected, source)


@slow
def test_t44_metadata_protocol():
    policy = AlignmentPolicy(
        binary_conversion="always",
        h3_only="sample_stigma",
        stigma_central=0.37,
        stigma_provenance="test",
    )
    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    decision = _decide(par, policy=policy)
    patch, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )
    report = BinaryConversionReport(decision=decision, record=record)
    meta = metadata_from_report(report)
    assert meta is not None
    assert meta.required_sampling == ("STIGMA",)
    assert meta.stigma_provenance == "test"
    assert metadata_from_report(None) is None

    # The required_sampling metadata is MetaPulsar-owned and produced without
    # nltiming installed: combination is self-contained. The contract that a
    # STIGMA marked required_sampling is *sampled* (never frozen) is enforced
    # downstream at point-of-use by nltiming's sampler
    # (nonlinear_timing_model._reject_delta_flat_required_sampling), not by
    # refusing to convert here.


def test_t45_intrinsic_dots_finite_difference():
    x_p, e1p, e2p = J2145["A1"], J2145["EPS1"], J2145["EPS2"]
    h3, stig = J2145["H3"], 0.5
    nb = 2 * math.pi / (J2145["PB"] * 86400)
    xdot, e1dot, e2dot = 1e-14, 1e-16, -2e-16
    x_i, e1i, e2i = _absorbed_to_intrinsic(x_p, e1p, e2p, h3, stig, nb)
    e1d_i, e2d_i = _intrinsic_dots(
        x_p, e1p, e2p, e1i, e2i, x_i, xdot, e1dot, e2dot, h3=h3, stig=stig, nb=nb
    )
    # finite difference over ±5 yr
    dt = 5 * 365.25 * 86400
    x_p2 = x_p + xdot * dt
    e1p2 = e1p + e1dot * dt
    e2p2 = e2p + e2dot * dt
    x_i2, e1i2, e2i2 = _absorbed_to_intrinsic(x_p2, e1p2, e2p2, h3, stig, nb)
    e1d_fd = (e1i2 - e1i) / dt
    e2d_fd = (e2i2 - e2i) / dt
    assert e1d_i == pytest.approx(float(e1d_fd), rel=1e-3)
    assert e2d_i == pytest.approx(float(e2d_fd), rel=1e-3)


def test_remediation_message_lists_five():
    msg = remediation_message()
    assert msg.count("\n  ") >= 5
    assert "per_pta" in msg


# ---------------------------------------------------------------------------
# Regression tests for review findings
# ---------------------------------------------------------------------------


@slow
def test_plain_ell1_with_m2_sini_converts():
    """§5.1: M2/SINI are a SUPPORTED plain-ELL1 shape and pass through.

    Regression for an F6 tolerance that used max|d_shapiro difference| instead
    of peak|delayS_source|, making tol_shapiro self-referential (~5x too tight)
    so every ELL1+M2/SINI stack raised fidelity_check_failed.
    """
    par = _ell1_dict(a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"])
    par["M2"] = ["0.5 0"]
    par["SINI"] = ["0.75 0"]
    decision = _decide(par)
    assert decision.outcome == "convert"
    assert decision.target_family == "DD"

    patch, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(),
        ell1h_shapiro="full",
    )
    fid = record.fidelity
    assert fid.shapiro_max_abs_s is not None
    assert fid.tolerance_shapiro_s is not None
    assert fid.shapiro_max_abs_s <= fid.tolerance_shapiro_s
    assert fid.total_max_abs_s <= fid.tolerance_total_s

    # The tolerance is set by the SOURCE Shapiro amplitude (~7 us here), not by
    # the difference it bounds. The self-referential form collapses onto the
    # floor and would reject this conversion.
    e_max = math.hypot(J2145["EPS1"], J2145["EPS2"])
    floor = AlignmentPolicy().binary_fidelity_floor_s
    self_referential = 3.0 * e_max * fid.shapiro_max_abs_s + floor
    assert self_referential < fid.shapiro_max_abs_s  # the old formula rejected
    assert fid.tolerance_shapiro_s > 4.0 * self_referential

    # M2/SINI are untouched pass-through keys (§7.3 minimal delta).
    touched = {k.upper() for k, _ in patch.added_lines} | {
        k.upper() for k in patch.removed_keys
    }
    assert not touched & {"M2", "SINI"}


@slow
def test_binary_fidelity_tolerance_factor_scales_reported_tolerances():
    """AlignmentPolicy.binary_fidelity_tolerance_factor multiplies §7.5 budgets."""
    par = _ell1_dict(a1=J2145["A1"], eps1=J2145["EPS1"], eps2=J2145["EPS2"])
    par["M2"] = ["0.5 0"]
    par["SINI"] = ["0.75 0"]
    decision = _decide(par, policy=AlignmentPolicy(binary_conversion="always"))
    assert decision.outcome == "convert"

    _, record_1 = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(
            binary_conversion="always", binary_fidelity_tolerance_factor=1.0
        ),
        ell1h_shapiro="full",
    )
    _, record_2 = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(
            binary_conversion="always", binary_fidelity_tolerance_factor=2.0
        ),
        ell1h_shapiro="full",
    )
    fid1, fid2 = record_1.fidelity, record_2.fidelity
    assert fid2.tolerance_total_s == pytest.approx(2.0 * fid1.tolerance_total_s)
    assert fid2.tolerance_roemer_s == pytest.approx(2.0 * fid1.tolerance_roemer_s)
    assert fid1.tolerance_shapiro_s is not None
    assert fid2.tolerance_shapiro_s is not None
    assert fid2.tolerance_shapiro_s == pytest.approx(2.0 * fid1.tolerance_shapiro_s)
    # Measured residuals are unchanged; only the budget scales.
    assert fid2.total_max_abs_s == pytest.approx(fid1.total_max_abs_s)
    assert fid2.shapiro_max_abs_s == pytest.approx(fid1.shapiro_max_abs_s)


def _t2_kepler_dict(*, h3=None, free=False):
    flag = "1" if free else "0"
    d = {
        "PSR": ["J2145-0750"],
        "RAJ": ["21:45:50.4"],
        "DECJ": ["-07:50:18"],
        "F0": ["62.295"],
        "PEPOCH": ["55000"],
        "DM": ["9.0"],
        "BINARY": ["T2"],
        "PB": [f"{J2145['PB']} {flag}"],
        "A1": [f"{J2145['A1']} {flag}"],
        "T0": ["55000.0 " + flag],
        "ECC": [f"1.9e-5 {flag}"],
        "OM": [f"160.0 {flag}"],
    }
    if h3 is not None:
        d["H3"] = [f"{h3} {flag}"]
    return d


def test_t2_kepler_is_never_ell1_family():
    """§5.5: the gate never fires on T2-Kepler, with or without H-terms.

    Regression: classifying `T2` + `H3` as ELL1H on a par with no EPS made a
    legitimate T2-Kepler stack raise ell1h_h3_only_underdetermined by default,
    and would have fed the orthometric map EPS defaults of zero.
    """
    plain = _decide(_t2_kepler_dict())
    assert (plain.outcome, plain.reason) == ("skip", "not_ell1_family")
    assert plain.source_family is None

    with_h3 = _decide(_t2_kepler_dict(h3=J2145["H3"]))
    assert (with_h3.outcome, with_h3.reason) == ("skip", "not_ell1_family")
    assert with_h3.source_family is None

    # Even the sampling route must not resurrect it.
    sampled = _decide(
        _t2_kepler_dict(h3=J2145["H3"]),
        policy=AlignmentPolicy(
            binary_conversion="always",
            h3_only="sample_stigma",
            stigma_central=0.37,
            stigma_provenance="test",
        ),
    )
    assert sampled.outcome == "skip"

    # T2 *with* EPS carrying H-terms is still ELL1H (Tempo2 semantics).
    t2_eps = _ell1_dict(binary="T2")
    t2_eps["H3"] = [f"{J2145['H3']} 1"]
    eps_decision = _decide(t2_eps)
    assert eps_decision.source_family == "ELL1H"
    assert eps_decision.reason == "ell1h_h3_only_underdetermined"


@slow
def test_ddh_patch_uses_engine_native_stigma_spelling():
    """DDH orthometric ratio is always written as portable ``STIG``.

    ``readParfile.C`` has no STIGMA/VARSIGMA branch and ``DDHmodel.C`` aborts
    with "both h3 and stig must be set", so a STIGMA line makes any tempo2 load
    of the published par exit. PINT accepts STIG as an alias, so both packages
    get the same portable spelling.
    """
    par = _ell1_dict(binary="ELL1H")
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    patch, _ = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )

    pint_dict = copy.deepcopy(par)
    t2_dict = copy.deepcopy(par)
    apply_binary_patch(pint_dict, patch, timing_package="pint")
    apply_binary_patch(t2_dict, patch, timing_package="tempo2")
    assert "STIG" in pint_dict and "STIGMA" not in pint_dict
    assert "STIG" in t2_dict and "STIGMA" not in t2_dict
    # libstempo is spelled tempo2 for this purpose.
    lst_dict = copy.deepcopy(par)
    apply_binary_patch(lst_dict, patch, timing_package="libstempo")
    assert "STIG" in lst_dict

    assert_postconditions(
        {"PINT": pint_dict, "T2": t2_dict},
        target_family="DDH",
        pre_nonbinary={
            "PINT": _nonbinary_snapshot(par),
            "T2": _nonbinary_snapshot(par),
        },
    )
    assert create_pint_model(pint_dict).STIGMA.value == pytest.approx(
        create_pint_model(t2_dict).STIGMA.value
    )


@slow
def test_ddh_patch_propagates_uncertainties_through_the_map():
    """§7.6: target errors come from the full map, not a source passthrough.

    In the absorbed gauge H3/STIGMA feed A1/ECC/OM/T0, so a converted DDH par
    whose ECC/OM/T0 carry no error column (or A1 carrying the *source* error)
    would misreport the conversion.
    """
    par = _ell1_dict(binary="ELL1H")
    par["H3"] = [f"{J2145['H3']} 1 1e-9"]
    par["STIGMA"] = ["0.5 1 0.02"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    patch, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )
    assert record.uncertainty_propagation == "diagonal_jacobian"

    lines = {k.upper(): v.split() for k, v in patch.added_lines if k.upper() != "C"}
    for name in ("A1", "ECC", "OM", "T0"):
        assert len(lines[name]) == 3, f"{name} has no propagated uncertainty"
        assert float(lines[name][2]) > 0.0

    # A1's error is NOT the source A1 error: the STIGMA/H3 partials dominate.
    source_a1_sigma = float(par["A1"][0].split()[2])
    assert float(lines["A1"][2]) > 10.0 * source_a1_sigma

    # Zeroing the STIGMA uncertainty must shrink A1's error (partials are real).
    par_no_stig_err = copy.deepcopy(par)
    par_no_stig_err["STIGMA"] = ["0.5 1"]
    patch2, _ = convert_shared_binary(
        par_no_stig_err,
        _decide(par_no_stig_err, policy=policy),
        pta_names=("PINT", "T2"),
        policy=policy,
        ell1h_shapiro="absorbed",
    )
    lines2 = {k.upper(): v.split() for k, v in patch2.added_lines if k.upper() != "C"}
    assert float(lines2["A1"][2]) < float(lines["A1"][2])


@slow
def test_factory_exposes_conversion_metadata_end_to_end(tmp_path):
    """The §8.5a channel must survive the whole factory path, not just the unit.

    ``metadata_from_report`` was covered directly and the factory's report
    propagation separately, but nothing checked that a MetaPulsar built by
    ``create_metapulsar`` actually answers ``conversion_metadata()`` — which is
    the exact call nltiming's ``for_pulsar`` makes.
    """
    from metapulsar import create_metapulsar

    par = COMMON_HEAD + (
        "BINARY ELL1H\n"
        f"PB {J2145['PB']} 1\n"
        f"A1 {J2145['A1']} 1\n"
        f"TASC {J2145['TASC']} 1\n"
        f"EPS1 {J2145['EPS1']} 1\n"
        f"EPS2 {J2145['EPS2']} 1\n"
        f"H3 {J2145['H3']} 1\n"
    )
    tim = "FORMAT 1\nfake 1400.0 55000.0000000 1.0 @\nfake 1400.0 55100.0000000 1.0 @\n"
    files = {}
    for pta, package in (("A", "pint"), ("B", "tempo2")):
        par_path = tmp_path / f"{pta}.par"
        tim_path = tmp_path / f"{pta}.tim"
        par_path.write_text(par, encoding="utf-8")
        tim_path.write_text(tim, encoding="utf-8")
        files[pta] = [
            {
                "par": str(par_path),
                "tim": str(tim_path),
                "par_content": par,
                "timing_package": package,
            }
        ]

    mp = create_metapulsar(
        files,
        combination_strategy="shared",
        combine_components=["binary"],
        add_dm_derivatives=False,
        use_pulse_numbers="no",
        alignment_policy=AlignmentPolicy(
            ne_sw=0.0,
            h3_only="sample_stigma",
            stigma_central=0.37,
            stigma_provenance="mass-function closure, m_p=1.4",
        ),
    )
    meta = mp.conversion_metadata()
    assert meta is not None
    assert meta.target_family == "DDH"
    assert meta.gauge == "absorbed"
    assert meta.required_sampling == ("STIGMA",)
    assert meta.stigma_central == pytest.approx(0.37)
    assert meta.stigma_provenance == "mass-function closure, m_p=1.4"

    # And nltiming's probe accepts exactly this object.
    from nltiming.nonlinear_timing_model import _probe_conversion_metadata

    assert _probe_conversion_metadata(mp) is meta or (
        _probe_conversion_metadata(mp).required_sampling == ("STIGMA",)
    )


# ---------------------------------------------------------------------------
# AEI-DR3 survey regressions (`bugs_2026-08-06_todo.md` B3 / B5)
# ---------------------------------------------------------------------------


def test_ddh_splice_honors_the_source_a1dot_spelling():
    """B3: a source spelling ``XDOT`` must not gain a second ``A1DOT`` line.

    ``source_model.as_parfile()`` re-emits the source spelling (PINT
    ``use_alias``), so an alias-blind lookup in the DDH splice appends a rival
    canonical line and PINT refuses to reload the result. MPTA DR2 J1435-6100
    and J1525-5545 are the real cases.
    """
    par = _ell1_dict(
        binary="ELL1H",
        extra={
            "H3": ["2.75e-6 1 5.7e-8"],
            "STIG": ["0.87 1 0.01"],
            "XDOT": ["1.2163770006801874e-14 1 8.46e-15"],
        },
    )
    decision = _decide(par)
    assert decision.outcome == "convert", decision.reason

    patch, record = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(),
        ell1h_shapiro="absorbed",
    )
    applied = copy.deepcopy(par)
    apply_binary_patch(applied, patch, timing_package="pint")

    a1dot_keys = [k for k in applied if k.upper() in ("A1DOT", "XDOT")]
    assert len(a1dot_keys) == 1, a1dot_keys
    # The result must survive the round trip that used to fail.
    create_pint_model(_par_text_from_dict(applied))
    assert record is not None


def test_bare_ell1h_label_is_a_plain_ell1_par():
    """B5: ``BINARY ELL1H`` with no amplitude is ELL1 with zero Shapiro.

    NANOGrav 15y J1802-2124 ships ``BINARY ELL1H`` + ``NHARMS 7`` and nothing
    else. Both engines deliver zero Shapiro delay from that, so the plain DD
    target is exact; the orthometric gate must not claim it.
    """
    par = _ell1_dict(binary="ELL1H", extra={"NHARMS": ["7"]})

    decision = _decide(par)

    assert decision.source_family == "ELL1"
    assert decision.outcome == "convert", decision.reason
    assert decision.target_family == "DD"


@pytest.mark.parametrize("h3", [None, "0.0", "0"])
def test_bare_ell1h_label_variants(h3):
    """Absent or explicitly-zero H3, with or without the harmonic count."""
    extra = {"NHARMS": ["7"]}
    if h3 is not None:
        extra["H3"] = [h3]
    par = _ell1_dict(binary="ELL1H", extra=extra)

    assert _decide(par).source_family == "ELL1"


def test_live_h3_still_routes_to_the_orthometric_gate():
    """The B5 relaxation must not swallow a real H3-only par (Case D)."""
    par = _ell1_dict(binary="ELL1H", extra={"H3": ["1.8e-7 1 5.4e-8"]})

    decision = _decide(par)

    assert decision.source_family == "ELL1H"
    assert decision.outcome == "unsupported"
    assert decision.reason == "ell1h_h3_only_underdetermined"


def test_nharms_alone_does_not_make_a_plain_ell1_orthometric():
    """A stray harmonic count on ``BINARY ELL1`` is inert, not an amplitude."""
    par = _ell1_dict(binary="ELL1", extra={"NHARMS": ["7"]})

    decision = _decide(par)

    assert decision.source_family == "ELL1"
    assert decision.outcome == "convert", decision.reason


def test_bare_ell1h_conversion_drops_the_inert_markers():
    """DD output must not keep ``NHARMS``/zero ``H3`` (§8.4 forbids them)."""
    par = _ell1_dict(binary="ELL1H", extra={"NHARMS": ["7"], "H3": ["0.0"]})
    decision = _decide(par)
    assert decision.outcome == "convert", decision.reason

    patch, _ = convert_shared_binary(
        par,
        decision,
        pta_names=("PINT", "T2"),
        policy=AlignmentPolicy(),
        ell1h_shapiro="absorbed",
    )
    dicts = {"PINT": copy.deepcopy(par), "T2": copy.deepcopy(par)}
    pre = {pta: _nonbinary_snapshot(d) for pta, d in dicts.items()}
    for pkg, d in zip(("pint", "tempo2"), dicts.values()):
        apply_binary_patch(d, patch, timing_package=pkg)

    for d in dicts.values():
        assert not [k for k in d if k.upper() in ("NHARM", "NHARMS", "H3")]
    assert_postconditions(dicts, target_family="DD", pre_nonbinary=pre)


# ---------------------------------------------------------------------------
# Par-unit boundary regressions (`feature_par_units.md` §8, tests 9-15)
# ---------------------------------------------------------------------------

_J2317_PAR = Path("data/aei-dr2/nanograv_9y/par/J2317+1439.par")


@pytest.mark.real_data
@pytest.mark.skipif(not _J2317_PAR.exists(), reason="AEI-DR2 NG 9y data not present")
def test_j2317_gate_reads_si_and_skips():
    """NG J2317+1439: SI gate reads end the 5.9e27-fold false conversion.

    The raw-token gate read `EPS1DOT 0.005038` as 5.038e-3 /s and produced
    scale_s = 3.9e17 s, forcing a conversion PINT then refused
    (`Eccentricity should be in the range of [0,1)`). Read in SI the scale is
    0.066 ns, below the 1 ns threshold.
    """
    from pint.models.model_builder import parse_parfile

    par = parse_parfile(StringIO(_J2317_PAR.read_text(encoding="utf-8")))
    decision = _decide(par)
    assert decision.outcome == "skip"
    assert decision.reason == "below_threshold"
    assert decision.scale is not None
    assert decision.scale.scale_s == pytest.approx(6.569e-11, rel=1e-3)
    assert decision.scale.e_max < 1e-6
    assert decision.scale.a1_max_lt_s == pytest.approx(2.313949, rel=1e-6)


def test_gate_unit_sanity_fires_on_prescaled_dots():
    """EPS dots big enough to imply e > 1 are a unit error, not a pulsar."""
    par = _ell1_dict(
        a1=10.0,
        eps1=1.4e-5,
        eps2=0.0,
        start=54000.0,
        finish=56000.0,
        extra={"EPS1DOT": ["70000 1"], "EPS2DOT": ["0 0"]},
    )
    with pytest.raises(BinaryConversionError, match="gate_unit_sanity"):
        _decide(par)


def test_gate_unit_sanity_bounds_a1():
    par = _ell1_dict(
        a1=10.0,
        eps1=1.4e-5,
        eps2=0.0,
        start=54000.0,
        finish=56000.0,
        extra={"XDOT": ["2e8 1"]},
    )
    with pytest.raises(BinaryConversionError, match="gate_unit_sanity"):
        _decide(par)


@slow
def test_audit_dispatch_passes_non_canonical_keys():
    """C5 dispatch: SI compare for declared axes, token compare elsewhere.

    A par carrying non-binary passthrough keys (DM1, PX) and canonical-unit
    Shapiro terms (M2/SINI) must survive C5 unchanged — proving
    ``has_canonical_unit`` routes each key to the right comparison instead of
    raising ParUnitError on keys outside ``CANONICAL_SI``.
    """
    par = _ell1_dict(a1=10.0, eps1=7e-6, eps2=-1.8e-5)
    par["M2"] = ["0.25 0"]
    par["SINI"] = ["0.9 0"]
    par["PX"] = ["1.2 0"]
    par["DM1"] = ["1e-3 0"]
    policy = AlignmentPolicy(binary_conversion="always")
    decision = _decide(par, policy=policy)
    assert decision.outcome == "convert"
    patch, _ = convert_shared_binary(
        par, decision, pta_names=("PINT", "T2"), policy=policy, ell1h_shapiro="full"
    )
    serialized = copy.deepcopy(par)
    apply_binary_patch(serialized, patch)
    assert serialized["M2"] == ["0.25 0"]
    assert serialized["PX"] == ["1.2 0"]
    assert serialized["DM1"] == ["1e-3 0"]


@slow
def test_ddh_edot_map_analytic_reference():
    """DDH conversion with EPS1DOT reproduces the analytic EDOT/OMDOT.

    Case A (full gauge), xdot = 0: the intrinsic dots equal the printed dots,
    so edot = eps1*eps1dot/ecc and omdot = eps1dot*eps2/ecc^2 exactly. This is
    the test the row-A/row-B mix in the `_ddh_map` inputs lacked: with the
    EPS dot read as a raw token the converted EDOT would be off by 1e12.
    """
    from metapulsar.pint_helpers import si_from_model as _si

    par = _ell1_dict(
        a1=J2145["A1"],
        eps1=J2145["EPS1"],
        eps2=J2145["EPS2"],
        binary="ELL1H",
    )
    par["H3"] = [f"{J2145['H3']} 1 1e-10"]
    par["STIGMA"] = ["0.5 1 0.01"]
    par["EPS1DOT"] = ["0.005038 0"]  # scaled spelling -> 5.038e-15 /s
    source = create_pint_model(par, ell1h_shapiro="full")
    converted, _unc = _convert_ell1h_block(
        source, "full", AlignmentPolicy(binary_conversion="always")
    )

    e1, e2, e1dot = J2145["EPS1"], J2145["EPS2"], 5.038e-15
    ecc = math.hypot(e1, e2)
    expected_edot = e1 * e1dot / ecc
    expected_omdot = e1dot * e2 / ecc**2  # rad/s

    assert _si(converted, "EDOT") == pytest.approx(expected_edot, rel=1e-9)
    assert _si(converted, "OMDOT") == pytest.approx(expected_omdot, rel=1e-9)


# ---------------------------------------------------------------------------
# Cross-package emission portability (§8 tests 14-15)
# ---------------------------------------------------------------------------

_T2_HEAD = """PSRJ J2145-0750
RAJ 21:45:50.4
DECJ -07:50:18
F0 62.295
PEPOCH 55000
POSEPOCH 55000
DMEPOCH 55000
DM 9.0
EPHEM DE421
CLK TT(TAI)
UNITS TDB
BINARY ELL1
PB 6.83890261
A1 10.1641056
TASC 55000.0
EPS1 7e-6
EPS2 -1.8e-5
"""

_T2_TIM = """FORMAT 1
fake 1440.0 54990.0 1.0 pks
fake 1440.0 55000.0 1.0 pks
fake 1440.0 55010.0 1.0 pks
"""


def _load_both_engines(tmp_path, par_text):
    from metapulsar.sandbox_tempo2 import tempopulsar

    par = tmp_path / "cross.par"
    tim = tmp_path / "cross.tim"
    par.write_text(par_text, encoding="utf-8")
    tim.write_text(_T2_TIM, encoding="utf-8")
    pint_model = create_pint_model(par_text)
    t2_psr = tempopulsar(parfile=str(par), timfile=str(tim), dofit=False)
    return pint_model, t2_psr


@pytest.mark.requires_libstempo
def test_eps_dot_emission_portable_across_engines(tmp_path):
    """A `token_from_si` EPS dot reads back identically in PINT and tempo2."""
    from metapulsar.pint_helpers import si_from_model as _si
    from metapulsar.pint_helpers import token_from_si as _tok

    value_si = 5.038e-15  # 1/s
    par_text = _T2_HEAD + f"EPS1DOT {_tok('EPS1DOT', value_si)} 0\n"
    pint_model, t2_psr = _load_both_engines(tmp_path, par_text)

    pint_si = _si(pint_model, "EPS1DOT")
    t2_si = float(t2_psr["EPS1DOT"].val)  # tempo2 stores 1/s after rescale
    assert pint_si == pytest.approx(value_si, rel=1e-12)
    assert t2_si == pytest.approx(value_si, rel=1e-9)
    assert pint_si == pytest.approx(t2_si, rel=1e-9)


@pytest.mark.requires_libstempo
@pytest.mark.xfail(
    strict=False,
    reason="tempo2 rescales val[0] but never err[0] (readParfile.C), so a "
    "scaled-spelling sigma diverges between engines; upstream bug, see "
    "feature_par_units.md §6",
)
def test_row_b_uncertainty_spelling_across_engines(tmp_path):
    """Scaled-spelling XDOT sigma: PINT rescales it, tempo2 appears not to."""
    par_text = _T2_HEAD + "XDOT 0.005485 1 0.001182\n"
    pint_model, t2_psr = _load_both_engines(tmp_path, par_text)

    pint_sigma = float(pint_model.A1DOT.uncertainty.to_value(pint_model.A1DOT.units))
    # PINT applies the magnitude heuristic to the error token too.
    assert pint_sigma == pytest.approx(1.182e-15, rel=1e-9)
    key = "A1DOT" if "A1DOT" in t2_psr.pars(which="set") else "XDOT"
    t2_sigma = float(t2_psr[key].err)
    assert t2_sigma == pytest.approx(pint_sigma, rel=1e-6)
