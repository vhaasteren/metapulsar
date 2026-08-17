from pathlib import Path

import numpy as np
import pytest

from metapulsar import create_metapulsar
from metapulsar.pint_helpers import (
    OrbitalChartError,
    align_orbital_chart,
    create_pint_model,
    format_longdouble_par_value,
)
from tests.helpers import make_tim_metadata

HYBRID = """PSRJ           J2241-5236
BINARY         ELL1
PB             0.14567224091722622131    1  0.00000000001841468870
FB1            1.1506575242964346268e-21 1  1.0823396568139769955e-21
FB2            7.9332763696732937957e-28 1  1.4456394062568747941e-28
F0             457.31006798137
PEPOCH         59000
TASC           59098.944896021848802     1  0.00000001207041607589
A1             0.02579530367559909903    1
EPS1           2.7614888783320161061e-06 1
EPS2           1.7355143975169744114e-06 1
RAJ            22:41:42.01
DECJ           -52:36:36.2
EPHEM          DE421
CLK            TT(BIPM2015)
UNITS          TDB
"""

HYBRID_TABS = (
    "PSRJ\tJ2241-5236\nBINARY\tELL1\n"
    "PB\t0.14567224020163716786\t1\n"
    "FB1\t-6.1045637989274866087e-21\t1\n"
    "F0\t457.31006798137\nPEPOCH\t59000\n"
    "TASC\t59098.944896021848802\t1\nA1\t0.0257953036755991\t1\n"
    "EPS1\t2.76148887833202e-06\t1\nEPS2\t1.73551439751697e-06\t1\n"
    "RAJ\t22:41:42.01\nDECJ\t-52:36:36.2\n"
    "EPHEM\tDE421\nCLK\tTT(BIPM2015)\nUNITS\tTDB\n"
)

NATIVE_FB0 = HYBRID.replace(
    "PB             0.14567224091722622131    1  0.00000000001841468870",
    "FB0            7.945284565678293050190222625056053e-05 1",
)

PB_ONLY = (
    "\n".join(line for line in HYBRID.splitlines() if not line.startswith("FB")) + "\n"
)


def _align(par_text, *, timing_package="tempo2", model_text=None, **kw):
    """Align with a model built from the same text."""
    model = create_pint_model(model_text if model_text is not None else par_text)
    return align_orbital_chart(par_text, model, timing_package=timing_package, **kw)


def _fb0_line(text):
    return next(line for line in text.splitlines() if line.split()[0] == "FB0")


def test_hybrid_is_aligned():
    out, changed = _align(HYBRID)
    assert changed is True
    assert not any(line.split()[0] == "PB" for line in out.splitlines())
    tokens = _fb0_line(out).split()
    pb = np.longdouble("0.14567224091722622131")
    assert np.longdouble(tokens[1]) == 1 / (np.longdouble(86400) * pb)
    assert tokens[2] == "1"  # fit flag verbatim
    sigma_pb = np.longdouble("0.00000000001841468870")
    assert np.longdouble(tokens[3]) == sigma_pb / (np.longdouble(86400) * pb * pb)


def test_tab_separated_par_is_aligned():
    """Published par files use tabs; a space-anchored pattern silently misses."""
    out, changed = _align(HYBRID_TABS)
    assert changed is True
    assert len(_fb0_line(out).split()) == 3  # no uncertainty column to carry


def test_alignment_is_idempotent():
    once, first = _align(HYBRID)
    twice, second = _align(once)
    assert first is True and second is False
    assert twice == once


def test_alignment_touches_exactly_one_line():
    """Relabel discipline: UNITS, BINARY, TASC, EPS*, FBn all untouched."""
    before, after = HYBRID.splitlines(), _align(HYBRID)[0].splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert before[differing[0]].split()[0] == "PB"
    assert after[differing[0]].split()[0] == "FB0"


@pytest.mark.parametrize("text", [NATIVE_FB0, PB_ONLY])
@pytest.mark.parametrize("package", ["tempo2", "pint"])
def test_par_already_agreeing_with_its_model_is_returned_unchanged(text, package):
    out, changed = _align(text, timing_package=package)
    assert changed is False
    assert out is text


def test_alignment_applies_to_pint_backed_pars_too():
    """Invariant 3: align every par. A hybrid PINT-backed reference would
    otherwise reintroduce PB as the shared binary chart."""
    out, changed = _align(HYBRID, timing_package="pint")
    assert changed is True
    assert _fb0_line(out).split()[0] == "FB0"


@pytest.mark.parametrize("model_name", ["ELL1H", "DD", "T2"])
def test_tempo2_non_fb_capable_binary_model_raises(model_name):
    text = HYBRID.replace("BINARY         ELL1", f"BINARY         {model_name}")
    with pytest.raises(OrbitalChartError, match="does not evaluate FB"):
        _align(text, timing_package="tempo2", model_text=HYBRID)


@pytest.mark.parametrize("model_name", ["ELL1H", "DD"])
def test_pint_backed_non_fb_capable_model_is_allowed(model_name):
    """PINT evaluates FBX for any binary model; the guard is tempo2-specific."""
    text = HYBRID.replace("BINARY         ELL1", f"BINARY         {model_name}")
    assert _align(text, timing_package="pint", model_text=HYBRID)[1] is True


@pytest.mark.parametrize(
    "extra, match",
    [
        ("BINARY         BTX\n", "duplicate BINARY"),
        ("PB             0.2   1\n", "duplicate PB"),
        ("FB1            2.0e-21   1\n", "duplicate FB1"),
    ],
)
def test_duplicate_entries_raise(extra, match):
    with pytest.raises(OrbitalChartError, match=match):
        _align(HYBRID + extra, model_text=HYBRID)


def test_free_fb0_with_neither_fb0_nor_pb_in_par_raises():
    text = (
        "\n".join(line for line in HYBRID.splitlines() if not line.startswith("PB "))
        + "\n"
    )
    with pytest.raises(OrbitalChartError, match="neither FB0 nor PB"):
        _align(text, model_text=HYBRID)


def test_fbj_is_not_an_fb_coefficient():
    assert _align(PB_ONLY + "FBJ            0.5   1\n", model_text=PB_ONLY)[1] is False


def test_formatter_round_trips_computed_values_exactly():
    """Parsed tokens round-trip under almost any formatter; the computed
    reciprocal and propagated sigma are the hard cases (acceptance criterion 6)."""
    pb = np.longdouble("0.14567224091722622131")
    for value in (
        1 / (np.longdouble(86400) * pb),
        np.longdouble("0.00000000001841468870") / (np.longdouble(86400) * pb * pb),
    ):
        assert np.longdouble(format_longdouble_par_value(value)) == value


def test_formatter_is_exact_over_random_longdoubles():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x = np.longdouble(rng.uniform(-1, 1)) * np.longdouble(10.0) ** int(
            rng.integers(-40, 40)
        )
        if not np.isfinite(x) or x == 0:
            continue
        x = x * (1 + np.longdouble(1) / np.longdouble(3))
        assert np.longdouble(format_longdouble_par_value(x)) == x


@pytest.mark.parametrize(
    "rejected",
    [
        lambda x: format(x, ".20g"),
        lambda x: f"{x:.20g}",
        lambda x: np.format_float_scientific(x, precision=19, unique=False),
        lambda x: np.format_float_scientific(
            x, precision=np.finfo(np.longdouble).precision, unique=False
        ),
    ],
)
def test_rejected_formatters_are_inexact(rejected):
    """NOTE: the assertion is ``any``, not ``all``, and must stay that way. On a
    quad long double ``precision=finfo.precision`` is exact for FB0 and inexact
    only for sigma_FB0; on 80-bit x86-64 the failure set differs. Do not tighten
    without re-measuring both widths."""
    pb = np.longdouble("0.14567224091722622131")
    values = [
        1 / (np.longdouble(86400) * pb),
        np.longdouble("0.00000000001841468870") / (np.longdouble(86400) * pb * pb),
    ]
    assert any(np.longdouble(rejected(v)) != v for v in values)
    assert all(np.longdouble(format_longdouble_par_value(v)) == v for v in values)


@pytest.mark.requires_ipta_data
def test_written_value_comes_from_the_par_not_the_model():
    """Pins invariant 2: the TCB/TDB trap.

    create_pint_model passes allow_tcb=True, so a TCB par yields a TDB model.
    Copying model.FB0 into the still-TCB par would be a 1.55e-8 relative error
    in orbital frequency -- about 97 us of residual on this pulsar.
    """
    import astropy.units as u

    par = Path("data-check/PPTA_DR3/J2241-5236.par")  # UNITS TCB
    if not par.exists():
        pytest.skip("PPTA DR3 J2241 not present")
    text = par.read_text()
    assert any(line.split()[:2] == ["UNITS", "TCB"] for line in text.splitlines())

    model = create_pint_model(text)
    out, changed = align_orbital_chart(
        text, model, timing_package="tempo2", pta_name="PPTA"
    )
    assert changed is True

    written = np.longdouble(_fb0_line(out).split()[1])
    pb_tcb = np.longdouble("0.14567224091722622131")
    assert written == 1 / (np.longdouble(86400) * pb_tcb)

    model_fb0 = np.longdouble(model.FB0.quantity.to_value(1 / u.s))
    assert written != model_fb0
    assert abs(written - model_fb0) / written == pytest.approx(1.55e-8, rel=0.05)


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_tempo2_residuals_unchanged_by_alignment(tmp_path):
    """The rewrite is a coordinate relabel, not a model change.

    PURE RELABEL CHECK. No unit conversion anywhere: both pars are the source
    par, still ``UNITS TCB``, differing only in the PB/FB0 line. A TCB->TDB
    conversion legitimately changes parameter values and would invalidate the
    measurement -- do not add one to this test.
    """
    from metapulsar.sandbox_tempo2 import tempopulsar

    par = Path("data-check/PPTA_DR3/J2241-5236.par")
    tim = Path("data-check/PPTA_DR3/J2241-5236.tim")
    if not par.exists():
        pytest.skip("PPTA DR3 J2241 not present")

    text = par.read_text()
    aligned, changed = align_orbital_chart(
        text, create_pint_model(text), timing_package="tempo2", pta_name="PPTA"
    )
    assert changed is True
    assert "UNITS          TCB" in aligned
    aligned_par = tmp_path / "J2241-5236_aligned.par"
    aligned_par.write_text(aligned, encoding="utf-8")

    before = tempopulsar(parfile=str(par), timfile=str(tim), dofit=False)
    after = tempopulsar(parfile=str(aligned_par), timfile=str(tim), dofit=False)

    assert "PB" in before.pars()
    assert "FB0" in after.pars() and "PB" not in after.pars()
    delta = np.abs(np.asarray(after.residuals()) - np.asarray(before.residuals()))
    assert delta.max() < 1e-9, f"max residual change {delta.max():.3e} s"


def _pm(**pta_texts):
    from metapulsar.parameter_manager import ParameterManager

    return ParameterManager(
        file_data={
            pta: {"par": None, "par_content": text, "timing_package": "tempo2"}
            for pta, text in pta_texts.items()
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
    )


def test_release_par_content_is_never_mutated():
    """Invariant 1. The single most important thing to keep true."""
    pm = _pm(PPTA=HYBRID)
    aligned = pm._aligned_parfile_contents()
    assert aligned["PPTA"][1] is True
    assert pm.file_data["PPTA"]["par_content"] == HYBRID
    assert pm._get_parfile_content("PPTA") == HYBRID


def test_engine_parfiles_writes_only_what_changed(tmp_path):
    """Invariant 5."""
    from metapulsar.parameter_manager import ParameterManager

    hybrid_par = tmp_path / "hybrid.par"
    hybrid_par.write_text(HYBRID)
    native_par = tmp_path / "native.par"
    native_par.write_text(NATIVE_FB0)

    pm = ParameterManager(
        file_data={
            "PPTA": {
                "par": hybrid_par,
                "par_content": HYBRID,
                "timing_package": "tempo2",
            },
            "MPTA": {
                "par": native_par,
                "par_content": NATIVE_FB0,
                "timing_package": "tempo2",
            },
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
        output_dir=tmp_path / "out",
    )
    paths = pm.engine_parfiles()
    assert paths["MPTA"] == native_par  # untouched, original path
    assert paths["PPTA"] != hybrid_par  # rewritten
    assert "FB0" in paths["PPTA"].read_text()
    assert hybrid_par.read_text() == HYBRID  # source file untouched


@pytest.mark.slow
def test_shared_merge_is_reference_order_independent():
    """The latent ordering bug: MPTA+PPTA built, PPTA+MPTA raised.

    Alignment runs before _make_parameters_shared, so the reference PTA is
    canonical whichever one it is. Uses UNITS TDB synthetics so the merge is
    exercised without tempo2's TCB->TDB transform; real TCB end-to-end coverage
    is covered below.
    """
    for order in (("MPTA", "PPTA"), ("PPTA", "MPTA")):
        texts = {"MPTA": NATIVE_FB0, "PPTA": HYBRID}
        pm = _pm(**{p: texts[p] for p in order})
        written = pm.make_parfiles_shared()
        for pta, path in written.items():
            body = Path(path).read_text()
            assert "FB0" in body, (order, pta)
            assert not any(line.split()[0] == "PB" for line in body.splitlines()), (
                order,
                pta,
            )


def test_shared_merge_aligns_hybrid_pint_reference():
    """Invariant 3 / poison-pill case: hybrid reference with timing_package=pint.

    If alignment were tempo2-only, PPTA would keep PB, merge would key the
    shared binary chart as PB (or drop FB0 from non-ref legs), and a tempo2
    MPTA leg that needed no alignment itself would still break. Align-all
    first makes the written shared pars FB0-chart for every PTA.
    """
    from metapulsar.parameter_manager import ParameterManager

    pm = ParameterManager(
        file_data={
            "PPTA": {
                "par": None,
                "par_content": HYBRID,
                "timing_package": "pint",
            },
            "MPTA": {
                "par": None,
                "par_content": NATIVE_FB0,
                "timing_package": "tempo2",
            },
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
    )
    written = pm.make_parfiles_shared()
    for pta, path in written.items():
        body = Path(path).read_text()
        assert "FB0" in body, pta
        assert not any(line.split()[0] == "PB" for line in body.splitlines()), pta


DATA = Path("data-check")
SRC = {"PPTA": ("PPTA_DR3", "J2241-5236"), "MPTA": ("MPTA_DR2", "J2241-5236")}


def _entry(release, name):
    par, tim = DATA / release / f"{name}.par", DATA / release / f"{name}.tim"
    return {
        "par": par,
        "tim": tim,
        "par_content": par.read_text(),
        "timing_package": "tempo2",
        "tim_metadata": make_tim_metadata(pn_status="none"),
    }


def _require(*paths):
    for p in paths:
        if not p.exists():
            pytest.skip(f"IPTA data not present: {p}")


def _build(file_data, strategy, reference=None):
    kw = dict(
        combination_strategy=strategy,
        use_pulse_numbers="no",
        combine_components=["astrometry", "spindown", "binary", "dispersion"],
    )
    if reference:
        kw["reference_pta"] = reference
    return create_metapulsar(file_data, **kw)


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
@pytest.mark.parametrize(
    "ptas, strategy, reference",
    [
        (["PPTA"], "per_pta", None),
        (["PPTA"], "shared", None),
        (["MPTA", "PPTA"], "per_pta", None),
        (["MPTA", "PPTA"], "shared", "MPTA"),
        (["PPTA", "MPTA"], "shared", "PPTA"),
    ],
)
def test_every_case_from_the_failure_map_now_builds(ptas, strategy, reference):
    """All five ordering cases. Four raised before this feature; the fifth passed
    only because MPTA happened to be the reference."""
    _require(
        *[DATA / SRC[p][0] / f"{SRC[p][1]}.{e}" for p in ptas for e in ("par", "tim")]
    )

    mp = _build({p: [_entry(*SRC[p])] for p in ptas}, strategy, reference)

    for pta in ptas:
        assert "FB0" in mp._pta_data[pta].fitpars
        assert "PB" not in mp._pta_data[pta].fitpars
    meta = next(k for k in mp.fitpars if k == "FB0" or k.startswith("FB0_"))
    assert all(v.par_key == "FB0" for v in mp._fitparameters[meta].values())
    col = mp._designmatrix[:, mp.fitpars.index(meta)]
    assert np.isfinite(col).all() and np.abs(col).sum() > 0.0
    assert np.isfinite(mp._residuals).all()
    rms_us = float(np.sqrt(np.mean(mp._residuals**2)) * 1e6)
    # Tight bound only when each PTA keeps its own solution. Multi-PTA shared
    # forces the reference solution onto the other PTA without a joint refit;
    # measured J2241 MPTA+PPTA shared RMS is ~100-400 us (rev 5.1).
    if len(ptas) == 1 or strategy == "per_pta":
        assert 0.1 < rms_us < 20.0
    else:
        assert 0.1 < rms_us < 1000.0


def _without_mode_aliases(lines):
    """Drop MODE/WEIGHT so MODE normalization does not cascade zip-diffs."""
    return [
        line
        for line in lines
        if not (line.split() and line.split()[0].upper() in ("MODE", "WEIGHT"))
    ]


def _assert_normalized_final_mode(par_text: str, mode: int = 1) -> None:
    lines = [line for line in par_text.splitlines() if line.split()]
    mode_lines = [
        line for line in lines if line.split()[0].upper() in ("MODE", "WEIGHT")
    ]
    assert mode_lines == [f"MODE {mode}"]
    assert lines[-1] == f"MODE {mode}"


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_per_pta_engine_par_differs_only_in_the_pb_line():
    """Invariants 1 and 5: no unit conversion, no sharing; PB→FB0 plus MODE norm."""
    _require(*[DATA / "PPTA_DR3" / f"J2241-5236.{e}" for e in ("par", "tim")])
    par = DATA / "PPTA_DR3" / "J2241-5236.par"
    mp = _build({"PPTA": [_entry(*SRC["PPTA"])]}, "per_pta")

    source = _without_mode_aliases(par.read_text().splitlines())
    consumed_text = mp._parfile_content_for_pta("PPTA")
    consumed = _without_mode_aliases(consumed_text.splitlines())
    assert len(source) == len(consumed)
    differing = [i for i, (a, b) in enumerate(zip(source, consumed)) if a != b]
    assert len(differing) == 1
    assert source[differing[0]].split()[0] == "PB"
    assert consumed[differing[0]].split()[0] == "FB0"
    assert any(line.split()[:2] == ["UNITS", "TCB"] for line in consumed)
    _assert_normalized_final_mode(consumed_text)


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_nonlinear_timing_is_available():
    """The acceptance criterion revision 2.1 could not meet: the mapping is an
    identity rename, so libstempo can set FB0 natively."""
    _require(*[DATA / "PPTA_DR3" / f"J2241-5236.{e}" for e in ("par", "tim")])
    mp = _build({"PPTA": [_entry(*SRC["PPTA"])]}, "shared")

    engines = {"tempo2": "libstempo"}
    assert mp.can_use_engines(engines, linearized=False) is True
    assert mp.timing_engine(engines, linearized=False) is not None


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_j1825_mpta_tempo2_unaffected():
    """No FB terms: orbital content untouched; MODE may be normalized last."""
    par = DATA / "MPTA_DR2" / "J1825-0319.par"
    tim = DATA / "MPTA_DR2" / "J1825-0319.tim"
    _require(par, tim)
    mp = _build({"MPTA": [_entry("MPTA_DR2", "J1825-0319")]}, "per_pta")

    assert mp.name
    assert np.isfinite(mp._residuals).all()
    assert not any(p == "FB0" or p.startswith("FB0_") for p in mp.fitpars)
    consumed = mp._parfile_content_for_pta("MPTA")
    assert _without_mode_aliases(consumed.splitlines()) == _without_mode_aliases(
        par.read_text().splitlines()
    )
    _assert_normalized_final_mode(consumed)


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_direct_construction_with_hybrid_chart_fails_loudly():
    """MetaPulsar(...) bypasses ParameterManager's par production, so the guard must
    raise a clear error rather than a bare list.index failure."""
    from metapulsar.metapulsar import MetaPulsar
    from metapulsar.sandbox_tempo2 import tempopulsar

    par = DATA / "PPTA_DR3" / "J2241-5236.par"
    tim = DATA / "PPTA_DR3" / "J2241-5236.tim"
    _require(par, tim)
    t2_psr = tempopulsar(parfile=str(par), timfile=str(tim), dofit=False)
    assert "PB" in t2_psr.pars()

    with pytest.raises(ValueError, match=r"hybrid PB\+FBn orbital chart"):
        MetaPulsar(
            {"PPTA": t2_psr},
            combination_strategy="per_pta",
            pta_files={
                "PPTA": {
                    "par_path": par,
                    "tim_path": tim,
                    "timing_package": "tempo2",
                }
            },
        )
