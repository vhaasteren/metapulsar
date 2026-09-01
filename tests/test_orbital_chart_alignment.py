from pathlib import Path

import numpy as np
import pytest

from metapulsar.pint_helpers import (
    OrbitalChartError,
    align_orbital_chart,
    create_pint_model,
    format_longdouble_par_value,
)

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
    exercised without tempo2's TCB->TDB transform.
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
