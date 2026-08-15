"""Unit tests for the combination par/tim writer."""

from __future__ import annotations

import re

import pytest

from metapulsar.combination_writer import (
    _first_data_toa_tokens,
    _infer_pulse_numbers,
    _is_noise_line,
    _modal_offset,
    _read_pn_sequence,
    _rewrite_tim_pn_sequential,
    align_combination_tzr,
    extract_fd_terms,
    fortran_d_to_e,
    renumber_combination_pulse_numbers,
    sanitize_fortran_exponents,
    write_combination_par,
    write_combination_tim,
)

PTA_A_PAR = """\
PSR J1234+5678
RAJ 12:34:56
DECJ +56:78:90
F0 100.0
DM 10.0
EPHEM DE440
CLK TT(BIPM2019)
UNITS TDB
TIMEEPH FB90
T2CMETHOD IAU2000B
JUMP -fe L-wide -0.000009449 1 0.000009439
FD1 1.2D-03 1 3.4D-04
FD2 -1.8D-04 1 4.1D-05
EFAC -f 1400 1.1
TNREDAMP -13.0
TRACK -2
"""

PTA_B_PAR = """\
PSR J1234+5678
RAJ 12:34:56
DECJ +56:78:90
F0 100.0
DM 10.0
EPHEM DE440
CLK TT(BIPM2019)
UNITS TDB
JUMP -sys system_b 0 1
FD1 2.0E-03 1
EQUAD -f 1400 1.0e-6
"""

MINIMAL_COMBO_PAR = """\
PSR J1909-3744
RAJ 19:09:47.4280
DECJ -37:44:14.326
F0 339.315686
F1 -1.61e-15
PEPOCH 55000
POSEPOCH 55000
DMEPOCH 55000
DM 10.39
EPHEM DE440
CLK UTC(NIST)
UNITS TDB
TRACK -2
"""


@pytest.mark.unit
def test_fortran_d_to_e():
    assert fortran_d_to_e("4.24225627D-05") == "4.24225627E-05"
    assert fortran_d_to_e("-1.55d+04") == "-1.55E+04"
    text = "FDJUMP1 -pta pta_a 1.2D-03 1 3.4D-04\n"
    sanitized = sanitize_fortran_exponents(text)
    assert "1.2D-03" not in sanitized and "3.4D-04" not in sanitized
    assert "1.2E-03" in sanitized and "3.4E-04" in sanitized


@pytest.mark.unit
def test_extract_fd_terms_retains_repeated_indices_in_document_order():
    text = "FD1 1.0 1\nFD2 2.0 1\nFD1 3.0 1 1e-5\n"
    terms = extract_fd_terms(text)
    assert [(idx, value) for idx, value, _ in terms] == [
        (1, "1.0"),
        (2, "2.0"),
        (1, "3.0"),
    ]
    assert terms[2][2] == "1e-5"


@pytest.mark.unit
def test_write_combination_par_shape(tmp_path):
    stats = write_combination_par(
        reference_pta="pta_a",
        pta_par_texts={
            "pta_a": PTA_A_PAR,
            "pta_b": PTA_B_PAR,
        },
        out_path=tmp_path / "X.par",
    )
    body = (tmp_path / "X.par").read_text()
    assert body.startswith("# Created:")
    assert "# By:      MetaPulsar" in body
    assert "# Product: combination" in body
    assert "# reference_pta: pta_a" in body
    assert "FD1 " not in body
    assert "FDJUMP1 -pta pta_a 1.2E-03 1 3.4E-04" in body
    assert "JUMP -pta pta_b 0.0 1" in body
    assert "FDJUMPDM -pta pta_b 0.0 1" in body
    assert "FDJUMPLOG Y" in body and "FDJUMP_SCALE LOG" in body
    assert "JUMP -fe L-wide" in body
    assert "JUMP -sys system_b" in body
    assert not any(_is_noise_line(line) for line in body.splitlines())
    assert re.search(r"^TRACK\s+-2\b", body, re.M)
    assert stats.n_fdjump >= 1 and stats.n_fdjumpdm == 1
    assert "JUMP -pta pta_a" not in body
    assert "FDJUMPDM -pta pta_a" not in body
    assert "FDJUMPDM = DM - DM_ref" in body


@pytest.mark.unit
def test_write_combination_par_drops_source_header_extras(tmp_path):
    """A reference par that already carries a MetaPulsar header contributes none of it.

    ``strip_metapulsar_par_header`` consumes the whole leading ``#`` block, so
    the merged file's provenance is its own -- a strip regression would leak the
    source's ``# Product:`` / ``# alignment_policy.*`` lines into the body.
    """
    from metapulsar.parfile_header import ensure_metapulsar_par_header

    stamped_reference = ensure_metapulsar_par_header(
        PTA_A_PAR,
        extra={
            "Product": "shared",
            "reference_pta": "somewhere_else",
            "alignment_policy.ephem": "DE421",
        },
    )
    write_combination_par(
        reference_pta="pta_a",
        pta_par_texts={"pta_a": stamped_reference, "pta_b": PTA_B_PAR},
        out_path=tmp_path / "X.par",
    )
    body = (tmp_path / "X.par").read_text()

    assert body.count("# Created:") == 1
    assert "# Product: combination" in body
    assert "# reference_pta: pta_a" in body
    assert "# Product: shared" not in body
    assert "# reference_pta: somewhere_else" not in body
    assert "# alignment_policy.ephem: DE421" not in body
    assert "PSR J1234+5678" in body


@pytest.mark.unit
def test_write_combination_par_fdjumpdm_delta(tmp_path):
    pta_a = PTA_A_PAR.replace("DM 10.0", "DM 10.0")
    pta_b = PTA_B_PAR.replace("DM 10.0", "DM 10.25")
    write_combination_par(
        reference_pta="pta_a",
        pta_par_texts={"pta_a": pta_a, "pta_b": pta_b},
        out_path=tmp_path / "delta.par",
    )
    body = (tmp_path / "delta.par").read_text()
    assert "FDJUMPDM -pta pta_b 0.25 1" in body
    assert "FDJUMPDM -pta pta_a" not in body


@pytest.mark.unit
def test_write_combination_par_missing_dm_errors(tmp_path):
    pta_b = PTA_B_PAR.replace("DM 10.0\n", "")
    with pytest.raises(ValueError, match="required DM missing"):
        write_combination_par(
            reference_pta="pta_a",
            pta_par_texts={"pta_a": PTA_A_PAR, "pta_b": pta_b},
            out_path=tmp_path / "bad.par",
        )


@pytest.mark.unit
def test_write_combination_par_track_pulse_numbers_false(tmp_path):
    write_combination_par(
        reference_pta="pta_a",
        pta_par_texts={"pta_a": PTA_A_PAR, "pta_b": PTA_B_PAR},
        out_path=tmp_path / "notrack.par",
        track_pulse_numbers=False,
    )
    body = (tmp_path / "notrack.par").read_text()
    assert not re.search(r"^TRACK\b", body, re.M)


@pytest.mark.unit
def test_write_combination_par_single_pta_no_pta_jump(tmp_path):
    stats = write_combination_par(
        reference_pta="pta_a",
        pta_par_texts={"pta_a": PTA_A_PAR},
        out_path=tmp_path / "single.par",
    )
    body = (tmp_path / "single.par").read_text()
    assert not re.search(r"^JUMP\s+-pta\b", body, re.M)
    assert not re.search(r"^FDJUMPDM\b", body, re.M)
    assert "FDJUMP1 -pta pta_a 1.2E-03 1 3.4E-04" in body
    assert "FDJUMPLOG Y" in body
    assert stats.n_fdjumpdm == 0
    assert stats.n_jumps == 1  # only the copied JUMP -fe line


@pytest.mark.unit
def test_write_combination_tim_include_tree(tmp_path):
    pulsar = "J1234+5678"
    tim_d = tmp_path / f"{pulsar}.tim.d"
    tim_d.mkdir()
    a = tim_d / "pta_a.tim"
    b = tim_d / "pta_b.tim"
    a.write_text("FORMAT 1\n toa00000 1400.0 55000.0 1.0 g\n", encoding="utf-8")
    b.write_text("FORMAT 1\n toa00000 1400.0 55001.0 1.0 g\n", encoding="utf-8")
    out = tmp_path / f"{pulsar}.tim"
    n = write_combination_tim(
        pulsar=pulsar,
        reference_pta="pta_a",
        pta_tim_paths={"pta_a": a, "pta_b": b},
        out_path=out,
    )
    assert n == 2
    text = out.read_text(encoding="utf-8")
    assert text.startswith("FORMAT 1\n")
    assert f"INCLUDE {pulsar}.tim.d/pta_a.tim" in text
    assert f"INCLUDE {pulsar}.tim.d/pta_b.tim" in text
    # Reference first.
    lines = [ln for ln in text.splitlines() if ln.startswith("INCLUDE")]
    assert lines[0].endswith("pta_a.tim")
    assert lines[1].endswith("pta_b.tim")


@pytest.mark.unit
def test_write_combination_tim_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_combination_tim(
            pulsar="X",
            reference_pta="pta_a",
            pta_tim_paths={"pta_a": tmp_path / "missing.tim"},
            out_path=tmp_path / "X.tim",
        )


@pytest.mark.unit
def test_noise_line_detection():
    assert _is_noise_line("EFAC -f 1400 1.1")
    assert _is_noise_line("TNREDAMP -13.0")
    assert _is_noise_line("RNAMP -14")
    assert not _is_noise_line("TRACK -2")
    assert not _is_noise_line("TIMEEPH FB90")
    assert not _is_noise_line("T2CMETHOD IAU2000B")
    assert not _is_noise_line("RAJ 12:34:56")
    assert not _is_noise_line("# comment")
    assert not _is_noise_line("")


@pytest.mark.unit
def test_align_combination_tzr_inserts_missing_keys(tmp_path):
    par = tmp_path / "x.par"
    par.write_text(MINIMAL_COMBO_PAR, encoding="utf-8")
    align_combination_tzr(
        par,
        tzrmjd="54510.123456789012",
        tzrfrq="1400.0",
        tzrsite="g",
    )
    body = par.read_text(encoding="utf-8")
    assert re.search(r"^TZRMJD 54510\.123456789012\s*$", body, re.M)
    assert re.search(r"^TZRFRQ 1400\.0\s*$", body, re.M)
    assert re.search(r"^TZRSITE g\s*$", body, re.M)


@pytest.mark.unit
def test_modal_offset_unimodal_with_outliers():
    result = _modal_offset([5, 5, 5, 6, 5])
    assert result.offset == 5
    assert result.n_toas == 5
    assert result.mode_fraction == 0.8
    assert result.max_deviation == 1


@pytest.mark.unit
def test_modal_offset_ties_break_toward_first_seen():
    # Equal counts: first-seen key wins (not min / not last).
    assert _modal_offset([5, 5, 6, 6]).offset == 5
    assert _modal_offset([6, 6, 5, 5]).offset == 6
    result = _modal_offset([5, 5, 6, 6])
    assert result.mode_fraction == 0.5
    assert result.max_deviation == 1


def _write_legs(tmp_path, ref_mjds, other_mjds):
    """Write a two-leg combination tree with placeholder ``-pn 0``."""
    pulsar = "J1909-3744"
    tim_d = tmp_path / f"{pulsar}.tim.d"
    tim_d.mkdir()
    ref = tim_d / "nanograv.tim"
    other = tim_d / "epta.tim"
    for path, mjds in ((ref, ref_mjds), (other, other_mjds)):
        lines = ["FORMAT 1"] + [
            f"toa{i:05d} 1400.0 {mjd} 1.0 g -pn 0" for i, mjd in enumerate(mjds, 1)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    par = tmp_path / f"{pulsar}.par"
    par.write_text(MINIMAL_COMBO_PAR, encoding="utf-8")
    tim = tmp_path / f"{pulsar}.tim"
    write_combination_tim(
        pulsar=pulsar,
        reference_pta="nanograv",
        pta_tim_paths={"nanograv": ref, "epta": other},
        out_path=tim,
    )
    # Align TZR exactly as the renumber does, then infer under that TZR so the
    # oracle matches the pulse numbers the renumber will compute internally.
    _n, frq, mjd, site = _first_data_toa_tokens(ref)
    align_combination_tzr(par, tzrmjd=mjd, tzrfrq=frq, tzrsite=site)
    inferred = _infer_pulse_numbers(par, tim)
    return par, tim, ref, other, inferred


@pytest.mark.unit
def test_renumber_keeps_leg_pn_and_applies_constant_offset(tmp_path):
    pytest.importorskip("pint.models")
    par, tim, ref, other, inferred = _write_legs(
        tmp_path, ["54510.0", "54511.0", "54512.0"], ["54600.0", "54601.0", "54602.0"]
    )
    ref_inf, other_inf = inferred[:3], inferred[3:]

    # Coherent leg -pn = inferred - K, with distinct per-leg constants.
    K_ref, K_other = 100, 777
    _rewrite_tim_pn_sequential(ref, [v - K_ref for v in ref_inf])
    _rewrite_tim_pn_sequential(other, [v - K_other for v in other_inf])

    stats = renumber_combination_pulse_numbers(
        combination_par_path=par,
        combination_tim_path=tim,
        ordered_pta_tims=[("nanograv", ref), ("epta", other)],
    )

    # The single modal offset per leg is recovered exactly; clusters are tight.
    assert stats.per_pta_offset == {"nanograv": K_ref, "epta": K_other}
    assert stats.per_pta_mode_fraction == {"nanograv": 1.0, "epta": 1.0}
    assert stats.per_pta_max_deviation == {"nanograv": 0, "epta": 0}
    assert stats.per_pta_n_toas == {"nanograv": 3, "epta": 3}
    assert stats.n_toas == 6
    assert stats.pn0_abs == ref_inf[0]
    assert stats.tzrmjd == "54510.0"

    # Written ladder = the inferred numbering re-origined to the first ref TOA;
    # every within-leg difference is the leg's own (i.e. inferred) difference.
    written = _read_pn_sequence(ref) + _read_pn_sequence(other)
    assert written == [v - ref_inf[0] for v in inferred]
    assert _read_pn_sequence(ref)[0] == 0
    comb_par = par.read_text(encoding="utf-8")
    assert re.search(r"^TZRMJD 54510\.0\s*$", comb_par, re.M)


@pytest.mark.unit
def test_renumber_accepts_plurality_without_majority(tmp_path):
    pytest.importorskip("pint.models")
    par, tim, ref, other, inferred = _write_legs(
        tmp_path,
        ["54510.0", "54511.0"],
        ["54600.0", "54601.0", "54602.0", "54603.0"],
    )
    ref_inf, other_inf = inferred[:2], inferred[2:]

    _rewrite_tim_pn_sequential(ref, [v - 100 for v in ref_inf])
    # Split the other leg 50/50 between two offsets: plurality/tie → first-seen (100).
    split = [
        other_inf[0] - 100,
        other_inf[1] - 100,
        other_inf[2] - 105,
        other_inf[3] - 105,
    ]
    _rewrite_tim_pn_sequential(other, split)

    stats = renumber_combination_pulse_numbers(
        combination_par_path=par,
        combination_tim_path=tim,
        ordered_pta_tims=[("nanograv", ref), ("epta", other)],
    )
    assert stats.per_pta_offset["epta"] == 100
    assert stats.per_pta_mode_fraction["epta"] == 0.5
