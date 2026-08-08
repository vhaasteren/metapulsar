"""Unit tests for the combination par/tim writer."""

from __future__ import annotations

import re

import pytest

from metapulsar.combination_writer import (
    _is_noise_line,
    extract_fd_terms,
    fortran_d_to_e,
    sanitize_fortran_exponents,
    write_combination_par,
    write_combination_tim,
)

PTA_A_PAR = """\
PSR J1234+5678
RAJ 12:34:56
DECJ +56:78:90
F0 100.0
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
EPHEM DE440
CLK TT(BIPM2019)
UNITS TDB
JUMP -sys system_b 0 1
FD1 2.0E-03 1
EQUAD -f 1400 1.0e-6
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
