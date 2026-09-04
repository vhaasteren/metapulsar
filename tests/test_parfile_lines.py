"""Line-level par-text primitives (:mod:`metapulsar.parfile_lines`)."""

import pytest

from metapulsar.parfile_lines import (
    is_active_par_line,
    is_flag_token,
    iter_active_par_lines,
    join_par_lines,
    par_line_key,
    replace_token,
    token_spans,
)


class TestIsActiveParLine:
    @pytest.mark.parametrize(
        "line",
        [
            "F0 186.494 1 0.0001",
            "   PSR B1855+09",
            "MODE 1",
            "JUMP -fe L-wide -0.000009449 1",
            "CLK TT(BIPM2015)",  # starts with C but is not a comment
            "CORRECT_TROPOSPHERE N",
        ],
    )
    def test_active(self, line):
        assert is_active_par_line(line)

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "# Created: 2026-08-12",
            "   # indented comment",
            "C this is a tempo2 comment",
            "c lowercase tempo2 comment",
            "C",  # bare C
            "  C  ",
        ],
    )
    def test_inactive(self, line):
        assert not is_active_par_line(line)


def test_iter_active_par_lines_reports_source_indices():
    text = "# header\nPSR B1855+09\n\nC comment\nF0 186.494\n"
    assert list(iter_active_par_lines(text)) == [
        (1, "PSR B1855+09"),
        (4, "F0 186.494"),
    ]


def test_par_line_key_upper_cases_first_token():
    assert par_line_key("  f0   186.494 1") == "F0"
    assert par_line_key("FDJUMP1 -pta nanograv_9y 1.6e-04 1") == "FDJUMP1"


class TestIsFlagToken:
    @pytest.mark.parametrize("token", ["-pta", "-fe", "-sys", "-cycle_post34"])
    def test_flags(self, token):
        assert is_flag_token(token)

    @pytest.mark.parametrize("token", ["-9.449e-06", "-1", "-", "pta", "MJD", ""])
    def test_values(self, token):
        assert not is_flag_token(token)


def test_token_spans_locates_every_token():
    line = "F0    186.494  1"
    spans = token_spans(line)
    assert [line[a:b] for a, b in spans] == ["F0", "186.494", "1"]


class TestReplaceToken:
    def test_leaves_surrounding_columns_intact(self):
        line = "PX                        0.2929 1 0.2186"
        out = replace_token(line, 1, "0.3929")
        assert out == "PX                        0.3929 1 0.2186"

    def test_replaces_mask_value_not_flag_value(self):
        line = "JUMP            -fe L-wide -0.000009449 1 0.000009439"
        out = replace_token(line, 3, "-9.449e-06")
        assert out == "JUMP            -fe L-wide -9.449e-06 1 0.000009439"

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            replace_token("F0 186.494", 5, "1.0")


class TestJoinParLines:
    def test_preserves_trailing_newline(self):
        assert join_par_lines(["A 1", "B 2"], like="A 1\nB 2\n") == "A 1\nB 2\n"

    def test_preserves_absent_trailing_newline(self):
        assert join_par_lines(["A 1", "B 2"], like="A 1\nB 2") == "A 1\nB 2"
