"""Tests for the canonical .tim writer (flattening + metadata stamping)."""

import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from metapulsar.tim_canonical import (
    TEMPO2_MAX_TIM_LINE_BYTES,
    TimCanonicalizationError,
    TimIncludeScopeError,
    TimLegacyFormatError,
    _TimeAccum,
    _bake_mjd_token,
    _extract_sat_corrections,
    _format_fraction,
    _pint_legacy_heuristic_hit,
    convert_jump_mjd_par_text,
    discover_effective_tim_mode,
    ensure_par_mode,
    flatten_tim,
    inject_pulse_numbers,
    parse_jump_mjd_windows,
    stamp_metadata_flags,
    stamp_mjd_jump_pta_flags,
    write_canonical_tim,
)
from metapulsar.tim_file_analyzer import TimFileAnalyzer


def _toa(mjd, flags=""):
    return f" obs1 1400.0 {mjd} 1.0 g{flags}"


def _toa_lines(text: str):
    return [
        line
        for line in text.splitlines()
        if line.startswith(" ") and len(line.split()) >= 5
    ]


def _oracle_baked(mjd_token: str, total_seconds: Fraction) -> str:
    return _bake_mjd_token(mjd_token, _TimeAccum(total=total_seconds))


class TestSatCorrections:
    """Tempo2 ``-addsat`` (seconds) and ``-padd`` (turns via F0) baked into SAT."""

    def test_extract_addsat_sums_and_drops_pairs(self):
        total, kept = _extract_sat_corrections(
            ["-sys", "A", "-addsat", "+1", "-foo", "bar"]
        )
        assert total == Fraction(1)
        assert kept == ["-sys", "A", "-foo", "bar"]

    def test_extract_tolerates_valueless_neighbour(self):
        # A bare -gis (no value) appears in EPTA DR1 tims next to -addsat.
        total, kept = _extract_sat_corrections(["-addsat", "-1", "-gis"])
        assert total == Fraction(-1)
        assert kept == ["-gis"]

    def test_extract_addsat_without_value_raises(self):
        with pytest.raises(TimCanonicalizationError, match="without a value"):
            _extract_sat_corrections(["-sys", "A", "-addsat"])

    def test_extract_padd_converts_turns_to_seconds_via_f0(self):
        # 0.5 turns at F0 = 2 Hz -> 0.25 s.
        total, kept = _extract_sat_corrections(["-padd", "0.5"], f0=Fraction(2))
        assert total == Fraction(1, 4)
        assert kept == []

    def test_extract_padd_and_addsat_combine(self):
        total, kept = _extract_sat_corrections(
            ["-padd", "0.5", "-addsat", "+1", "-sys", "A"], f0=Fraction(2)
        )
        assert total == Fraction(1) + Fraction(1, 4)
        assert kept == ["-sys", "A"]

    def test_extract_padd_left_untouched_without_f0(self):
        # Mode discovery walks without a par; -padd is preserved, not baked.
        total, kept = _extract_sat_corrections(["-padd", "0.5", "-sys", "A"])
        assert total == Fraction(0)
        assert kept == ["-padd", "0.5", "-sys", "A"]

    def test_bake_addsat_equals_equivalent_time_bake(self):
        # extra_seconds and cumulative TIME are one combined offset -> same MJD,
        # so the SAT shifts inherit TIME's exact-Fraction, round-once precision.
        token = "58000.123456789012345"
        assert _bake_mjd_token(
            token, _TimeAccum(total=Fraction(0)), extra_seconds=Fraction(1)
        ) == _bake_mjd_token(token, _TimeAccum(total=Fraction(1)))

    def test_flatten_bakes_addsat_and_drops_flag(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\n"
            f"{_toa('58000.5', ' -sys A -addsat +1 -foo bar')}\n"
            f"{_toa('58001.5', ' -addsat -1 -sys B')}\n",
            encoding="utf-8",
        )

        text = flatten_tim(root).text

        assert "-addsat" not in text
        lines = _toa_lines(text)
        assert lines[0].split()[2] == _oracle_baked("58000.5", Fraction(1))
        assert "-sys A" in lines[0] and "-foo bar" in lines[0]
        assert lines[1].split()[2] == _oracle_baked("58001.5", Fraction(-1))
        assert "-sys B" in lines[1]

    def test_flatten_bakes_padd_via_f0_and_drops_flag(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\n" f"{_toa('58000.5', ' -sys A -padd 0.5')}\n",
            encoding="utf-8",
        )
        f0 = Fraction(2)  # 0.5 turns / 2 Hz = 0.25 s

        text = flatten_tim(root, f0=f0).text

        assert "-padd" not in text
        line = _toa_lines(text)[0]
        assert line.split()[2] == _oracle_baked("58000.5", Fraction(1, 4))
        assert "-sys A" in line

    def test_flatten_leaves_padd_when_no_f0(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\n" f"{_toa('58000.5', ' -sys A -padd 0.5')}\n",
            encoding="utf-8",
        )

        text = flatten_tim(root).text  # no f0

        assert "-padd 0.5" in text
        assert _toa_lines(text)[0].split()[2] == "58000.5"  # unbaked


class TestFlatten:
    def test_inlines_include_in_place(self, tmp_path):
        (tmp_path / "tims").mkdir()
        (tmp_path / "tims" / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{_toa(58000.0)}\nINCLUDE tims/chunk.tim\n{_toa(58002.0)}\n",
            encoding="utf-8",
        )

        lines = flatten_tim(root).text.splitlines()

        assert lines[0] == "FORMAT 1"
        assert "INCLUDE tims/chunk.tim" not in lines
        assert [line.split()[2] for line in _toa_lines("\n".join(lines))] == [
            "58000.0",
            "58001.0",
            "58002.0",
        ]
        assert [line.split()[0] for line in _toa_lines("\n".join(lines))] == [
            "toa00001",
            "toa00002",
            "toa00003",
        ]

    def test_preserves_comments_and_drops_mode(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nMODE 1\nC commented out TOA\n# hash comment\n"
            f"T2EFAC -sys foo 1.2\n{_toa(58000.0)}\n",
            encoding="utf-8",
        )

        result = flatten_tim(root)

        assert "MODE 1" not in result.text
        assert result.effective_mode == 1
        assert "C commented out TOA" in result.text
        assert "# hash comment" in result.text
        assert "T2EFAC -sys foo 1.2" in result.text

    def test_cc_comment_is_not_a_toa(self, tmp_path):
        """EPTA Effelsberg rejects TOAs with tempo2's two-character marker.

        Read as data, ``CC`` shifts every field left by one and the archive
        name lands in the frequency column, which PINT then dies on.
        """
        rejected = (
            "CC ?c062776.align.pazr.30min 1345.999 56522.2190165978952"
            "   4.381  g  -group EFF.EBPP.1360"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{rejected}\nC RFI ??c059607\n{_toa(58000.0)}\n",
            encoding="utf-8",
        )

        result = flatten_tim(root)

        assert [line.split()[0] for line in _toa_lines(result.text)] == ["toa00001"]
        assert "?c062776.align.pazr.30min" not in " ".join(_toa_lines(result.text))
        assert rejected in result.text  # kept as a comment, not dropped
        assert "C RFI ??c059607" in result.text

    def test_tempo2_leading_c_reject_markers_are_comments(self, tmp_path):
        """Tempo2 comments any leading uppercase C; no space required.

        EPTA releases use ``C?`` / ``CC?`` / ``C????`` / glued ``Cc…`` and
        prose ``CTHE …``; treating those as TOAs put the archive name in the
        frequency column (PINT float error) or prose tokens in the MJD field.
        """
        rejected = [
            "C? TIME -1",
            "C? c030680.pazr.iter.30min 1409.252 54182.2281713759008 0.027 g",
            "CC? c062799.align.pazr.30min 1354.224 56522.287303339192132 1.275 g",
            "C???? c015621.align.pazr.30min 1419.557 51849.5401158655181 0.252 g",
            "Cc055877.align.pazr.30min 2625.499 55995.8838774191826 5.166 g",
            "CTHE 2007 TOAS may be wrong - the settings were not correct",
            "Clow S/N, no pulse c056508.align.pazr.30min 2625.499 56031.9 64.959 g",
        ]
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\n" + "\n".join(rejected) + f"\n{_toa(58000.0)}\n",
            encoding="utf-8",
        )

        result = flatten_tim(root)

        assert [line.split()[0] for line in _toa_lines(result.text)] == ["toa00001"]
        for line in rejected:
            assert f"# {line}" in result.text.splitlines()

    def test_lowercase_c_archive_name_remains_a_toa(self, tmp_path):
        """Leading lowercase c is a live Effelsberg name, not a comment."""
        live = (
            "c055446.align.pazr.30min 2625.499 55961.1689989786605 1.716 g"
            " -group EFF.EBPP.2639"
        )
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{live}\n", encoding="utf-8")

        result = flatten_tim(root)

        assert len(_toa_lines(result.text)) == 1
        assert _toa_lines(result.text)[0].split()[2] == "55961.1689989786605"

    def test_bare_cr_mid_record_is_whitespace_not_a_line_break(self, tmp_path):
        """EPTA JBO files embed a lone CR before ``-padd``."""
        # Binary write so universal newlines cannot rewrite the CR first.
        body = (
            b"FORMAT 1\n"
            b"obs1 1520.0 56427.902533854616534 24.153 jb"
            b" -group JBO.DFB.1520 -sys JBO.DFB.1520\r -padd 0.497259\n"
            b"CJ130515_212931.NEFTp 1520.0 56427.9 24.153 jb"
            b" -sys JBO.DFB.1520\r -padd 0.497259\n"
        )
        root = tmp_path / "root.tim"
        root.write_bytes(body)

        result = flatten_tim(root)

        assert [line.split()[0] for line in _toa_lines(result.text)] == ["toa00001"]
        toa_line = _toa_lines(result.text)[0]
        assert "-padd" in toa_line and "0.497259" in toa_line
        assert "-padd 0.497259" not in result.text.splitlines()
        assert any(
            line.startswith("# CJ130515_212931.NEFTp")
            for line in result.text.splitlines()
        )

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("C Er. c038950", "C Er. c038950"),
            ("CC ?c062776 1345.999", "CC ?c062776 1345.999"),
            ("# hash comment", "# hash comment"),
            ("   C indented", "C indented"),
            ("   # indented hash", "# indented hash"),
            ("c lowercase", "# c lowercase"),
            ("cc lowercase two", "# cc lowercase two"),
            ("C", "# C"),
            ("CC", "# CC"),
            ("C? TIME -1", "# C? TIME -1"),
            (
                "CC? c062799.align.pazr.30min 1354.224",
                "# CC? c062799.align.pazr.30min 1354.224",
            ),
            (
                "Cc055877.align.pazr.30min 2625.499",
                "# Cc055877.align.pazr.30min 2625.499",
            ),
            (
                "CTHE 2007 TOAS may be wrong - the settings were not correct",
                "# CTHE 2007 TOAS may be wrong - the settings were not correct",
            ),
        ],
    )
    def test_comments_are_emitted_in_a_pint_safe_shape(
        self, tmp_path, source, expected
    ):
        """PINT only honors an uppercase marker at column 0 with text after it.

        Tempo2 accepts every form, so the canonical file normalizes the rest to
        ``#`` rather than passing through a line PINT would parse as a TOA.
        """
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{source}\n{_toa(58000.0)}\n", encoding="utf-8")

        result = flatten_tim(root)

        assert expected in result.text.splitlines()

    def test_canonical_comments_round_trip_through_pint(self, tmp_path):
        toa = pytest.importorskip("pint.toa")
        comments = "\n".join(
            [
                "CC ?c062776.align.pazr.30min 1345.999 56522.2190165978952 4.381 g",
                "CC? c062799.align.pazr.30min 1354.224 56522.287303339192132 1.275 g",
                "C? TIME -1",
                "C Er. c038950",
                "   C indented",
                "c lowercase",
                "CC",
            ]
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{comments}\n{_toa(58000.0)}\n{_toa(58010.0)}\n",
            encoding="utf-8",
        )
        canonical = tmp_path / "canonical.tim"
        canonical.write_text(flatten_tim(root).text + "\n", encoding="utf-8")

        toas = toa.get_TOAs(str(canonical), planets=False, ephem="DE421")

        assert len(toas) == 2

    def test_nested_and_repeated_includes(self, tmp_path):
        (tmp_path / "inner.tim").write_text(
            f"FORMAT 1\n{_toa(58002.0)}\n", encoding="utf-8"
        )
        (tmp_path / "outer.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\nINCLUDE inner.tim\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\nINCLUDE outer.tim\nINCLUDE outer.tim\n", encoding="utf-8"
        )

        mjds = [line.split()[2] for line in _toa_lines(flatten_tim(root).text)]

        assert mjds == ["58001.0", "58002.0", "58001.0", "58002.0"]

    def test_rebuilds_toa_layout_with_safe_name(self, tmp_path):
        raw = " obs1  1400.00000   58000.1234567890123456   0.997  g  -sys x"
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{raw}\n", encoding="utf-8")

        line = _toa_lines(flatten_tim(root).text)[0]
        assert line.split()[0] == "toa00001"
        assert line.split()[2] == "58000.1234567890123456"
        assert "-sys x" in line
        assert not _pint_legacy_heuristic_hit(line)

    def test_dodges_pint_column_41_collision(self, tmp_path):
        root = tmp_path / "epta.tim"
        root.write_text(
            "FORMAT 1\n"
            " raw 2627.949 55758.3650868593914 10.917 g "
            "-group EFF.EBPP.2639\n",
            encoding="utf-8",
        )

        result = flatten_tim(root)
        line = _toa_lines(result.text)[0]

        assert result.column_dodge_count == 1
        assert not _pint_legacy_heuristic_hit(line)
        pint_toa = pytest.importorskip("pint.toa")
        assert pint_toa._toa_format(line, fmt="Tempo2") == "Tempo2"
        assert line.split()[1:] == [
            "2627.949",
            "55758.3650868593914",
            "10.917",
            "g",
            "-group",
            "EFF.EBPP.2639",
        ]

    def test_dodge_layout_is_idempotent(self, tmp_path):
        root = tmp_path / "epta.tim"
        root.write_text(
            "FORMAT 1\n raw 2627.949 55758.3650868593914 10.917 g\n",
            encoding="utf-8",
        )
        once = flatten_tim(root)
        canonical = tmp_path / "canonical.tim"
        canonical.write_text(once.text, encoding="utf-8")

        twice = flatten_tim(canonical)

        assert twice.text == once.text
        assert twice.column_dodge_count == 1

    def test_missing_include_raises(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nINCLUDE nope.tim\n", encoding="utf-8")

        with pytest.raises(TimCanonicalizationError, match="INCLUDE file not found"):
            flatten_tim(root)

    def test_circular_include_raises(self, tmp_path):
        a = tmp_path / "a.tim"
        b = tmp_path / "b.tim"
        a.write_text("FORMAT 1\nINCLUDE b.tim\n", encoding="utf-8")
        b.write_text("FORMAT 1\nINCLUDE a.tim\n", encoding="utf-8")

        with pytest.raises(TimCanonicalizationError, match="Circular INCLUDE"):
            flatten_tim(a)

    def test_legacy_toa_raises(self, tmp_path):
        root = tmp_path / "legacy.tim"
        root.write_text(" name 1400.0 58000.0 1.0\n", encoding="utf-8")

        with pytest.raises(TimLegacyFormatError, match="without 'FORMAT 1'"):
            flatten_tim(root)

    def test_non_format1_declaration_raises(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 2\n", encoding="utf-8")

        with pytest.raises(TimLegacyFormatError, match="unsupported"):
            flatten_tim(root)

    def test_tempo2_end_in_include_does_not_end_parent(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\nEND\n{_toa(58099.0)}\n",
            encoding="utf-8",
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n",
            encoding="utf-8",
        )

        text = flatten_tim(root, timing_package="tempo2").text

        assert "END" not in text
        assert "58001.0" in text
        assert "58002.0" in text
        assert "58099.0" not in text

    def test_pint_end_in_include_ends_parent(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\nEND\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n",
            encoding="utf-8",
        )

        text = flatten_tim(root, timing_package="pint").text

        assert "END" in text
        assert "58001.0" in text
        assert "58002.0" not in text


class TestIncludeScopeGuard:
    """tempo2 scopes stateful directives per included file; PINT leaks them."""

    def test_balanced_time_inside_include_bakes(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\nTIME 1\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        text = flatten_tim(root).text
        lines = _toa_lines(text)

        assert "TIME" not in text
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(-1))
        assert lines[1].split()[2] == "58002.0"

    def test_unbalanced_time_at_include_eof_is_scoped_not_refused(self, tmp_path):
        """tempo2 drops the child's residual TIME; the parent's TOAs are clean."""
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\nTIME -1\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        result = flatten_tim(root, timing_package="tempo2")
        lines = _toa_lines(result.text)
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(-1))
        assert lines[1].split()[2] == "58002.0"  # parent unshifted

        (resolution,) = result.include_scope_resolutions
        assert resolution.boundary == "include_eof"
        assert resolution.directive == "TIME"
        assert resolution.disposition == "scoped"
        assert resolution.offset_seconds == Fraction(-2)
        assert resolution.toas_emitted_before == 1
        assert resolution.path == (tmp_path / "chunk.tim").resolve()
        assert resolution.include_path is None

    def test_time_live_at_include_entry_is_scoped_not_refused(self, tmp_path):
        """tempo2 withholds the parent's live TIME from the included TOAs."""
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nTIME -1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n",
            encoding="utf-8",
        )

        result = flatten_tim(root, timing_package="tempo2")
        lines = _toa_lines(result.text)
        assert lines[0].split()[2] == "58001.0"  # child unshifted
        assert lines[1].split()[2] == _oracle_baked("58002.0", Fraction(-1))

        (resolution,) = result.include_scope_resolutions
        assert resolution.boundary == "include_entry"
        assert resolution.disposition == "scoped"
        assert resolution.offset_seconds == Fraction(-1)
        assert resolution.toas_emitted_before == 0
        assert resolution.path == root.resolve()
        assert resolution.include_path == (tmp_path / "chunk.tim").resolve()

    def test_time_live_at_included_end_is_scoped_not_refused(self, tmp_path):
        """tempo2's `endit` is function-local too, so END behaves like EOF."""
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\nEND\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        result = flatten_tim(root, timing_package="tempo2")
        lines = _toa_lines(result.text)
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(-1))
        assert lines[1].split()[2] == "58002.0"

        (resolution,) = result.include_scope_resolutions
        assert resolution.boundary == "include_end"
        assert resolution.offset_seconds == Fraction(-1)

    def test_time_scope_resolution_warns_naming_the_file(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nINCLUDE chunk.tim\n", encoding="utf-8")

        with patch.object(logger, "warning") as mock_warning:
            flatten_tim(root, timing_package="tempo2")
        messages = [str(call) for call in mock_warning.call_args_list]
        assert any("chunk.tim" in message for message in messages), messages
        assert any("include_eof" in message for message in messages), messages

    def test_time_scope_resolution_recorded_but_silent_during_discovery(self, tmp_path):
        """Discovery walks the same tree just before flatten; don't warn twice."""
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nMODE 1\nINCLUDE chunk.tim\n", encoding="utf-8")

        with patch.object(logger, "warning") as mock_warning:
            assert discover_effective_tim_mode(root, timing_package="tempo2") == 1
        assert not mock_warning.called

    def test_balanced_time_records_no_resolution(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\nTIME +1\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        assert (
            flatten_tim(root, timing_package="tempo2").include_scope_resolutions == ()
        )

    def test_pint_leg_records_carried_child_contribution_only(self, tmp_path):
        """PINT inherits the parent's total, so measure the child's own delta."""
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -2\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nTIME -1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n",
            encoding="utf-8",
        )

        result = flatten_tim(root, timing_package="pint")
        lines = _toa_lines(result.text)
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(-3))
        assert lines[1].split()[2] == _oracle_baked("58002.0", Fraction(-3))

        entry, eof = result.include_scope_resolutions
        assert entry.boundary == "include_entry"
        assert entry.disposition == "carried"
        assert entry.offset_seconds == Fraction(-1)
        assert eof.boundary == "include_eof"
        assert eof.disposition == "carried"
        # -2, not the -3 the shared accumulator holds on the way out.
        assert eof.offset_seconds == Fraction(-2)

    def test_unbalanced_efac_in_include_raises(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nEFAC 1.3\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nINCLUDE chunk.tim\n", encoding="utf-8")

        with pytest.raises(TimIncludeScopeError, match="EFAC"):
            flatten_tim(root)

    def test_efac_reset_to_default_allows_include(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nEFAC 1.3\n{_toa(58001.0)}\nEFAC 1\n",
            encoding="utf-8",
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nINCLUDE chunk.tim\n", encoding="utf-8")

        assert "EFAC 1.3" in flatten_tim(root, timing_package="tempo2").text

    @pytest.mark.parametrize("directive", ["ESET 2.5", "PROFILE_DIR profiles"])
    def test_other_tempo2_local_state_in_include_raises(self, tmp_path, directive):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{directive}\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nINCLUDE chunk.tim\n", encoding="utf-8")

        with pytest.raises(TimIncludeScopeError, match=directive.split()[0]):
            flatten_tim(root, timing_package="tempo2")

    def test_jump_state_can_cross_include(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nJUMP\nINCLUDE chunk.tim\nJUMP\n", encoding="utf-8")

        assert "INCLUDE" not in flatten_tim(root, timing_package="tempo2").text

    def test_pint_shared_time_leaks_across_include(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nTIME -1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n",
            encoding="utf-8",
        )

        text = flatten_tim(root, timing_package="pint").text
        lines = _toa_lines(text)
        assert "TIME" not in text
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(-1))
        assert lines[1].split()[2] == _oracle_baked("58002.0", Fraction(-1))

    def test_tempo2_time_is_file_local(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\nTIME 1\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        lines = _toa_lines(flatten_tim(root, timing_package="tempo2").text)
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(-1))
        assert lines[1].split()[2] == "58002.0"

    def test_unbalanced_time_without_include_bakes(self, tmp_path):
        """A single-file tim has no boundary to cross, so no guard applies."""
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME -1\n{_toa(58000.0)}\n", encoding="utf-8")

        text = flatten_tim(root).text
        assert "TIME" not in text
        assert _toa_lines(text)[0].split()[2] == _oracle_baked("58000.0", Fraction(-1))


class TestStamping:
    def test_stamps_all_three_flags(self):
        text = f"FORMAT 1\n{_toa(58000.0)}\n"

        out = stamp_metadata_flags(text, pta_name="epta_dr2", timing_package="pint")

        assert "-pta epta_dr2" in out
        assert "-pta_dataset epta_dr2" in out
        assert "-timing_package pint" in out

    def test_renames_existing_flags_case_insensitively(self):
        text = (
            "FORMAT 1\n"
            + _toa(58000.0, " -PTA PPTA -Timing_Package tempo1 -sys keep")
            + "\n"
        )

        out = stamp_metadata_flags(text, pta_name="ppta_dr2", timing_package="tempo2")

        assert "-pta_orig PPTA" in out
        assert "-timing_package_orig tempo1" in out
        assert "-sys keep" in out
        assert "-pta ppta_dr2" in out
        assert "-timing_package tempo2" in out

    @pytest.mark.parametrize("existing", ["-pta epta_dr2", "-PTA epta_dr2"])
    def test_same_valued_release_pta_is_still_preserved(self, existing):
        text = "FORMAT 1\n" + _toa(58000.0, f" {existing}") + "\n"

        out = stamp_metadata_flags(text, pta_name="epta_dr2", timing_package="tempo2")

        assert "-pta_orig epta_dr2" in out
        assert " -pta epta_dr2" in out

    def test_renames_every_occurrence(self):
        text = "FORMAT 1\n" + _toa(58000.0, " -pta A -pta B") + "\n"

        out = stamp_metadata_flags(text, pta_name="x", timing_package="pint")

        assert out.count("-pta_orig") == 2
        assert out.count(" -pta x") == 1

    def test_existing_orig_flag_raises(self):
        text = "FORMAT 1\n" + _toa(58000.0, " -pta A -pta_orig B") + "\n"

        with pytest.raises(TimCanonicalizationError, match="already present"):
            stamp_metadata_flags(text, pta_name="x", timing_package="pint")

    def test_leaves_other_flags_and_pn_untouched(self):
        text = "FORMAT 1\n" + _toa(58000.0, " -pn 42 -group foo -to -0.9e-6") + "\n"

        out = stamp_metadata_flags(text, pta_name="x", timing_package="pint")

        assert "-pn 42" in out
        assert "-group foo" in out
        assert "-to -0.9e-6" in out

    def test_directives_and_comments_not_stamped(self):
        text = "FORMAT 1\nMODE 1\nC obs9 1400.0 58000.0 1.0 g\n# note\n"

        out = stamp_metadata_flags(text, pta_name="x", timing_package="pint")

        assert "-pta" not in out

    def test_rejects_value_with_whitespace(self):
        with pytest.raises(TimCanonicalizationError, match="whitespace-free"):
            stamp_metadata_flags(
                f"FORMAT 1\n{_toa(58000.0)}\n",
                pta_name="two words",
                timing_package="pint",
            )

    def test_rejects_overlong_pta_key(self):
        with pytest.raises(TimCanonicalizationError, match="MAX_FLAG_LEN"):
            stamp_metadata_flags(
                f"FORMAT 1\n{_toa(58000.0)}\n",
                pta_name="p" * 40,
                timing_package="pint",
            )

    def test_rejects_multibyte_pta_key_over_tempo2_byte_limit(self):
        with pytest.raises(TimCanonicalizationError, match="encoded bytes"):
            stamp_metadata_flags(
                f"FORMAT 1\n{_toa(58000.0)}\n",
                pta_name="é" * 20,
                timing_package="pint",
            )

    def test_rejects_exceeding_tempo2_max_flags(self):
        # 37 existing + 3 canonical = 40, and tempo2 exits on the 40th.
        many = "".join(f" -f{i} v{i}" for i in range(37))
        with pytest.raises(TimCanonicalizationError, match="MAX_FLAGS"):
            stamp_metadata_flags(
                "FORMAT 1\n" + _toa(58000.0, many) + "\n",
                pta_name="x",
                timing_package="pint",
            )

    def test_counts_tempo2_info_implicit_flag(self):
        # 36 textual + 3 canonical + tempo2's implicit -i = 40.
        many = "".join(f" -f{i} v{i}" for i in range(36))
        text = "FORMAT 1\nINFO 2\n" + _toa(58000.0, many) + "\n"

        with pytest.raises(TimCanonicalizationError, match="MAX_FLAGS"):
            stamp_metadata_flags(text, pta_name="x", timing_package="tempo2")

    def test_skipped_info_does_not_change_implicit_flag_count(self):
        # INFO 2 stays active because tempo2 ignores INFO -1 inside SKIP.
        many = "".join(f" -f{i} v{i}" for i in range(36))
        text = "FORMAT 1\nINFO 2\nSKIP\nINFO -1\nNOSKIP\n" + _toa(58000.0, many) + "\n"

        with pytest.raises(TimCanonicalizationError, match="MAX_FLAGS"):
            stamp_metadata_flags(text, pta_name="x", timing_package="tempo2")

    def test_rejects_stamped_line_too_long_for_tempo2(self):
        text = "FORMAT 1\n" + _toa(58000.0, f" -long {'x' * 950}") + "\n"

        with pytest.raises(TimCanonicalizationError, match="safely read"):
            stamp_metadata_flags(text, pta_name="x", timing_package="tempo2")


class TestWriteCanonicalTim:
    def test_writes_stamped_standalone_file(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0, ' -pta EPTA')}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nINCLUDE chunk.tim\n", encoding="utf-8")

        out = write_canonical_tim(
            root,
            pta_name="epta_dr2",
            timing_package="tempo2",
            out_path=tmp_path / "out" / "epta_dr2.tim",
        )

        text = out.path.read_text(encoding="utf-8")
        assert "INCLUDE" not in text
        assert "-pta_orig EPTA" in text
        assert "-pta epta_dr2" in text
        assert "toa00001" in text

    def test_reports_metadata_and_dodge_count(self, tmp_path):
        root = tmp_path / "epta.tim"
        root.write_text(
            "FORMAT 1\n raw 2627.949 55758.3650868593914 10.917 g\n",
            encoding="utf-8",
        )

        result = write_canonical_tim(
            root,
            pta_name="epta",
            timing_package="pint",
            out_path=tmp_path / "canonical.tim",
        )

        assert result.column_dodge_count == 1
        assert result.tim_metadata.toa_count == 1
        assert result.tim_metadata.pn_status == "none"

    def test_toa_count_survives_canonicalization(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n{_toa(58002.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58003.0)}\n", encoding="utf-8"
        )
        before = TimFileAnalyzer().get_tim_metadata(root)

        out = write_canonical_tim(
            root,
            pta_name="pta1",
            timing_package="pint",
            out_path=tmp_path / "canon.tim",
        )
        after = TimFileAnalyzer().get_tim_metadata(out.path)

        assert after.toa_count == before.toa_count == 3
        assert after.mjd_min == before.mjd_min
        assert after.mjd_max == before.mjd_max

    def test_is_idempotent_on_own_output(self, tmp_path):
        """An exported artifact can be fed back in as a dataset unchanged."""
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{_toa(58000.0)}\n", encoding="utf-8")
        once = write_canonical_tim(
            root,
            pta_name="pta1",
            timing_package="pint",
            out_path=tmp_path / "once.tim",
        )
        twice = write_canonical_tim(
            once.path,
            pta_name="pta1",
            timing_package="pint",
            out_path=tmp_path / "twice.tim",
        )

        assert twice.path.read_text(encoding="utf-8") == once.path.read_text(
            encoding="utf-8"
        )

    def test_is_idempotent_with_jump_mjd_par_text(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{_toa(58000.0)}\n{_toa(58500.0)}\n", encoding="utf-8"
        )
        par_text = "PSRJ J0000+0000\nJUMP MJD 57900 59000 -1e-7 1\n"
        once = write_canonical_tim(
            root,
            pta_name="EPTA",
            timing_package="tempo2",
            out_path=tmp_path / "once.tim",
            par_text=par_text,
        )
        twice = write_canonical_tim(
            once.path,
            pta_name="EPTA",
            timing_package="tempo2",
            out_path=tmp_path / "twice.tim",
            par_text=par_text,
        )

        assert twice.path.read_text(encoding="utf-8") == once.path.read_text(
            encoding="utf-8"
        )
        assert once.path.read_text(encoding="utf-8").count("-mjd_jump_pta EPTA_1") == 2

    def test_restamping_under_a_new_pta_name_preserves_the_old_one(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{_toa(58000.0)}\n", encoding="utf-8")
        once = write_canonical_tim(
            root,
            pta_name="pta1",
            timing_package="pint",
            out_path=tmp_path / "once.tim",
        )
        twice = write_canonical_tim(
            once.path,
            pta_name="pta2",
            timing_package="pint",
            out_path=tmp_path / "twice.tim",
        )

        text = twice.path.read_text(encoding="utf-8")
        assert "-pta_orig pta1" in text
        assert "-pta pta2" in text
        assert "-pta_dataset_orig pta1" in text
        assert "-timing_package_orig pint" in text


class TestInjectPulseNumbers:
    @staticmethod
    def _write(path: Path, *lines: str) -> Path:
        path.write_text("FORMAT 1\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_joins_by_name_and_replaces_all_existing_pn(self, tmp_path):
        canonical = self._write(
            tmp_path / "canonical.tim",
            " toa00001 1400 58000 1 g -PN 8 -pn 9 -sys a",
            " toa00002 1400 58001 1 g -sys b",
        )
        derived = self._write(
            tmp_path / "derived.tim",
            " toa00002 1400 58001 1 g -pn 22.0",
            " toa00001 1400 58000 1 g -pn 11",
        )

        count = inject_pulse_numbers(canonical, derived_tim=derived)
        lines = _toa_lines(canonical.read_text(encoding="utf-8"))

        assert count == 2
        assert lines[0].split()[-4:] == ["-sys", "a", "-pn", "11"]
        assert lines[1].split()[-4:] == ["-sys", "b", "-pn", "22"]
        assert sum(token.lower() == "-pn" for token in lines[0].split()) == 1

    def test_leaves_unmatched_canonical_toa_byte_identical(self, tmp_path):
        unmatched = " toa00001 1400 58000 1 g -sys skipped"
        canonical = self._write(
            tmp_path / "canonical.tim",
            "SKIP",
            unmatched,
            "NOSKIP",
            " toa00002 1400 58001 1 g -sys active",
        )
        derived = self._write(
            tmp_path / "derived.tim",
            " toa00002 1400 58001 1 g -pn 22",
        )

        assert inject_pulse_numbers(canonical, derived_tim=derived) == 1

        assert unmatched in canonical.read_text(encoding="utf-8").splitlines()

    @pytest.mark.parametrize(
        ("derived_line", "message"),
        [
            (" toa00001 1400 58000 1 g", "exactly one -pn"),
            (" toa00001 1400 58000 1 g -pn nope", "non-integral"),
            (" toa00001 1400 58000 1 g -pn 1.5", "non-integral"),
            (" toa99999 1400 58000 1 g -pn 1", "absent from canonical"),
        ],
    )
    def test_rejects_bad_derived_data_without_modifying_file(
        self, tmp_path, derived_line, message
    ):
        canonical = self._write(
            tmp_path / "canonical.tim", " toa00001 1400 58000 1 g -sys keep"
        )
        before = canonical.read_bytes()
        derived = self._write(tmp_path / "derived.tim", derived_line)

        with pytest.raises(TimCanonicalizationError, match=message):
            inject_pulse_numbers(canonical, derived_tim=derived)

        assert canonical.read_bytes() == before

    def test_rejects_duplicate_derived_name_without_modifying_file(self, tmp_path):
        canonical = self._write(tmp_path / "canonical.tim", " toa00001 1400 58000 1 g")
        before = canonical.read_bytes()
        derived = self._write(
            tmp_path / "derived.tim",
            " toa00001 1400 58000 1 g -pn 1",
            " toa00001 1400 58000 1 g -pn 2",
        )

        with pytest.raises(TimCanonicalizationError, match="Duplicate TOA name"):
            inject_pulse_numbers(canonical, derived_tim=derived)

        assert canonical.read_bytes() == before

    def test_rejects_tempo2_flag_overflow_without_modifying_file(self, tmp_path):
        flags = "".join(f" -f{i} v{i}" for i in range(39))
        canonical = self._write(
            tmp_path / "canonical.tim", f" toa00001 1400 58000 1 g{flags}"
        )
        before = canonical.read_bytes()
        derived = self._write(
            tmp_path / "derived.tim", " toa00001 1400 58000 1 g -pn 1"
        )

        with pytest.raises(TimCanonicalizationError, match="MAX_FLAGS"):
            inject_pulse_numbers(canonical, derived_tim=derived)

        assert canonical.read_bytes() == before

    def test_rejects_tempo2_line_overflow_without_modifying_file(self, tmp_path):
        prefix = " toa00001 1400 58000 1 g -long "
        padding = "x" * (TEMPO2_MAX_TIM_LINE_BYTES - len(prefix))
        canonical = self._write(tmp_path / "canonical.tim", prefix + padding)
        before = canonical.read_bytes()
        derived = self._write(
            tmp_path / "derived.tim", " toa00001 1400 58000 1 g -pn 1"
        )

        with pytest.raises(TimCanonicalizationError, match="safely read"):
            inject_pulse_numbers(canonical, derived_tim=derived)

        assert canonical.read_bytes() == before


class TestJumpMjd:
    def test_parse_order_skips_comments_and_flag_jumps(self):
        par = (
            "PSRJ J0000+0000\n"
            "JUMP MJD 55000 56000 0 0\n"
            "# JUMP MJD 57000 58000 1 0\n"
            "JUMP -f foo -1e-6 1\n"
            "JUMP MJD 56000 57000 1e-7 1\n"
        )

        windows = parse_jump_mjd_windows(par)

        assert windows == [
            (Decimal("55000"), Decimal("56000"), ("0", "0")),
            (Decimal("56000"), Decimal("57000"), ("1e-7", "1")),
        ]

    def test_tempo2_selection_is_half_open(self):
        text = (
            "FORMAT 1\n"
            f"{_toa(57999.9)}\n"
            f"{_toa(58000.0)}\n"
            f"{_toa(58999.999)}\n"
            f"{_toa(59000.0)}\n"
        )
        out = stamp_mjd_jump_pta_flags(
            text,
            pta_name="PPTA",
            windows=[(Decimal("58000"), Decimal("59000"))],
            timing_package="tempo2",
        )
        lines = [line for line in out.splitlines() if line.startswith(" ")]
        assert "-mjd_jump_pta" not in lines[0]
        assert "-mjd_jump_pta PPTA_1" in lines[1]
        assert "-mjd_jump_pta PPTA_1" in lines[2]
        assert "-mjd_jump_pta" not in lines[3]

    def test_pint_selection_is_closed(self):
        text = (
            "FORMAT 1\n"
            f"{_toa(57999.9)}\n"
            f"{_toa(58000.0)}\n"
            f"{_toa(58999.999)}\n"
            f"{_toa(59000.0)}\n"
        )
        out = stamp_mjd_jump_pta_flags(
            text,
            pta_name="PPTA",
            windows=[(Decimal("58000"), Decimal("59000"))],
            timing_package="pint",
        )
        lines = [line for line in out.splitlines() if line.startswith(" ")]
        assert "-mjd_jump_pta" not in lines[0]
        assert "-mjd_jump_pta PPTA_1" in lines[1]
        assert "-mjd_jump_pta PPTA_1" in lines[2]
        assert "-mjd_jump_pta PPTA_1" in lines[3]

    def test_values_use_pta_name_and_one_based_index(self):
        text = f"FORMAT 1\n{_toa(55500.0)}\n{_toa(56500.0)}\n"
        out = stamp_mjd_jump_pta_flags(
            text,
            pta_name="EPTA",
            windows=[
                (Decimal("55000"), Decimal("56000")),
                (Decimal("56000"), Decimal("57000")),
            ],
            timing_package="tempo2",
        )
        assert "-mjd_jump_pta EPTA_1" in out
        assert "-mjd_jump_pta EPTA_2" in out

    def test_unaffected_toas_have_no_flag(self):
        text = f"FORMAT 1\n{_toa(50000.0)}\n"
        out = stamp_mjd_jump_pta_flags(
            text,
            pta_name="EPTA",
            windows=[(Decimal("55000"), Decimal("56000"))],
            timing_package="tempo2",
        )
        assert "-mjd_jump_pta" not in out

    def test_stamping_runs_after_metadata_flags(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{_toa(58000.0)}\n", encoding="utf-8")
        out = write_canonical_tim(
            root,
            pta_name="EPTA",
            timing_package="tempo2",
            out_path=tmp_path / "out.tim",
            par_text="JUMP MJD 57000 59000 -2e-7 1\n",
        )
        line = next(
            line
            for line in out.path.read_text(encoding="utf-8").splitlines()
            if " -pta " in line
        )
        assert "-pta EPTA" in line
        assert "-pta_dataset EPTA" in line
        assert "-timing_package tempo2" in line
        assert "-mjd_jump_pta EPTA_1" in line

    def test_overlapping_windows_raise(self):
        text = f"FORMAT 1\n{_toa(58500.0)}\n"
        with pytest.raises(TimCanonicalizationError, match="overlapping JUMP MJD"):
            stamp_mjd_jump_pta_flags(
                text,
                pta_name="EPTA",
                windows=[
                    (Decimal("58000"), Decimal("59000")),
                    (Decimal("58400"), Decimal("58600")),
                ],
                timing_package="tempo2",
            )

    def test_adjacent_boundary_differs_by_package(self):
        text = f"FORMAT 1\n{_toa(59000.0)}\n"
        windows = [
            (Decimal("58000"), Decimal("59000")),
            (Decimal("59000"), Decimal("60000")),
        ]
        with pytest.raises(TimCanonicalizationError, match="overlapping JUMP MJD"):
            stamp_mjd_jump_pta_flags(
                text,
                pta_name="NG",
                windows=windows,
                timing_package="pint",
            )
        out = stamp_mjd_jump_pta_flags(
            text,
            pta_name="NG",
            windows=windows,
            timing_package="tempo2",
        )
        assert "-mjd_jump_pta NG_2" in out
        assert "-mjd_jump_pta NG_1" not in out

    def test_renames_existing_release_flag(self):
        text = "FORMAT 1\n" + _toa(58000.0, " -mjd_jump_pta releaseval") + "\n"
        out = stamp_mjd_jump_pta_flags(
            text,
            pta_name="EPTA",
            windows=[(Decimal("57000"), Decimal("59000"))],
            timing_package="tempo2",
        )
        assert "-mjd_jump_pta_orig releaseval" in out
        assert "-mjd_jump_pta EPTA_1" in out

    def test_convert_jump_mjd_par_text(self):
        release = parse_jump_mjd_windows(
            "JUMP MJD 58925 65000 -2e-7 1\nJUMP -f keep 0 1\n"
        )
        engine = (
            "PSRJ J0613-0200\n" "JUMP -f keep 0 1\n" "JUMP MJD 58925 65000 -2e-7 1\n"
        )
        out = convert_jump_mjd_par_text(
            engine, pta_name="PPTA", release_windows=release
        )
        assert "JUMP -mjd_jump_pta PPTA_1 -2e-7 1" in out
        assert "JUMP MJD" not in out
        assert "JUMP -f keep 0 1" in out

    def test_convert_raises_on_missing_engine_window(self):
        release = parse_jump_mjd_windows("JUMP MJD 58925 65000 -2e-7 1\n")
        with pytest.raises(TimCanonicalizationError, match="missing from engine"):
            convert_jump_mjd_par_text(
                "PSRJ J0000+0000\n",
                pta_name="PPTA",
                release_windows=release,
            )

    def test_convert_raises_on_extra_engine_window(self):
        release = parse_jump_mjd_windows("JUMP MJD 58925 65000 -2e-7 1\n")
        with pytest.raises(TimCanonicalizationError, match="no matching release"):
            convert_jump_mjd_par_text(
                "JUMP MJD 58925 65000 -2e-7 1\nJUMP MJD 10000 20000 0 0\n",
                pta_name="PPTA",
                release_windows=release,
            )

    def test_flag_budget_includes_mjd_jump(self):
        # 36 existing + 3 metadata already stamped would leave room for one more;
        # start from a line that already has 39 flags so one more hits MAX_FLAGS.
        many = "".join(f" -f{i} v{i}" for i in range(39))
        text = "FORMAT 1\n" + _toa(58000.0, many) + "\n"
        with pytest.raises(TimCanonicalizationError, match="MAX_FLAGS"):
            stamp_mjd_jump_pta_flags(
                text,
                pta_name="x",
                windows=[(Decimal("57000"), Decimal("59000"))],
                timing_package="pint",
            )


class TestExactTimeArithmetic:
    """§6.1 Exact, bounded TIME arithmetic."""

    def test_time_86400_adds_one_day(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME 86400\n{_toa(58000.0)}\n", encoding="utf-8")
        mjd = _toa_lines(flatten_tim(root).text)[0].split()[2]
        assert mjd == _oracle_baked("58000.0", Fraction(86400))
        assert Fraction(mjd) == Fraction("58001")

    def test_cumulative_offsets_within_one_file(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nTIME 1\nTIME 2\n{_toa(58000.0)}\nTIME -3\n{_toa(58001.0)}\n",
            encoding="utf-8",
        )
        lines = _toa_lines(flatten_tim(root).text)
        assert lines[0].split()[2] == _oracle_baked("58000.0", Fraction(3))
        assert lines[1].split()[2] == "58001.0"

    def test_exact_sum_across_magnitudes(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nTIME 1e9\nTIME 1\nTIME -1e9\n{_toa(58000.0)}\n",
            encoding="utf-8",
        )
        mjd = _toa_lines(flatten_tim(root).text)[0].split()[2]
        assert mjd == _oracle_baked("58000.0", Fraction(1))

    def test_exponent_span_cancellation_keeps_time_live(self, tmp_path):
        """§6.1.4: exact total 1e-30 s; a digit-budgeted Decimal gives 0E-18.

        The residual is far below the emitted MJD scale, so the observable is
        the recorded INCLUDE-boundary offset, which pins the *exact* value a
        budgeted accumulator would have lost.
        """
        (tmp_path / "child.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\nTIME 1e9\nTIME 1e-30\nTIME -1e9\nINCLUDE child.tim\n",
            encoding="utf-8",
        )
        result = flatten_tim(root, timing_package="tempo2")
        (resolution,) = result.include_scope_resolutions
        assert resolution.boundary == "include_entry"
        assert resolution.offset_seconds == Fraction(1, 10**30)

    def test_true_cancellation_balances_include_boundary(self, tmp_path):
        (tmp_path / "child.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            "FORMAT 1\nTIME 1e9\nTIME -1e9\nINCLUDE child.tim\n", encoding="utf-8"
        )
        result = flatten_tim(root, timing_package="tempo2")
        assert _toa_lines(result.text)[0].split()[2] == "58001.0"
        assert result.include_scope_resolutions == ()

    def test_forty_digit_fractional_delta(self, tmp_path):
        delta = "0." + ("0" * 39) + "1"
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME {delta}\n{_toa(58000.0)}\n", encoding="utf-8")
        mjd = _toa_lines(flatten_tim(root).text)[0].split()[2]
        assert mjd == _oracle_baked("58000.0", Fraction(delta))

    def test_half_even_rounding_at_output_scale(self):
        # Exactly halfway between two 17-digit fixed-point values rounds to even.
        half = Fraction(1, 2 * 10**17)
        assert _format_fraction(Fraction(58000) + half, 17) == (
            "58000.00000000000000000"
        )
        # Last digit of the lower neighbour is odd (…00001); half rounds up to …2.
        odd_base = Fraction(58000) + Fraction(1, 10**17)
        assert _format_fraction(odd_base + half, 17) == "58000.00000000000000002"

    def test_zero_and_cancelling_time_preserves_mjd_token(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nTIME 0\n{_toa('58000.12345678901234567')}\n"
            f"TIME 5\nTIME -5\n{_toa('58001.5')}\n",
            encoding="utf-8",
        )
        lines = _toa_lines(flatten_tim(root).text)
        assert lines[0].split()[2] == "58000.12345678901234567"
        assert lines[1].split()[2] == "58001.5"

    @pytest.mark.parametrize("token", ["nope", "NaN", "Infinity", "-Infinity"])
    def test_invalid_time_raises(self, tmp_path, token):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME {token}\n{_toa(58000.0)}\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError):
            flatten_tim(root)

    def test_time_missing_argument_raises(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME\n{_toa(58000.0)}\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError, match="TIME without offset"):
            flatten_tim(root)

    def test_bounded_decimal_rejects_amplifying_exponents(self, tmp_path):
        for directive in (
            "TIME 1e999999999",
            "TIME 1e-999999999",
        ):
            root = tmp_path / "root.tim"
            root.write_text(
                f"FORMAT 1\n{directive}\n{_toa(58000.0)}\n", encoding="utf-8"
            )
            with pytest.raises(TimCanonicalizationError):
                flatten_tim(root)

        for mjd in ("1e999999999", "1e-999999999"):
            root = tmp_path / "root.tim"
            root.write_text(f"FORMAT 1\n{_toa(mjd)}\n", encoding="utf-8")
            with pytest.raises(TimCanonicalizationError):
                flatten_tim(root)

    def test_scientific_notation_mjd_uses_min_seventeen_digits(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME 1\n{_toa('5.8e4')}\n", encoding="utf-8")
        mjd = _toa_lines(flatten_tim(root).text)[0].split()[2]
        assert "." in mjd
        assert len(mjd.split(".")[1]) >= 17
        assert mjd == _oracle_baked("5.8e4", Fraction(1))


class TestMjdValidation:
    """§6.2 MJD validation, input and output."""

    @pytest.mark.parametrize("mjd", ["NaN", "Infinity", "nope", "-1", "1000001"])
    def test_zero_time_still_validates_mjd(self, tmp_path, mjd):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{_toa(mjd)}\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError):
            flatten_tim(root)
        assert not (tmp_path / "out.tim").exists()

    @pytest.mark.parametrize("mjd", ["NaN", "Infinity", "nope", "-1", "1000001"])
    def test_live_time_still_validates_mjd(self, tmp_path, mjd):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME 1\n{_toa(mjd)}\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError):
            flatten_tim(root)

    def test_baked_mjd_range_quotes_exact_fraction(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME -1e9\n{_toa(0.5)}\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError, match=r"-624973/54") as exc:
            flatten_tim(root)
        assert "0.5" in str(exc.value)

    def test_near_boundary_acceptance_and_rejection(self, tmp_path):
        # 999999.9 + small TIME stays <= 1e6
        root = tmp_path / "ok.tim"
        root.write_text(f"FORMAT 1\nTIME 1\n{_toa('999999.9')}\n", encoding="utf-8")
        assert _toa_lines(flatten_tim(root).text)

        # Crossing 1e6 raises
        root = tmp_path / "bad.tim"
        root.write_text(f"FORMAT 1\nTIME 86400\n{_toa('999999.9')}\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError, match="Baked TOA MJD"):
            flatten_tim(root)


class TestShortDataLines:
    """Data lines with < 5 fields: tempo2 drops them, so canonicalization does.

    EPTA DR2's Jodrell files continue a TOA's flags onto the next physical line
    (``-padd <value>`` alone). Tempo2's reader leaves ``valid == 0`` for those
    and never increments ``nobs``, and the line matches no directive keyword
    either, so the release solution was fitted without those flags.
    """

    def test_short_line_is_dropped_not_refused(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{_toa(58000.0)}\n -padd 0.497259\n{_toa(58001.0)}\n",
            encoding="utf-8",
        )

        result = flatten_tim(root)

        assert [line.split()[2] for line in _toa_lines(result.text)] == [
            "58000.0",
            "58001.0",
        ]
        assert "padd" not in result.text
        # Names stay dense: the dropped line never consumed a counter slot.
        assert [line.split()[0] for line in _toa_lines(result.text)] == [
            "toa00001",
            "toa00002",
        ]

    def test_drop_is_recorded_with_provenance(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{_toa(58000.0)}\n -padd 0.497259\n{_toa(58001.0)}\n"
            "-padd -0.22244\n",
            encoding="utf-8",
        )

        dropped = flatten_tim(root).dropped_lines

        assert [d.line_number for d in dropped] == [3, 5]
        assert [d.text for d in dropped] == ["-padd 0.497259", "-padd -0.22244"]
        assert [d.toas_emitted_before for d in dropped] == [1, 2]
        assert {d.path for d in dropped} == {root.resolve()}

    def test_drop_is_reported_from_the_file_that_held_it(self, tmp_path):
        (tmp_path / "tims").mkdir()
        (tmp_path / "tims" / "jbo.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n -padd 0.497259\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{_toa(58000.0)}\nINCLUDE tims/jbo.tim\n", encoding="utf-8"
        )

        dropped = flatten_tim(root).dropped_lines

        assert len(dropped) == 1
        assert dropped[0].path == (tmp_path / "tims" / "jbo.tim").resolve()
        assert dropped[0].line_number == 3

    def test_clean_tree_records_nothing(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{_toa(58000.0)}\n", encoding="utf-8")

        assert flatten_tim(root).dropped_lines == ()

    def test_write_canonical_tim_surfaces_the_drop(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n{_toa(58000.0)}\n -padd 0.497259\n", encoding="utf-8"
        )

        result = write_canonical_tim(
            root,
            pta_name="EPTA",
            timing_package="tempo2",
            out_path=tmp_path / "out.tim",
        )

        assert len(result.dropped_lines) == 1
        assert result.tim_metadata.toa_count == 1
        assert "padd" not in (tmp_path / "out.tim").read_text(encoding="utf-8")

    def test_directive_lines_are_untouched(self, tmp_path):
        """A short line is only dropped when it classifies as data."""
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nSKIP\n{_toa(58000.0)}\nNOSKIP\n", encoding="utf-8")

        result = flatten_tim(root)

        assert result.dropped_lines == ()
        assert "SKIP" in result.text and "NOSKIP" in result.text


class TestTraversalSwitches:
    """§6.3 Single traversal, two switches, legacy policy."""

    def test_discover_returns_last_mode(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nMODE 0\n{_toa(58000.0)}\nMODE 1\n{_toa(58001.0)}\n",
            encoding="utf-8",
        )
        assert discover_effective_tim_mode(root) == 1

    def test_mode_only_inside_tempo2_skip_is_none(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nSKIP\nMODE 1\n{_toa(58000.0)}\nNOSKIP\n{_toa(58001.0)}\n",
            encoding="utf-8",
        )
        assert discover_effective_tim_mode(root, timing_package="tempo2") is None
        result = flatten_tim(root, timing_package="tempo2")
        assert result.effective_mode is None
        assert "MODE" not in result.text
        assert "TIME" not in result.text

    def test_structural_parity_format1_matrix(self, tmp_path):
        cases = {
            "missing_include": "FORMAT 1\nINCLUDE nope.tim\n",
            "circular": None,  # built below
            "format_no_arg": "FORMAT\n",
            "bad_time": f"FORMAT 1\nTIME nope\n{_toa(58000.0)}\n",
            "bad_mode": f"FORMAT 1\nMODE nope\n{_toa(58000.0)}\n",
            # TIME is baked, so only the *emitted* directives still refuse.
            "unbalanced_efac": None,
        }
        a = tmp_path / "a.tim"
        b = tmp_path / "b.tim"
        a.write_text("FORMAT 1\nINCLUDE b.tim\n", encoding="utf-8")
        b.write_text("FORMAT 1\nINCLUDE a.tim\n", encoding="utf-8")
        cases["circular"] = a
        child = tmp_path / "child.tim"
        child.write_text(f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8")
        unbalanced = tmp_path / "unbalanced.tim"
        unbalanced.write_text(
            "FORMAT 1\nEFAC 1.3\nINCLUDE child.tim\n", encoding="utf-8"
        )
        cases["unbalanced_efac"] = unbalanced

        for name, content in cases.items():
            if isinstance(content, Path):
                path = content
            else:
                path = tmp_path / f"{name}.tim"
                path.write_text(content, encoding="utf-8")
            with pytest.raises(Exception) as flat_exc:
                flatten_tim(path, timing_package="tempo2")
            with pytest.raises(Exception) as disc_exc:
                discover_effective_tim_mode(path, timing_package="tempo2")
            assert type(flat_exc.value) is type(disc_exc.value), name

    def test_structural_parity_on_dropped_short_line(self, tmp_path):
        """A short FORMAT 1 line is not an error, so parity is "both survive"."""
        path = tmp_path / "short_toa.tim"
        path.write_text("FORMAT 1\nMODE 1\n obs1 1400.0 58000.0\n", encoding="utf-8")

        assert discover_effective_tim_mode(path, timing_package="tempo2") == 1
        result = flatten_tim(path, timing_package="tempo2")
        assert _toa_lines(result.text) == []
        assert len(result.dropped_lines) == 1

    @pytest.mark.parametrize("fmt", ["0", "2", "tempo1"])
    def test_explicit_legacy_format_discovery(self, tmp_path, fmt):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT {fmt}\nMODE 1\n name 1400.0 58000.0 1.0\n", encoding="utf-8"
        )
        assert discover_effective_tim_mode(root) == 1
        with pytest.raises(TimLegacyFormatError):
            flatten_tim(root)

    def test_untagged_legacy_tolerated_by_discovery(self, tmp_path):
        root = tmp_path / "legacy.tim"
        root.write_text("MODE 1\n name 1400.0 58000.0 1.0\n", encoding="utf-8")
        assert discover_effective_tim_mode(root) == 1
        with pytest.raises(TimLegacyFormatError):
            flatten_tim(root)

    def test_untagged_legacy_write_canonical_converts(self, tmp_path):
        """§6.3.20: write_canonical_tim converts untagged Princeton via PINT."""
        root = tmp_path / "legacy.tim"
        root.write_text(
            "MODE 1\n"
            "1               1400.000 54510.2858714192189    1.50\n"
            "1               1400.000 54520.2767051885166    1.50\n",
            encoding="utf-8",
        )
        par = (
            Path(__file__).parent / "fixtures" / "sample_parfiles" / "simple.par"
        ).read_text(encoding="utf-8")
        with pytest.raises(TimLegacyFormatError):
            flatten_tim(root)
        assert discover_effective_tim_mode(root, timing_package="pint") == 1
        out = write_canonical_tim(
            root,
            pta_name="EPTA",
            timing_package="pint",
            out_path=tmp_path / "out.tim",
            par_text=par,
        )
        text = out.path.read_text(encoding="utf-8")
        assert text.startswith("FORMAT 1\n")
        assert len(_toa_lines(text)) == 2

    def test_legacy_include_descent_finds_mode(self, tmp_path):
        child = tmp_path / "child.tim"
        child.write_text("FORMAT 1\nMODE 1\n", encoding="utf-8")
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 0\nINCLUDE child.tim\n", encoding="utf-8")
        assert discover_effective_tim_mode(root) == 1

        # Included FORMAT 1 child of a legacy parent is still tokenized.
        child.write_text(f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8")
        root.write_text(
            "FORMAT 0\nMODE 1\nINCLUDE child.tim\n name 1400.0 58000.0 1.0\n",
            encoding="utf-8",
        )
        assert discover_effective_tim_mode(root) == 1

    def test_argumentless_format_is_policy_error_not_legacy(self, tmp_path):
        # §1.4.1: MetaPulsar policy — bare FORMAT is TimCanonicalizationError,
        # not TimLegacyFormatError, so write_canonical_tim does not convert.
        root = tmp_path / "root.tim"
        root.write_text("FORMAT\nMODE 1\n", encoding="utf-8")
        with pytest.raises(TimCanonicalizationError, match="FORMAT without a value"):
            flatten_tim(root)
        with pytest.raises(TimCanonicalizationError, match="FORMAT without a value"):
            discover_effective_tim_mode(root)
        with pytest.raises(TimCanonicalizationError, match="FORMAT without a value"):
            write_canonical_tim(
                root,
                pta_name="EPTA",
                timing_package="pint",
                out_path=tmp_path / "out.tim",
                par_text="PSRJ J0000+0000\n",
            )

    def test_parity_on_acceptance(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nMODE 1\n{_toa(58000.0)}\n", encoding="utf-8")
        assert discover_effective_tim_mode(root) == flatten_tim(root).effective_mode

    def test_tempo2_skip_omits_time_and_mode(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nSKIP\nTIME nope\nMODE 1\nNOSKIP\n{_toa(58000.0)}\n",
            encoding="utf-8",
        )
        result = flatten_tim(root, timing_package="tempo2")
        assert "TIME" not in result.text
        assert "MODE" not in result.text
        assert result.effective_mode is None
        assert _toa_lines(result.text)[0].split()[2] == "58000.0"

    def test_pint_child_to_parent_time_leak(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )
        lines = _toa_lines(flatten_tim(root, timing_package="pint").text)
        assert lines[0].split()[2] == _oracle_baked("58001.0", Fraction(1))
        assert lines[1].split()[2] == _oracle_baked("58002.0", Fraction(1))


class TestNamesArtifactAndHelpers:
    """§6.4 names, ensure_par_mode, JUMP on baked MJD, flatten idempotency."""

    def test_names_are_toa_digits_and_avoid_heuristics(self, tmp_path):
        # Princeton-style one-char name and a long Parkes-like name both rewrite.
        root = tmp_path / "root.tim"
        long_name = "a" * 27
        root.write_text(
            f"FORMAT 1\n a 1400.0 58000.0 1.0 g\n"
            f" {long_name} 1400.0 58001.0 1.0 g\n",
            encoding="utf-8",
        )
        lines = _toa_lines(flatten_tim(root).text)
        for line in lines:
            assert re.fullmatch(r"toa\d{5}", line.split()[0])
            assert not _pint_legacy_heuristic_hit(line)

    def test_flatten_idempotent(self, tmp_path):
        # Artifact has no MODE/TIME; re-flattening a MODE-free file is stable.
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME 1\n{_toa(58000.0)}\n", encoding="utf-8")
        once = flatten_tim(root)
        twice_path = tmp_path / "once.tim"
        twice_path.write_text(once.text, encoding="utf-8")
        twice = flatten_tim(twice_path)
        assert twice.text == once.text
        assert twice.effective_mode == once.effective_mode is None

    def test_comments_and_existing_to_flag_preserved(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\n# note\n{_toa(58000.0, ' -to -0.9e-6')}\n",
            encoding="utf-8",
        )
        text = flatten_tim(root).text
        assert "# note" in text
        assert "-to -0.9e-6" in text

    def test_jump_mjd_follows_baked_mjd(self, tmp_path):
        # Source MJD 57999.5 with TIME +86400 → 58000.5, inside [58000, 59000).
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME 86400\n{_toa(57999.5)}\n", encoding="utf-8")
        out = write_canonical_tim(
            root,
            pta_name="EPTA",
            timing_package="tempo2",
            out_path=tmp_path / "out.tim",
            par_text="JUMP MJD 58000 59000 -1e-7 1\n",
        )
        assert "-mjd_jump_pta EPTA_1" in out.path.read_text(encoding="utf-8")

    def test_ensure_par_mode_drops_weight_appends_last(self):
        par = "PSRJ J0000+0000\nWEIGHT 1\nF0 1\nMODE 0\nF1 -1e-15\n"
        out = ensure_par_mode(par, 1)
        lines = out.splitlines()
        assert lines[-1] == "MODE 1"
        assert (
            sum(1 for line in lines if line.split() and line.split()[0] == "MODE") == 1
        )
        assert not any(line.split() and line.split()[0] == "WEIGHT" for line in lines)
        assert ensure_par_mode(out, 1) == out
        assert out.endswith("\n") == par.endswith("\n")

    def test_write_pn_tim_libstempo_omits_mode(self):
        """§6.4.30: ancillary PN writer must not emit MODE."""
        import inspect

        from metapulsar.pint_helpers import _write_pn_tim_libstempo

        assert "MODE 1" not in inspect.getsource(_write_pn_tim_libstempo)
