"""Tests for the canonical .tim writer (flattening + metadata stamping)."""

import pytest

from metapulsar.tim_canonical import (
    TimCanonicalizationError,
    TimIncludeScopeError,
    TimLegacyFormatError,
    flatten_tim,
    stamp_metadata_flags,
    write_canonical_tim,
)
from metapulsar.tim_file_analyzer import TimFileAnalyzer


def _toa(mjd, flags=""):
    return f" obs1 1400.0 {mjd} 1.0 g{flags}"


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

        lines = flatten_tim(root).splitlines()

        assert lines[0] == "FORMAT 1"
        assert "INCLUDE tims/chunk.tim" not in lines
        assert [line.split()[2] for line in lines[1:]] == [
            "58000.0",
            "58001.0",
            "58002.0",
        ]

    def test_preserves_comments_and_directives(self, tmp_path):
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nMODE 1\nC commented out TOA\n# hash comment\n"
            f"T2EFAC -sys foo 1.2\n{_toa(58000.0)}\n",
            encoding="utf-8",
        )

        text = flatten_tim(root)

        assert "MODE 1" in text
        assert "C commented out TOA" in text
        assert "# hash comment" in text
        assert "T2EFAC -sys foo 1.2" in text

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

        mjds = [line.split()[2] for line in flatten_tim(root).splitlines()[1:]]

        assert mjds == ["58001.0", "58002.0", "58001.0", "58002.0"]

    def test_preserves_toa_lines_verbatim(self, tmp_path):
        raw = " obs1  1400.00000   58000.1234567890123456   0.997  g  -sys x"
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\n{raw}\n", encoding="utf-8")

        assert raw in flatten_tim(root).splitlines()

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

        text = flatten_tim(root, timing_package="tempo2")

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

        text = flatten_tim(root, timing_package="pint")

        assert "END" in text
        assert "58001.0" in text
        assert "58002.0" not in text


class TestIncludeScopeGuard:
    """tempo2 scopes stateful directives per included file; PINT leaks them.

    Flattening imposes PINT's semantics, so unbalanced state must be refused.
    """

    def test_balanced_time_inside_include_flattens(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\nTIME 1\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        text = flatten_tim(root)

        assert "TIME -1" in text and "TIME 1" in text
        assert [line.split()[2] for line in text.splitlines() if "obs1" in line] == [
            "58001.0",
            "58002.0",
        ]

    def test_unbalanced_time_in_include_raises(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\nTIME -1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text(
            f"FORMAT 1\nINCLUDE chunk.tim\n{_toa(58002.0)}\n", encoding="utf-8"
        )

        with pytest.raises(TimIncludeScopeError, match="ends with TIME still active"):
            flatten_tim(root)

    def test_time_live_at_include_entry_raises(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nTIME -1\nINCLUDE chunk.tim\n", encoding="utf-8")

        with pytest.raises(TimIncludeScopeError, match="INCLUDE with TIME"):
            flatten_tim(root)

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

        assert "EFAC 1.3" in flatten_tim(root, timing_package="tempo2")

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

        assert "INCLUDE" not in flatten_tim(root, timing_package="tempo2")

    def test_pint_shared_state_can_cross_include(self, tmp_path):
        (tmp_path / "chunk.tim").write_text(
            f"FORMAT 1\n{_toa(58001.0)}\n", encoding="utf-8"
        )
        root = tmp_path / "root.tim"
        root.write_text("FORMAT 1\nTIME -1\nINCLUDE chunk.tim\n", encoding="utf-8")

        assert "TIME -1" in flatten_tim(root, timing_package="pint")

    def test_unbalanced_time_without_include_is_allowed(self, tmp_path):
        """A single-file tim has no boundary to cross, so no guard applies."""
        root = tmp_path / "root.tim"
        root.write_text(f"FORMAT 1\nTIME -1\n{_toa(58000.0)}\n", encoding="utf-8")

        assert "TIME -1" in flatten_tim(root)


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

        text = out.read_text(encoding="utf-8")
        assert "INCLUDE" not in text
        assert "-pta_orig EPTA" in text
        assert "-pta epta_dr2" in text

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
        after = TimFileAnalyzer().get_tim_metadata(out)

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
            once,
            pta_name="pta1",
            timing_package="pint",
            out_path=tmp_path / "twice.tim",
        )

        assert twice.read_text(encoding="utf-8") == once.read_text(encoding="utf-8")

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
            once,
            pta_name="pta2",
            timing_package="pint",
            out_path=tmp_path / "twice.tim",
        )

        text = twice.read_text(encoding="utf-8")
        assert "-pta_orig pta1" in text
        assert "-pta pta2" in text
        assert "-pta_dataset_orig pta1" in text
        assert "-timing_package_orig pint" in text
