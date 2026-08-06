"""Unit tests for TimFileAnalyzer class."""

import pytest
from pathlib import Path

from metapulsar.tim_file_analyzer import TimFileAnalyzer, TimMetadata


def _tempo2_line(mjd: float, *, pn=None, extra_flags: str = "") -> str:
    """Create a FORMAT 1 TOA line (short, under 80 chars)."""
    base = f" obs1 1400.0 {mjd} 1.0 g"
    if pn is not None:
        base += f" -pn {pn}"
    if extra_flags:
        base += f" {extra_flags}"
    return base


class TestTimFileAnalyzer:
    """Test cases for TimFileAnalyzer.get_tim_metadata()."""

    def setup_method(self):
        self.analyzer = TimFileAnalyzer()
        self.test_data_dir = Path("tests/data/tim_files")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        if self.test_data_dir.exists():
            for file in self.test_data_dir.glob("*.tim"):
                file.unlink()
            self.test_data_dir.rmdir()

    def _create_test_tim_file(self, filename: str, content: str) -> Path:
        file_path = self.test_data_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_basic_timespan_and_count(self):
        content = f"""FORMAT 1
{_tempo2_line(55087.1109722889085)}
{_tempo2_line(55090.1109722889085)}
{_tempo2_line(55093.1109722889085)}
"""
        tim_file = self._create_test_tim_file("basic.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.toa_count == 3
        assert meta.timespan_days == pytest.approx(6.0)
        assert meta.mjd_min == pytest.approx(55087.1109722889085)
        assert meta.mjd_max == pytest.approx(55093.1109722889085)
        assert meta.pn_status == "none"
        assert meta.pn_without_count == 3

    def test_empty_file(self):
        tim_file = self._create_test_tim_file("empty.tim", "")
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.toa_count == 0
        assert meta.timespan_days == 0.0
        assert meta.mjd_min is None
        assert meta.pn_status == "none"

    def test_single_toa_zero_timespan(self):
        content = f"FORMAT 1\n{_tempo2_line(55087.1109722889085)}\n"
        tim_file = self._create_test_tim_file("single_toa.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.toa_count == 1
        assert meta.timespan_days == 0.0

    def test_missing_file(self):
        missing_file = self.test_data_dir / "missing.tim"
        meta = self.analyzer.get_tim_metadata(missing_file)

        assert meta.toa_count == 0
        assert meta.timespan_days == 0.0

    def test_short_format1_lines(self):
        """Short FORMAT 1 lines (<80 chars) must be counted as TOAs."""
        content = (
            "FORMAT 1\n" " obs1 1400.0 58000.0 1.0 g\n" " obs2 1400.0 58001.0 1.0 g\n"
        )
        tim_file = self._create_test_tim_file("short_format1.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.toa_count == 2
        assert meta.timespan_days == pytest.approx(1.0)

    def test_pn_complete(self):
        content = (
            "FORMAT 1\n"
            f"{_tempo2_line(58000.0, pn=0)}\n"
            f"{_tempo2_line(58001.0, pn=1)}\n"
        )
        tim_file = self._create_test_tim_file("pn_complete.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.pn_status == "complete"
        assert meta.pn_with_count == 2
        assert meta.pn_without_count == 0

    def test_pn_none(self):
        content = "FORMAT 1\n" f"{_tempo2_line(58000.0)}\n" f"{_tempo2_line(58001.0)}\n"
        tim_file = self._create_test_tim_file("pn_none.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.pn_status == "none"
        assert meta.pn_with_count == 0
        assert meta.pn_without_count == 2

    def test_pn_mixed(self):
        content = (
            "FORMAT 1\n" f"{_tempo2_line(58000.0, pn=0)}\n" f"{_tempo2_line(58001.0)}\n"
        )
        tim_file = self._create_test_tim_file("pn_mixed.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.pn_status == "mixed"
        assert meta.pn_with_count == 1
        assert meta.pn_without_count == 1

    def test_long_line_with_extra_flags(self):
        content = (
            "FORMAT 1\n"
            " c036915.align.pazr.30min 1345.999 55087.1109722889085 2.890 g "
            "-flag1 value1 -pn 42 -group foo\n"
        )
        tim_file = self._create_test_tim_file("long_flags.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)

        assert meta.toa_count == 1
        assert meta.pn_status == "complete"

    def test_include_single_file(self):
        main_content = f"""FORMAT 1
INCLUDE included.tim
{_tempo2_line(55087.1109722889085)}
"""
        included_content = f"FORMAT 1\n{_tempo2_line(55090.1109722889085)}\n"
        main_file = self._create_test_tim_file("main.tim", main_content)
        self._create_test_tim_file("included.tim", included_content)

        meta = self.analyzer.get_tim_metadata(main_file)
        assert meta.toa_count == 2
        assert meta.timespan_days == pytest.approx(3.0)

    def test_include_multiple_files(self):
        main_content = f"""FORMAT 1
INCLUDE file1.tim
INCLUDE file2.tim
{_tempo2_line(55087.1109722889085)}
"""
        self._create_test_tim_file(
            "file1.tim", f"FORMAT 1\n{_tempo2_line(55090.1109722889085)}\n"
        )
        self._create_test_tim_file(
            "file2.tim", f"FORMAT 1\n{_tempo2_line(55093.1109722889085)}\n"
        )
        main_file = self._create_test_tim_file("main_multi.tim", main_content)

        meta = self.analyzer.get_tim_metadata(main_file)
        assert meta.toa_count == 3
        assert meta.timespan_days == pytest.approx(6.0)

    def test_include_missing_file(self):
        main_content = (
            f"FORMAT 1\nINCLUDE missing.tim\n{_tempo2_line(55087.1109722889085)}\n"
        )
        main_file = self._create_test_tim_file("main_missing.tim", main_content)
        meta = self.analyzer.get_tim_metadata(main_file)

        assert meta.toa_count == 1
        assert meta.timespan_days == 0.0
        assert any("not found" in w for w in meta.parse_warnings)

    def test_include_circular_reference(self):
        file_a_content = (
            f"FORMAT 1\nINCLUDE file_b.tim\n{_tempo2_line(55087.1109722889085)}\n"
        )
        file_b_content = (
            f"FORMAT 1\nINCLUDE file_a.tim\n{_tempo2_line(55090.1109722889085)}\n"
        )
        file_a = self._create_test_tim_file("file_a.tim", file_a_content)
        self._create_test_tim_file("file_b.tim", file_b_content)

        meta = self.analyzer.get_tim_metadata(file_a)
        assert meta.toa_count >= 1
        assert meta.timespan_days >= 0.0
        assert any("Circular INCLUDE" in w for w in meta.parse_warnings)

    def test_include_same_file_twice(self):
        """Repeated INCLUDE of the same file counts TOAs each time."""
        chunk_content = f"FORMAT 1\n{_tempo2_line(55090.1109722889085)}\n"
        self._create_test_tim_file("chunk.tim", chunk_content)
        main_content = f"""FORMAT 1
INCLUDE chunk.tim
{_tempo2_line(55087.1109722889085)}
INCLUDE chunk.tim
"""
        main_file = self._create_test_tim_file("repeat_include.tim", main_content)
        meta = self.analyzer.get_tim_metadata(main_file)

        assert meta.toa_count == 3
        assert meta.timespan_days == pytest.approx(3.0)
        assert not any("Circular INCLUDE" in w for w in meta.parse_warnings)

    def test_include_nested(self):
        file_a_content = f"FORMAT 1\nINCLUDE file_b_nested.tim\n{_tempo2_line(55087.1109722889085)}\n"
        file_b_content = f"FORMAT 1\nINCLUDE file_c_nested.tim\n{_tempo2_line(55090.1109722889085)}\n"
        file_c_content = f"FORMAT 1\n{_tempo2_line(55093.1109722889085)}\n"
        file_a = self._create_test_tim_file("file_a_nested.tim", file_a_content)
        self._create_test_tim_file("file_b_nested.tim", file_b_content)
        self._create_test_tim_file("file_c_nested.tim", file_c_content)

        meta = self.analyzer.get_tim_metadata(file_a)
        assert meta.toa_count == 3
        assert meta.timespan_days == pytest.approx(6.0)

    def test_format_inheritance_in_include(self):
        """Included file without FORMAT inherits parent FORMAT 1 state."""
        main_content = f"FORMAT 1\nINCLUDE child.tim\n{_tempo2_line(58000.0)}\n"
        child_content = f"{_tempo2_line(58001.0)}\n"
        main_file = self._create_test_tim_file("inherit_main.tim", main_content)
        self._create_test_tim_file("child.tim", child_content)

        meta = self.analyzer.get_tim_metadata(main_file)
        assert meta.toa_count == 2

    def test_directives_skipped(self):
        content = f"""FORMAT 1
JUMP 55000 55001
TIME 55000
T2EFAC -sys foo 1.0
TNEQUAD -sys bar 2.0
{_tempo2_line(55087.1109722889085)}
{_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("directives.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)
        assert meta.toa_count == 2
        assert meta.timespan_days == pytest.approx(3.0)

    def test_malformed_lines_skipped(self):
        content = f"""FORMAT 1
{_tempo2_line(55087.1109722889085)}
corrupted line
{_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("corrupted.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)
        assert meta.toa_count == 2
        assert meta.lines_skipped >= 1

    def test_comments_only(self):
        content = """FORMAT 1
# comment
C another comment
JUMP 55000 55001
"""
        tim_file = self._create_test_tim_file("comments_only.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)
        assert meta.toa_count == 0

    def test_cc_comment_is_not_counted_as_a_toa(self):
        """A ``CC`` line read as data records a phantom TOA at its freq token."""
        content = f"""FORMAT 1
CC ?c062776.align.pazr.30min 1345.999 56522.2190165978952 4.381 g -group EFF
CC? c062799.align.pazr.30min 1354.224 56522.287303339192132 1.275 g
C? TIME -1
Cc055877.align.pazr.30min 2625.499 55995.8838774191826 5.166 g
CTHE 2007 TOAS may be wrong - the settings were not correct
   C indented comment
cc lowercase two-char marker
{_tempo2_line(55087.1109722889085)}
{_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cc_comments.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)
        assert meta.toa_count == 2
        assert meta.mjd_min == pytest.approx(55087.1109722889085)

    def test_bare_cr_mid_record_does_not_invent_a_toa(self, tmp_path):
        body = (
            b"FORMAT 1\n"
            + _tempo2_line(55087.1109722889085).encode()
            + b" -padd 0.1\r -group WSRT\n"
        )
        tim_file = tmp_path / "cr_mid.tim"
        tim_file.write_bytes(body)
        meta = self.analyzer.get_tim_metadata(tim_file)
        assert meta.toa_count == 1
        assert meta.mjd_min == pytest.approx(55087.1109722889085)

    def test_cache_hit_behavior(self):
        content = f"FORMAT 1\n{_tempo2_line(55087.1109722889085)}\n{_tempo2_line(55090.1109722889085)}\n"
        tim_file = self._create_test_tim_file("cache_test.tim", content)

        meta1 = self.analyzer.get_tim_metadata(tim_file)
        assert tim_file.resolve() in self.analyzer._file_cache

        meta2 = self.analyzer.get_tim_metadata(tim_file)
        assert meta1 == meta2

    def test_cache_clear_functionality(self):
        content = f"FORMAT 1\n{_tempo2_line(55087.1109722889085)}\n"
        tim_file = self._create_test_tim_file("cache_clear.tim", content)
        self.analyzer.get_tim_metadata(tim_file)
        assert tim_file.resolve() in self.analyzer._file_cache

        self.analyzer.clear_cache()
        assert len(self.analyzer._file_cache) == 0

        meta = self.analyzer.get_tim_metadata(tim_file)
        assert meta.toa_count == 1

    def test_metadata_is_frozen(self):
        content = f"FORMAT 1\n{_tempo2_line(58000.0)}\n"
        tim_file = self._create_test_tim_file("frozen.tim", content)
        meta = self.analyzer.get_tim_metadata(tim_file)
        assert isinstance(meta, TimMetadata)
        with pytest.raises(AttributeError):
            meta.toa_count = 99  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__])
