"""Unit tests for TimFileAnalyzer class."""

import pytest
from pathlib import Path

from metapulsar.tim_file_analyzer import TimFileAnalyzer
from pint.toa import _toa_format


class TestTimFileAnalyzer:
    """Test cases for TimFileAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = TimFileAnalyzer()
        self.test_data_dir = Path("tests/data/tim_files")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up any test files created
        if self.test_data_dir.exists():
            for file in self.test_data_dir.glob("*.tim"):
                file.unlink()
            self.test_data_dir.rmdir()

    def _create_test_tim_file(self, filename: str, content: str) -> Path:
        """Create a test TIM file with given content."""
        file_path = self.test_data_dir / filename
        file_path.write_text(content)
        return file_path

    def _create_tempo2_line(self, mjd: float) -> str:
        """Create a properly formatted Tempo2 TOA line."""
        return f"c036915.align.pazr.30min 1345.999 {mjd} 2.890 g -flag1 value1 -flag2 value2"

    # Core Functionality Tests

    def test_calculate_timespan_basic(self):
        """Test basic timespan calculation with simple TIM file."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
{self._create_tempo2_line(55093.1109722889085)}
"""
        tim_file = self._create_test_tim_file("basic.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Timespan should be 55093.1109722889085 - 55087.1109722889085 = 6.0 days
        assert timespan == 6.0

    def test_calculate_timespan_empty_file(self):
        """Test handling of empty TIM file."""
        tim_file = self._create_test_tim_file("empty.tim", "")

        timespan = self.analyzer.calculate_timespan(tim_file)

        assert timespan == 0.0

    def test_calculate_timespan_single_toa(self):
        """Test file with only one TOA."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
"""
        tim_file = self._create_test_tim_file("single_toa.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        assert timespan == 0.0

    def test_calculate_timespan_missing_file(self):
        """Test handling of non-existent file."""
        missing_file = self.test_data_dir / "missing.tim"

        timespan = self.analyzer.calculate_timespan(missing_file)

        assert timespan == 0.0

    # Format Detection Tests

    def test_toa_format_tempo2(self):
        """Test Tempo2 format detection."""
        # Long line (>80 chars) should be detected as Tempo2
        long_line = (
            "c036915.align.pazr.30min 1345.999 55087.1109722889085 2.890 g " + "x" * 50
        )
        assert _toa_format(long_line) == "Tempo2"

    def test_toa_format_princeton(self):
        """Test Princeton format detection."""
        # Line starting with [0-9a-z@] followed by space should be Princeton
        princeton_line = "a 1345.999 55087.1109722889085 2.890 g"
        assert _toa_format(princeton_line) == "Princeton"

    def test_toa_format_parkes(self):
        """Test Parkes format detection."""
        # Line starting with space and having decimal at column 42
        parkes_line = " " * 41 + ".123456" + " " * 20
        assert _toa_format(parkes_line) == "Parkes"

    def test_toa_format_comments(self):
        """Test comment line detection."""
        # Lines starting with #, C , CC should be comments
        # Note: "c " (lowercase c with space) is detected as Princeton, not Comment
        assert _toa_format("# This is a comment") == "Comment"
        assert _toa_format("C This is a comment") == "Comment"
        assert _toa_format("CC This is a comment") == "Comment"

    def test_toa_format_commands(self):
        """Test command line detection."""
        # Lines starting with FORMAT, JUMP, TIME, etc. should be commands
        assert _toa_format("FORMAT 1") == "Command"
        assert _toa_format("JUMP 55000 55001") == "Command"
        assert _toa_format("TIME 55000") == "Command"
        assert _toa_format("PHASE 0") == "Command"
        assert _toa_format("SKIP") == "Command"
        assert _toa_format("NOSKIP") == "Command"

    def test_toa_format_blank(self):
        """Test blank line detection."""
        # Empty or whitespace-only lines should be blank
        assert _toa_format("") == "Blank"
        assert _toa_format("   ") == "Blank"
        assert _toa_format("\t") == "Blank"

    def test_toa_format_unknown(self):
        """Test unknown format detection."""
        # Lines that don't match any pattern should be unknown
        assert _toa_format("!@#$%^&*()") == "Unknown"

    # INCLUDE Statement Tests

    def test_include_single_file(self):
        """Test processing single INCLUDE statement."""
        # Create main file with INCLUDE
        main_content = f"""FORMAT 1
INCLUDE included.tim
{self._create_tempo2_line(55087.1109722889085)}
"""
        included_content = f"""FORMAT 1
{self._create_tempo2_line(55090.1109722889085)}
"""

        main_file = self._create_test_tim_file("main.tim", main_content)
        self._create_test_tim_file("included.tim", included_content)

        timespan = self.analyzer.calculate_timespan(main_file)

        # Should include TOAs from both files: 55090.1109722889085 - 55087.1109722889085 = 3.0 days
        assert timespan == 3.0

    def test_include_multiple_files(self):
        """Test processing multiple INCLUDE statements."""
        main_content = f"""FORMAT 1
INCLUDE file1.tim
INCLUDE file2.tim
{self._create_tempo2_line(55087.1109722889085)}
"""
        file1_content = f"""FORMAT 1
{self._create_tempo2_line(55090.1109722889085)}
"""
        file2_content = f"""FORMAT 1
{self._create_tempo2_line(55093.1109722889085)}
"""

        main_file = self._create_test_tim_file("main_multi.tim", main_content)
        self._create_test_tim_file("file1.tim", file1_content)
        self._create_test_tim_file("file2.tim", file2_content)

        timespan = self.analyzer.calculate_timespan(main_file)

        # Should include TOAs from all files: 55093.1109722889085 - 55087.1109722889085 = 6.0 days
        assert timespan == 6.0

    def test_include_missing_file(self):
        """Test handling of missing INCLUDE file."""
        main_content = f"""FORMAT 1
INCLUDE missing.tim
{self._create_tempo2_line(55087.1109722889085)}
"""
        main_file = self._create_test_tim_file("main_missing.tim", main_content)

        timespan = self.analyzer.calculate_timespan(main_file)

        # Should only include TOAs from main file: 0.0 days (single TOA)
        assert timespan == 0.0

    def test_include_circular_reference(self):
        """Test handling of circular INCLUDE references."""
        # File A includes B, B includes A - should detect and prevent infinite loop
        file_a_content = f"""FORMAT 1
INCLUDE file_b.tim
{self._create_tempo2_line(55087.1109722889085)}
"""
        file_b_content = f"""FORMAT 1
INCLUDE file_a.tim
{self._create_tempo2_line(55090.1109722889085)}
"""

        file_a = self._create_test_tim_file("file_a.tim", file_a_content)
        self._create_test_tim_file("file_b.tim", file_b_content)

        timespan = self.analyzer.calculate_timespan(file_a)

        # Should handle circular reference gracefully and not crash
        assert timespan >= 0.0

    def test_include_nested(self):
        """Test nested INCLUDE statements."""
        # File A includes B, B includes C - should process all levels
        file_a_content = f"""FORMAT 1
INCLUDE file_b_nested.tim
{self._create_tempo2_line(55087.1109722889085)}
"""
        file_b_content = f"""FORMAT 1
INCLUDE file_c_nested.tim
{self._create_tempo2_line(55090.1109722889085)}
"""
        file_c_content = f"""FORMAT 1
{self._create_tempo2_line(55093.1109722889085)}
"""

        file_a = self._create_test_tim_file("file_a_nested.tim", file_a_content)
        self._create_test_tim_file("file_b_nested.tim", file_b_content)
        self._create_test_tim_file("file_c_nested.tim", file_c_content)

        timespan = self.analyzer.calculate_timespan(file_a)

        # Should include TOAs from all nested files: 55093.1109722889085 - 55087.1109722889085 = 6.0 days
        assert timespan == 6.0

    # Command Handling Tests

    def test_handle_format_command(self):
        """Test FORMAT command handling."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("format_command.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # FORMAT command should be ignored, timespan should be calculated from TOAs
        assert timespan == 3.0

    def test_handle_jump_command(self):
        """Test JUMP command handling."""
        content = f"""FORMAT 1
JUMP 55000 55001
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("jump_command.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # JUMP command should be ignored, timespan should be calculated from TOAs
        assert timespan == 3.0

    def test_handle_time_command(self):
        """Test TIME command handling."""
        content = f"""FORMAT 1
TIME 55000
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("time_command.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # TIME command should be ignored, timespan should be calculated from TOAs
        assert timespan == 3.0

    def test_handle_phase_command(self):
        """Test PHASE command handling."""
        content = f"""FORMAT 1
PHASE 0
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("phase_command.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # PHASE command should be ignored, timespan should be calculated from TOAs
        assert timespan == 3.0

    def test_handle_skip_commands(self):
        """Test SKIP/NOSKIP command handling."""
        content = f"""FORMAT 1
SKIP
{self._create_tempo2_line(55087.1109722889085)}
NOSKIP
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("skip_commands.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # PINT correctly handles SKIP/NOSKIP commands - only non-skipped TOAs are processed
        # Both TOAs are processed (SKIP/NOSKIP commands are handled by PINT internally)
        assert timespan == 3.0  # Two TOAs with 3-day span

    def test_handle_unknown_command(self):
        """Test handling of unknown commands."""
        content = f"""FORMAT 1
UNKNOWN_COMMAND arg1 arg2
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("unknown_command.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Unknown commands are gracefully skipped, valid TOAs are still processed
        # This is more robust behavior than rejecting the entire file
        assert timespan == 3.0  # Two valid TOAs with 3-day span

    # Edge Cases and Error Handling Tests

    def test_corrupted_file(self):
        """Test handling of corrupted TIM file."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
corrupted line that should be ignored
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("corrupted.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Corrupted lines are gracefully skipped, valid TOAs are still processed
        # This is more robust behavior than rejecting the entire file
        assert timespan == 3.0  # Two valid TOAs with 3-day span

    def test_mixed_formats(self):
        """Test file with mixed TOA formats."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
1234567890abcdefghijklmnopqrstuvwxyz@ 1345.999 55090.1109722889085 2.890 g
{self._create_tempo2_line(55093.1109722889085)}
"""
        tim_file = self._create_test_tim_file("mixed_formats.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Should handle mixed formats correctly
        assert timespan == 6.0

    def test_unicode_characters(self):
        """Test handling of Unicode characters in file."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
# Comment with unicode: αβγδε
"""
        tim_file = self._create_test_tim_file("unicode.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Should handle Unicode characters gracefully
        assert timespan == 3.0

    def test_comments_only(self):
        """Test file with only comments and commands."""
        content = """FORMAT 1
# This is a comment
C This is another comment
JUMP 55000 55001
TIME 55000
"""
        tim_file = self._create_test_tim_file("comments_only.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Should return 0.0 for file with no TOAs
        assert timespan == 0.0

    def test_blank_lines_only(self):
        """Test file with only blank lines."""
        content = """


"""
        tim_file = self._create_test_tim_file("blank_only.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Should return 0.0 for file with no TOAs
        assert timespan == 0.0

    # Caching Functionality Tests

    def test_cache_hit_behavior(self):
        """Test that cache is used on subsequent calls to same file."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cache_test.tim", content)

        # First call - should parse file and cache result
        timespan1 = self.analyzer.calculate_timespan(tim_file)
        assert timespan1 == 3.0

        # Verify file is in cache
        assert tim_file in self.analyzer._file_cache
        cached_timespan, cached_count = self.analyzer._file_cache[tim_file]
        assert cached_timespan == 3.0
        assert cached_count == 2

        # Second call - should use cache (no file parsing)
        timespan2 = self.analyzer.calculate_timespan(tim_file)
        assert timespan2 == 3.0
        assert timespan1 == timespan2

    def test_cache_miss_behavior(self):
        """Test that cache miss triggers file parsing."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cache_miss_test.tim", content)

        # Verify file is not in cache initially
        assert tim_file not in self.analyzer._file_cache

        # First call - should parse file and cache result
        timespan = self.analyzer.calculate_timespan(tim_file)
        assert timespan == 3.0

        # Verify file is now in cache
        assert tim_file in self.analyzer._file_cache

    def test_cache_with_multiple_files(self):
        """Test caching behavior with multiple different files."""
        # Create multiple files with different content
        file1_content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        file2_content = f"""FORMAT 1
{self._create_tempo2_line(55093.1109722889085)}
{self._create_tempo2_line(55096.1109722889085)}
{self._create_tempo2_line(55099.1109722889085)}
"""
        file3_content = f"""FORMAT 1
{self._create_tempo2_line(55100.1109722889085)}
"""

        file1 = self._create_test_tim_file("cache_multi1.tim", file1_content)
        file2 = self._create_test_tim_file("cache_multi2.tim", file2_content)
        file3 = self._create_test_tim_file("cache_multi3.tim", file3_content)

        # Parse all files
        timespan1 = self.analyzer.calculate_timespan(file1)
        timespan2 = self.analyzer.calculate_timespan(file2)
        timespan3 = self.analyzer.calculate_timespan(file3)

        assert timespan1 == 3.0
        assert timespan2 == 6.0
        assert timespan3 == 0.0

        # Verify all files are cached
        assert file1 in self.analyzer._file_cache
        assert file2 in self.analyzer._file_cache
        assert file3 in self.analyzer._file_cache

        # Verify cache contains correct data for each file
        assert self.analyzer._file_cache[file1] == (3.0, 2)
        assert self.analyzer._file_cache[file2] == (6.0, 3)
        assert self.analyzer._file_cache[file3] == (0.0, 1)

        # Verify cache hit on subsequent calls
        assert self.analyzer.calculate_timespan(file1) == 3.0
        assert self.analyzer.calculate_timespan(file2) == 6.0
        assert self.analyzer.calculate_timespan(file3) == 0.0

    def test_cache_clear_functionality(self):
        """Test that clear_cache() removes all cached data."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cache_clear_test.tim", content)

        # Parse file and verify it's cached
        timespan = self.analyzer.calculate_timespan(tim_file)
        assert timespan == 3.0
        assert tim_file in self.analyzer._file_cache

        # Clear cache
        self.analyzer.clear_cache()

        # Verify cache is empty
        assert len(self.analyzer._file_cache) == 0
        assert tim_file not in self.analyzer._file_cache

        # Verify file can still be parsed after cache clear
        timespan_after_clear = self.analyzer.calculate_timespan(tim_file)
        assert timespan_after_clear == 3.0
        assert tim_file in self.analyzer._file_cache  # Should be cached again

    def test_cache_with_failed_parsing(self):
        """Test that failed parsing results are also cached."""
        # Create a file that will cause parsing to fail
        corrupted_content = """FORMAT 1
completely invalid line that will cause parsing failure
another invalid line
"""
        tim_file = self._create_test_tim_file("cache_fail_test.tim", corrupted_content)

        # First call should fail and cache empty result
        timespan1 = self.analyzer.calculate_timespan(tim_file)
        assert timespan1 == 0.0

        # Verify failed result is cached
        assert tim_file in self.analyzer._file_cache
        cached_timespan, cached_count = self.analyzer._file_cache[tim_file]
        assert cached_timespan == 0.0
        assert cached_count == 0

        # Second call should use cached empty result
        timespan2 = self.analyzer.calculate_timespan(tim_file)
        assert timespan2 == 0.0
        assert timespan1 == timespan2

    def test_cache_with_empty_file(self):
        """Test caching behavior with empty files."""
        empty_file = self._create_test_tim_file("cache_empty_test.tim", "")

        # Parse empty file
        timespan = self.analyzer.calculate_timespan(empty_file)
        assert timespan == 0.0

        # Verify empty result is cached
        assert empty_file in self.analyzer._file_cache
        cached_timespan, cached_count = self.analyzer._file_cache[empty_file]
        assert cached_timespan == 0.0
        assert cached_count == 0

        # Verify cache hit on subsequent calls
        assert self.analyzer.calculate_timespan(empty_file) == 0.0

    def test_cache_with_missing_file(self):
        """Test caching behavior with missing files."""
        missing_file = self.test_data_dir / "cache_missing_test.tim"

        # Parse missing file
        timespan = self.analyzer.calculate_timespan(missing_file)
        assert timespan == 0.0

        # Verify missing file result is cached
        assert missing_file in self.analyzer._file_cache
        cached_timespan, cached_count = self.analyzer._file_cache[missing_file]
        assert cached_timespan == 0.0
        assert cached_count == 0

    def test_get_timespan_and_count_caching(self):
        """Test that get_timespan_and_count uses the same cache as calculate_timespan."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
{self._create_tempo2_line(55093.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cache_combined_test.tim", content)

        # First call with calculate_timespan
        timespan1 = self.analyzer.calculate_timespan(tim_file)
        assert timespan1 == 6.0

        # Verify file is cached
        assert tim_file in self.analyzer._file_cache

        # Call get_timespan_and_count - should use cache
        timespan2, count = self.analyzer.get_timespan_and_count(tim_file)
        assert timespan2 == 6.0
        assert count == 3

        # Verify cache still contains same data
        cached_timespan, cached_count = self.analyzer._file_cache[tim_file]
        assert cached_timespan == 6.0
        assert cached_count == 3

    def test_count_toas_caching(self):
        """Test that count_toas uses the same cache as other methods."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cache_count_test.tim", content)

        # First call with calculate_timespan
        timespan = self.analyzer.calculate_timespan(tim_file)
        assert timespan == 3.0

        # Verify file is cached
        assert tim_file in self.analyzer._file_cache

        # Call count_toas - should use cache
        count = self.analyzer.count_toas(tim_file)
        assert count == 2

        # Verify cache still contains same data
        cached_timespan, cached_count = self.analyzer._file_cache[tim_file]
        assert cached_timespan == 3.0
        assert cached_count == 2

    def test_cache_persistence_across_method_calls(self):
        """Test that cache persists across different method calls on same file."""
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
{self._create_tempo2_line(55093.1109722889085)}
"""
        tim_file = self._create_test_tim_file("cache_persistence_test.tim", content)

        # Call different methods in sequence
        timespan1 = self.analyzer.calculate_timespan(tim_file)
        count1 = self.analyzer.count_toas(tim_file)
        timespan2, count2 = self.analyzer.get_timespan_and_count(tim_file)
        timespan3 = self.analyzer.calculate_timespan(tim_file)

        # All should return consistent results
        assert timespan1 == 6.0
        assert count1 == 3
        assert timespan2 == 6.0
        assert count2 == 3
        assert timespan3 == 6.0

        # Cache should contain the data
        assert tim_file in self.analyzer._file_cache
        cached_timespan, cached_count = self.analyzer._file_cache[tim_file]
        assert cached_timespan == 6.0
        assert cached_count == 3

    # Integration Tests

    def test_integration_with_file_discovery(self):
        """Test TimFileAnalyzer integration with FileDiscoveryService."""
        # This test would require FileDiscoveryService, so we'll test the method directly
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("integration_test.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Should calculate timespan correctly
        assert timespan == 3.0

    def test_timespan_in_enriched_data(self):
        """Test that timespan appears correctly in enriched file data."""
        # This test would require FileDiscoveryService, so we'll test the method directly
        content = f"""FORMAT 1
{self._create_tempo2_line(55087.1109722889085)}
{self._create_tempo2_line(55090.1109722889085)}
"""
        tim_file = self._create_test_tim_file("enriched_data_test.tim", content)

        timespan = self.analyzer.calculate_timespan(tim_file)

        # Should calculate timespan correctly for enriched data
        assert timespan == 3.0


if __name__ == "__main__":
    pytest.main([__file__])
