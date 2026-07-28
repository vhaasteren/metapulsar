"""Comprehensive unit tests for selection utilities.

This module tests the create_staggered_selection function with various
scenarios including basic flag selection, staggered selection, and frequency
filtering.
"""

import numpy as np

from metapulsar.selection_utils import create_staggered_selection


# Mock flags data (realistic values)
mock_flags = {
    "group": np.array(["ASP_430", "ASP_800", "ASP_430", "ASP_800", "ASP_430"]),
    "B": np.array(["1", "2", "1", "2", "1"]),
    "f": np.array(["GASP_430", "GASP_800", "GASP_430", "GASP_800", "GASP_430"]),
    "empty_flag": np.array(["ASP_430", "", "ASP_800", "", "ASP_1400"]),
}

# Mock frequency data
mock_freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])


class TestBasicFlagSelection:
    """Test basic flag selection functionality."""

    def test_single_flag_all_values(self):
        """Test selection with single flag, all values (None)"""
        sel_func = create_staggered_selection("efac", {"group": None})

        # Test with flags = {"group": ["ASP_430", "ASP_800", "ASP_430"]}
        flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)
        expected = {
            "efac_ASP_430": np.array([True, False, True]),
            "efac_ASP_800": np.array([False, True, False]),
        }

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_single_flag_specific_value(self):
        """Test selection with single flag, specific value"""
        sel_func = create_staggered_selection("efac", {"group": "ASP_430"})

        # Test with flags = {"group": ["ASP_430", "ASP_800", "ASP_430"]}
        flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)
        expected = {"efac_ASP_430": np.array([True, False, True])}

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_staggered_flag_all_values(self):
        """Test staggered selection with all values"""
        sel_func = create_staggered_selection("ecorr", {("group", "f"): None})

        # Test with flags = {"group": ["ASP_430", "ASP_800"], "f": ["GASP_430", "GASP_800"]}
        flags = {
            "group": np.array(["ASP_430", "ASP_800"]),
            "f": np.array(["GASP_430", "GASP_800"]),
        }
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        expected = {
            "ecorr_ASP_430": np.array([True, False]),
            "ecorr_ASP_800": np.array([False, True]),
        }

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_staggered_flag_specific_value(self):
        """Test staggered selection with specific value"""
        sel_func = create_staggered_selection("ecorr", {("group", "f"): "ASP_430"})

        # Test with flags = {"group": ["ASP_430", "ASP_800"], "f": ["GASP_430", "GASP_800"]}
        flags = {
            "group": np.array(["ASP_430", "ASP_800"]),
            "f": np.array(["GASP_430", "GASP_800"]),
        }
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        expected = {"ecorr_ASP_430": np.array([True, False])}

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_flag_values(self):
        """Test that empty string flag values are excluded"""
        sel_func = create_staggered_selection("efac", {"empty_flag": None})

        # Test with flags = {"empty_flag": ["ASP_430", "", "ASP_800"]}
        flags = {"empty_flag": np.array(["ASP_430", "", "ASP_800"])}
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)
        expected = {
            "efac_ASP_430": np.array([True, False, False]),
            "efac_ASP_800": np.array([False, False, True]),
        }

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_missing_flags(self):
        """Test behavior when required flags are missing"""
        sel_func = create_staggered_selection("ecorr", {("missing_flag", "f"): None})

        # Test with only "f" flag available
        flags = {"f": np.array(["GASP_430", "GASP_800"])}
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        expected = {
            "ecorr_GASP_430": np.array([True, False]),
            "ecorr_GASP_800": np.array([False, True]),
        }

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_no_matching_values(self):
        """Test when no flag values match criteria"""
        sel_func = create_staggered_selection("efac", {"group": "ASP_1400"})

        # Test with flags = {"group": ["ASP_430", "ASP_800"]}
        flags = {"group": np.array(["ASP_430", "ASP_800"])}
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        expected = {"efac_ASP_1400": np.array([False, False])}

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_empty_flag_criteria(self):
        """Test with empty flag criteria"""
        sel_func = create_staggered_selection("test", {})

        flags = {"group": np.array(["ASP_430", "ASP_800"])}
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestFrequencyFiltering:
    """Test frequency band filtering functionality."""

    def test_frequency_filtering(self):
        """Test frequency band filtering"""
        sel_func = create_staggered_selection(
            "band", {"group": None}, freq_range=(200, 400)
        )

        flags = {
            "group": np.array(["ASP_430", "ASP_800", "ASP_430", "ASP_800", "ASP_430"])
        }
        freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        result = sel_func(flags, freqs)

        # Should only select frequencies in range [200, 400)
        # ASP_430 appears at positions 0,2,4 but only position 2 (freq 300) is in range
        # ASP_800 appears at positions 1,3 but only position 1 (freq 200) is in range
        expected = {
            "band_ASP_430": np.array([False, False, True, False, False]),
            "band_ASP_800": np.array([False, True, False, False, False]),
        }

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_no_frequency_filtering(self):
        """Test behavior without frequency filtering"""
        sel_func = create_staggered_selection("efac", {"group": None})

        flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)
        expected = {
            "efac_ASP_430": np.array([True, False, True]),
            "efac_ASP_800": np.array([False, True, False]),
        }

        assert isinstance(result, dict)
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])


class TestStaggeredSelection:
    """Test staggered selection functionality specifically."""

    def test_staggered_primary_available(self):
        """Test staggered selection when primary flag is available"""
        sel_func = create_staggered_selection("ecorr", {("group", "f"): None})

        flags = {
            "group": np.array(["ASP_430", "ASP_800"]),
            "f": np.array(["GASP_430", "GASP_800"]),
        }
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        expected = {
            "ecorr_ASP_430": np.array([True, False]),
            "ecorr_ASP_800": np.array([False, True]),
        }

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_staggered_fallback_used(self):
        """Test staggered selection when fallback flag is used"""
        sel_func = create_staggered_selection("ecorr", {("missing_group", "f"): None})

        flags = {"f": np.array(["GASP_430", "GASP_800"])}
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        expected = {
            "ecorr_GASP_430": np.array([True, False]),
            "ecorr_GASP_800": np.array([False, True]),
        }

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_staggered_partial_primary_uses_fallback_per_toa(self):
        """Test per-TOA fallback when primary flag exists but is empty on some TOAs."""
        sel_func = create_staggered_selection("efeq", {("group", "f"): None})

        flags = {
            "group": np.array(["EPTA_BE", "", ""]),
            "f": np.array(["", "NG_BE", "PPTA_BE"]),
        }
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)
        expected = {
            "efeq_EPTA_BE": np.array([True, False, False]),
            "efeq_NG_BE": np.array([False, True, False]),
            "efeq_PPTA_BE": np.array([False, False, True]),
        }

        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

        combined = np.zeros(len(freqs), dtype=bool)
        for mask in result.values():
            combined |= mask
        np.testing.assert_array_equal(combined, np.ones(len(freqs), dtype=bool))

    def test_staggered_no_flags_available(self):
        """Test staggered selection when no flags are available"""
        sel_func = create_staggered_selection("ecorr", {("missing1", "missing2"): None})

        flags = {"other_flag": np.array(["value1", "value2"])}
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestReturnTypes:
    """Test that return types are correct."""

    def test_return_type_structure(self):
        """Test that return values have correct structure"""
        sel_func = create_staggered_selection("test", {"group": None})

        flags = {"group": np.array(["ASP_430", "ASP_800"])}
        freqs = np.array([100.0, 200.0])

        result = sel_func(flags, freqs)

        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(value, np.ndarray) for value in result.values())
        assert all(value.dtype == bool for value in result.values())
        assert all(len(value) == len(freqs) for value in result.values())

    def test_mask_coverage(self):
        """Test that masks provide complete coverage"""
        sel_func = create_staggered_selection("test", {"group": None})

        flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)

        # All masks should be boolean arrays of the same length as freqs
        for mask in result.values():
            assert len(mask) == len(freqs)
            assert mask.dtype == bool

        # Masks should be mutually exclusive and collectively exhaustive
        combined_mask = np.zeros(len(freqs), dtype=bool)
        for mask in result.values():
            combined_mask |= mask

        # Should cover all elements
        assert np.all(combined_mask)
