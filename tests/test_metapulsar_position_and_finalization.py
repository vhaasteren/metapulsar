"""
Tests for MetaPulsar position setup and finalization functionality.

This module tests the position setup, from_files class method, and validation
methods implemented in the MetaPulsar class.
"""

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


class TestMetaPulsarPositionAndFinalization:
    """Test class for MetaPulsar position setup and finalization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pulsars = {
            "test_pta1": create_mock_libstempo(
                n_toas=30, name="J1857+0943", telescope="test_pta1", seed=10
            ),
            "test_pta2": create_mock_libstempo(
                n_toas=30, name="J1857+0943", telescope="test_pta2", seed=20
            ),
        }
        self.metapulsar = MetaPulsar(self.pulsars, combination_strategy="per_pta")

    def test_setup_position_and_planets_basic(self):
        """Test basic position and planetary data setup."""
        # Check that position attributes are set
        assert hasattr(self.metapulsar, "_raj")
        assert hasattr(self.metapulsar, "_decj")
        assert hasattr(self.metapulsar, "_pos")
        assert hasattr(self.metapulsar, "_pos_t")
        assert hasattr(self.metapulsar, "_planetssb")
        assert hasattr(self.metapulsar, "_sunssb")
        assert hasattr(self.metapulsar, "_pdist")

        # Check position values
        assert isinstance(self.metapulsar._raj, (int, float))
        assert isinstance(self.metapulsar._decj, (int, float))
        assert isinstance(self.metapulsar._pos, np.ndarray)
        assert isinstance(self.metapulsar._pos_t, np.ndarray)

    def test_setup_position_and_planets_shape(self):
        """Test that position arrays have correct shapes."""
        n_toas = len(self.metapulsar._toas)

        # _pos is the sky unit vector from the reference pulsar: shape (3,)
        assert self.metapulsar._pos.shape == (3,)
        # _pos_t is the per-TOA position array: shape (n_toas, 3)
        assert self.metapulsar._pos_t.shape == (n_toas, 3)

    def test_setup_position_and_planets_empty_pulsars(self):
        """Test position setup with empty pulsar list."""
        # Empty pulsars should raise an exception
        with pytest.raises(StopIteration):
            MetaPulsar({}, combination_strategy="per_pta")

    def test_validate_consistency_success(self):
        """Test successful consistency validation."""
        pulsar_name = self.metapulsar.validate_consistency()
        assert pulsar_name == "J1857+0943"

    def test_validate_consistency_different_pulsars(self):
        """Test consistency validation with different pulsars."""
        inconsistent_mp = MetaPulsar(
            {
                "pta1": create_mock_libstempo(
                    n_toas=10, name="J1857+0943", telescope="test_pta1", seed=1
                ),
                "pta2": create_mock_libstempo(
                    n_toas=10, name="J1900+0000", telescope="test_pta2", seed=2
                ),
            },
            combination_strategy="per_pta",
        )

        with pytest.raises(ValueError, match="Not all the same pulsar"):
            inconsistent_mp.validate_consistency()

    def test_validate_consistency_no_pulsars(self):
        """Test consistency validation with no pulsars."""
        # Empty pulsars should raise an exception during construction
        with pytest.raises(StopIteration):
            MetaPulsar({}, combination_strategy="per_pta")

    def test_validate_consistency_no_epulsars(self):
        """Test consistency validation before Enterprise Pulsars are created."""
        # Create MetaPulsar but don't initialize it
        mp = MetaPulsar.__new__(MetaPulsar)
        mp._epulsars = None

        with pytest.raises(ValueError, match="No Enterprise Pulsars created yet"):
            mp.validate_consistency()

    def test_position_attributes_consistency(self):
        """Test that _pos is the reference pulsar's sky unit vector and _pos_t is per-PTA."""
        ref_psr = next(iter(self.metapulsar._epulsars.values()))

        # _pos should be the reference pulsar's sky unit vector
        np.testing.assert_array_equal(self.metapulsar._pos, ref_psr._pos)

        # _pos_t should be filled per-PTA from each pulsar's _pos_t
        pta_slices = self.metapulsar._get_pta_slices()
        for pta, psr in self.metapulsar._epulsars.items():
            np.testing.assert_array_almost_equal(
                self.metapulsar._pos_t[pta_slices[pta], :], psr._pos_t
            )

    def test_planetary_data_setup(self):
        """Test that planetary data is properly set up per-PTA."""
        ref_psr = next(iter(self.metapulsar._epulsars.values()))

        # _pdist is still taken directly from the reference pulsar
        assert self.metapulsar._pdist is ref_psr._pdist

        # _planetssb and _sunssb are now per-PTA sliced arrays
        pta_slices = self.metapulsar._get_pta_slices()
        for pta, psr in self.metapulsar._epulsars.items():
            np.testing.assert_array_equal(
                self.metapulsar._planetssb[pta_slices[pta], :, :], psr._planetssb
            )
            np.testing.assert_array_equal(
                self.metapulsar._sunssb[pta_slices[pta], :], psr._sunssb
            )

    def test_position_coordinates(self):
        """Test that position coordinates are properly set."""
        ref_psr = next(iter(self.metapulsar._epulsars.values()))

        # RA and Dec should match reference pulsar
        assert self.metapulsar._raj == ref_psr._raj
        assert self.metapulsar._decj == ref_psr._decj

    def test_bj_name_generation(self):
        """Test that B/J name generation is called."""
        # This test verifies that the bj_name_from_pulsar function is called
        # The actual name generation is tested in position_helpers tests
        # Here we just verify the method doesn't crash
        self.metapulsar._setup_position_and_planets()
        # If we get here without error, the B/J name generation worked

    def test_position_and_finalization_integration(self):
        """Test that all position setup and finalization methods work together."""
        # Test that position setup works with validation
        pulsar_name = self.metapulsar.validate_consistency()
        assert pulsar_name == "J1857+0943"

        # Test that position attributes are properly set
        assert hasattr(self.metapulsar, "_raj")
        assert hasattr(self.metapulsar, "_decj")
        assert hasattr(self.metapulsar, "_pos")

    def test_validate_consistency_with_missing_names(self):
        """Test consistency validation with pulsars missing name attributes."""
        adapted_pulsar = create_mock_libstempo(
            n_toas=10, name="J1857+0943", telescope="test_pta", seed=1
        )
        delattr(adapted_pulsar, "name")

        # This should raise an AttributeError when trying to access the missing name
        with pytest.raises(AttributeError):
            MetaPulsar({"test_pta": adapted_pulsar}, combination_strategy="per_pta")

    def test_all_equal_helper_method(self):
        """Test the _all_equal helper method."""
        # Test with equal values
        assert self.metapulsar._all_equal([1, 1, 1, 1])
        assert self.metapulsar._all_equal(["a", "a", "a"])
        assert self.metapulsar._all_equal([])  # Empty list

        # Test with different values
        assert not self.metapulsar._all_equal([1, 2, 3])
        assert not self.metapulsar._all_equal(["a", "b", "c"])
        assert not self.metapulsar._all_equal([1, 1, 2])
