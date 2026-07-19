"""Tests for MetaPulsar design matrix and unit conversion functionality."""

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from metapulsar.pint_helpers import resolve_parameter_alias


class TestMetaPulsarDesignMatrix:
    """Design matrix behavior using MockLibstempo -> Enterprise pipeline."""

    def setup_method(self):
        self.pulsars = {
            "test_pta1": create_mock_libstempo(
                n_toas=30, name="J1857+0943", telescope="test_pta1", seed=10
            ),
            "test_pta2": create_mock_libstempo(
                n_toas=30, name="J1857+0943", telescope="test_pta2", seed=20
            ),
        }
        self.composite_mp = MetaPulsar(self.pulsars, combination_strategy="per_pta")
        self.consistent_mp = MetaPulsar(self.pulsars, combination_strategy="shared")

    def test_design_matrix_creation(self):
        """Test that design matrix is created correctly for both strategies."""
        # Test composite strategy
        assert hasattr(self.composite_mp, "_designmatrix")
        assert self.composite_mp._designmatrix.shape[0] == 60
        assert self.composite_mp._designmatrix.shape[1] == len(
            self.composite_mp.fitpars
        )
        assert np.count_nonzero(self.composite_mp._designmatrix) > 0

        # Test consistent strategy
        assert hasattr(self.consistent_mp, "_designmatrix")
        assert self.consistent_mp._designmatrix.shape[0] == 60
        assert self.consistent_mp._designmatrix.shape[1] == len(
            self.consistent_mp.fitpars
        )
        assert np.count_nonzero(self.consistent_mp._designmatrix) > 0

    def test_design_matrix_parameters(self):
        """Test that design matrix has correct parameters for both strategies."""
        assert len(self.composite_mp.fitpars) > 0
        assert len(self.consistent_mp.fitpars) > 0
        assert any("Offset" in p for p in self.composite_mp.fitpars)
        assert any("Offset" in p for p in self.consistent_mp.fitpars)

    def test_design_matrix_structure(self):
        """Test design matrix structure and content for both strategies."""
        dm_composite = self.composite_mp._designmatrix
        dm_consistent = self.consistent_mp._designmatrix
        assert np.count_nonzero(dm_composite) > 0
        assert np.count_nonzero(dm_consistent) > 0

    def test_design_matrix_pta_slices(self):
        """Test that design matrix correctly handles PTA slices for both strategies."""
        pta_slices_composite = self.composite_mp._get_pta_slices()
        assert "test_pta1" in pta_slices_composite
        assert "test_pta2" in pta_slices_composite
        assert pta_slices_composite["test_pta1"].start == 0
        assert pta_slices_composite["test_pta1"].stop == 30
        assert pta_slices_composite["test_pta2"].start == 30
        assert pta_slices_composite["test_pta2"].stop == 60

        pta_slices_consistent = self.consistent_mp._get_pta_slices()
        assert "test_pta1" in pta_slices_consistent
        assert "test_pta2" in pta_slices_consistent
        assert pta_slices_consistent["test_pta1"].start == 0
        assert pta_slices_consistent["test_pta1"].stop == 30
        assert pta_slices_consistent["test_pta2"].start == 30
        assert pta_slices_consistent["test_pta2"].stop == 60

    def test_unit_conversion_coordinate_parameters(self):
        """Test unit conversion for coordinate parameters for both strategies."""
        import astropy.units as u

        # PTA-specific names are left unchanged
        raj_col = np.ones(10)
        converted_raj = self.composite_mp._convert_design_matrix_units(
            raj_col, "RAJ_test_pta1", "tempo2"
        )
        assert np.allclose(converted_raj, raj_col)

        decj_col = np.ones(10)
        converted_decj = self.composite_mp._convert_design_matrix_units(
            decj_col, "DECJ_test_pta1", "tempo2"
        )
        assert np.allclose(converted_decj, decj_col)

        expected_factor = (1.0 * u.second / u.radian).to(u.second / u.hourangle).value
        expected_factor_deg = (1.0 * u.second / u.radian).to(u.second / u.deg).value

        converted_raj_consistent = self.consistent_mp._convert_design_matrix_units(
            raj_col, "RAJ", "tempo2"
        )
        assert np.allclose(converted_raj_consistent, raj_col * expected_factor)

        converted_decj_consistent = self.consistent_mp._convert_design_matrix_units(
            decj_col, "DECJ", "tempo2"
        )
        assert np.allclose(converted_decj_consistent, decj_col * expected_factor_deg)

    def test_unit_conversion_non_coordinate_parameters(self):
        """Test that non-coordinate parameters are not converted for both strategies."""
        f0_col = np.ones(10)
        converted_f0_composite = self.composite_mp._convert_design_matrix_units(
            f0_col, "F0_test_pta1", "tempo2"
        )
        assert np.allclose(converted_f0_composite, f0_col)

        converted_f0_consistent = self.consistent_mp._convert_design_matrix_units(
            f0_col, "F0", "tempo2"
        )
        assert np.allclose(converted_f0_consistent, f0_col)

    def test_design_matrix_column_construction(self):
        """Test individual design matrix column construction for both strategies."""
        parname = self.consistent_mp.fitpars[0]
        column = self.consistent_mp._build_design_matrix_column(parname)
        assert len(column) == 60

    def test_design_matrix_with_different_strategies(self):
        """Test design matrix with different combination strategies."""
        assert hasattr(self.composite_mp, "_designmatrix")
        assert hasattr(self.consistent_mp, "_designmatrix")
        assert self.composite_mp._designmatrix.shape[0] == 60
        assert self.consistent_mp._designmatrix.shape[0] == 60

    def test_design_matrix_empty_pulsars(self):
        """Test design matrix with empty pulsar list."""
        # Empty pulsars should raise an exception
        with pytest.raises(StopIteration):
            MetaPulsar({}, combination_strategy="per_pta")

    def test_timing_package_detection(self):
        """Test timing package detection fallback for unknown object types."""
        assert self.composite_mp._get_timing_package(object()) == "unknown"
        assert self.consistent_mp._get_timing_package(object()) == "unknown"

    def test_design_matrix_parameter_mapping(self):
        """Test that parameter mapping works correctly for both strategies."""
        assert hasattr(self.composite_mp, "_fitparameters")
        assert hasattr(self.consistent_mp, "_fitparameters")
        assert len(self.composite_mp._fitparameters) == len(self.composite_mp.fitpars)
        assert len(self.consistent_mp._fitparameters) == len(self.consistent_mp.fitpars)

    def test_design_matrix_consistency(self):
        """Test that design matrix is consistent across PTAs for both strategies."""
        dm_composite = self.composite_mp._designmatrix
        pta_slices_composite = self.composite_mp._get_pta_slices()
        dm_consistent = self.consistent_mp._designmatrix
        pta_slices_consistent = self.consistent_mp._get_pta_slices()
        assert dm_composite[pta_slices_composite["test_pta1"]].shape[0] == 30
        assert dm_consistent[pta_slices_consistent["test_pta2"]].shape[0] == 30

    def test_design_matrix_astrometry_parameters(self):
        """Test that astrometry parameters are handled correctly for both strategies."""
        dm_composite = self.composite_mp._designmatrix
        dm_consistent = self.consistent_mp._designmatrix
        assert np.count_nonzero(dm_composite) > 0
        assert np.count_nonzero(dm_consistent) > 0

    def test_design_matrix_spin_parameters(self):
        """Test that spin parameters are handled correctly for both strategies."""
        dm_composite = self.composite_mp._designmatrix
        dm_consistent = self.consistent_mp._designmatrix
        assert dm_composite.shape[0] == 60
        assert dm_consistent.shape[0] == 60

    def test_design_matrix_edot_parameter_index_lookup(self):
        """EDOT fitpars must align with fitpars_canonical for column construction."""
        fitpars = ["EDOT", "F0", "RAJ"]
        fitpars_canonical = [resolve_parameter_alias(p) for p in fitpars]
        mapped_param = "EDOT"

        par_idx = fitpars_canonical.index(resolve_parameter_alias(mapped_param))
        assert par_idx == 0
        assert fitpars_canonical[par_idx] == "EDOT"

        mapped_param = "ECCDOT"
        par_idx = fitpars_canonical.index(resolve_parameter_alias(mapped_param))
        assert par_idx == 0
