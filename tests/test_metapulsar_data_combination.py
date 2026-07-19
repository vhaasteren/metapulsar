"""Tests for MetaPulsar data combination functionality."""

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


class TestMetaPulsarDataCombination:
    @pytest.fixture
    def mock_pulsars(self):
        return {
            "test_pta1": create_mock_libstempo(
                n_toas=50,
                name="J1857+0943",
                telescope="test_pta1",
                include_astrometry=True,
                include_spin=True,
                seed=10,
            ),
            "test_pta2": create_mock_libstempo(
                n_toas=50,
                name="J1857+0943",
                telescope="test_pta2",
                include_astrometry=True,
                include_spin=True,
                seed=20,
            ),
        }

    def test_timing_data_combination_basic(self, mock_pulsars):
        metapulsar = MetaPulsar(mock_pulsars, combination_strategy="per_pta")
        assert len(metapulsar._toas) == 100
        assert len(metapulsar._residuals) == 100
        assert len(metapulsar._toaerrs) == 100
        assert len(metapulsar._ssbfreqs) == 100
        assert len(metapulsar._telescope) == 100
        assert isinstance(metapulsar._toas, np.ndarray)
        assert isinstance(metapulsar._residuals, np.ndarray)
        assert isinstance(metapulsar._toaerrs, np.ndarray)
        assert isinstance(metapulsar._ssbfreqs, np.ndarray)
        assert isinstance(metapulsar._telescope, np.ndarray)

    def test_flag_combination(self, mock_pulsars):
        metapulsar = MetaPulsar(mock_pulsars, combination_strategy="per_pta")
        assert isinstance(metapulsar._flags, np.ndarray)
        assert metapulsar._flags.dtype.names is not None
        assert "telescope" in metapulsar._flags.dtype.names
        assert "backend" in metapulsar._flags.dtype.names
        assert "pta_dataset" in metapulsar._flags.dtype.names
        assert "timing_package" in metapulsar._flags.dtype.names
        assert "pta" in metapulsar._flags.dtype.names
        assert len(metapulsar._flags) == 100
        assert np.all(metapulsar._flags["pta_dataset"][:50] == "test_pta1")
        assert np.all(metapulsar._flags["pta_dataset"][50:] == "test_pta2")

    def test_pta_slice_calculation(self, mock_pulsars):
        metapulsar = MetaPulsar(mock_pulsars, combination_strategy="per_pta")
        slices = metapulsar._get_pta_slices()
        assert "test_pta1" in slices
        assert "test_pta2" in slices
        assert slices["test_pta1"] == slice(0, 50)
        assert slices["test_pta2"] == slice(50, 100)

    def test_timing_data_combination_empty_pulsars(self):
        with pytest.raises(StopIteration):
            MetaPulsar({}, combination_strategy="per_pta")

    def test_timing_data_combination_single_pulsar(self):
        mock_psr = create_mock_libstempo(
            n_toas=25,
            name="J1857+0943",
            telescope="single_pta",
            include_astrometry=True,
            include_spin=True,
            seed=7,
        )
        metapulsar = MetaPulsar(
            {"single_pta": mock_psr}, combination_strategy="per_pta"
        )
        assert len(metapulsar._toas) == 25
        assert len(metapulsar._residuals) == 25

    def test_timing_data_combination_different_sizes(self):
        pulsars = {
            "small_pta": create_mock_libstempo(
                n_toas=30, name="J1857+0943", telescope="small_pta", seed=1
            ),
            "large_pta": create_mock_libstempo(
                n_toas=70, name="J1857+0943", telescope="large_pta", seed=2
            ),
        }
        metapulsar = MetaPulsar(pulsars, combination_strategy="per_pta")
        assert len(metapulsar._toas) == 100
        slices = metapulsar._get_pta_slices()
        assert slices["small_pta"] == slice(0, 30)
        assert slices["large_pta"] == slice(30, 100)
