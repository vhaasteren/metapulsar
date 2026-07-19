"""Integration tests: MockLibstempo -> MetaPulsar pipeline."""

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


class TestMockLibstempoMetaPulsarIntegration:
    @pytest.fixture
    def two_pta_metapulsar(self):
        mock_lt1 = create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta1", seed=10
        )
        mock_lt2 = create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta2", seed=20
        )
        return MetaPulsar(
            {"pta1": mock_lt1, "pta2": mock_lt2},
            combination_strategy="per_pta",
        )

    def test_construction_succeeds(self, two_pta_metapulsar):
        mp = two_pta_metapulsar
        assert mp is not None
        assert len(mp._toas) == 60

    def test_offset_handled_automatically(self, two_pta_metapulsar):
        mp = two_pta_metapulsar
        offset_params = [p for p in mp.fitpars if "Offset" in p]
        assert len(offset_params) >= 1

    def test_design_matrix_shape_matches_fitpars(self, two_pta_metapulsar):
        mp = two_pta_metapulsar
        assert mp._designmatrix.shape == (60, len(mp.fitpars))

    def test_pta_slices_correct(self, two_pta_metapulsar):
        mp = two_pta_metapulsar
        slices = mp._get_pta_slices()
        assert slices["pta1"] == slice(0, 30)
        assert slices["pta2"] == slice(30, 60)

    def test_flags_include_pta_dataset(self, two_pta_metapulsar):
        mp = two_pta_metapulsar
        assert "pta_dataset" in mp._flags.dtype.names
        assert np.all(mp._flags["pta_dataset"][:30] == "pta1")
        assert np.all(mp._flags["pta_dataset"][30:] == "pta2")

    def test_consistent_strategy(self):
        mock_lt1 = create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta1", seed=10
        )
        mock_lt2 = create_mock_libstempo(
            n_toas=30, name="J1857+0943", telescope="pta2", seed=20
        )
        mp = MetaPulsar(
            {"pta1": mock_lt1, "pta2": mock_lt2},
            combination_strategy="shared",
        )
        assert len(mp._toas) == 60
        assert mp._designmatrix.shape[1] == len(mp.fitpars)
