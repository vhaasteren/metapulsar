"""Tests for MetaPulsar data combination functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar, PtaFiles
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

    def test_timing_data_combination_basic(self, mock_pulsars, mock_metapulsar):
        metapulsar = mock_metapulsar(mock_pulsars, combination_strategy="per_pta")
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

    def test_flag_combination(self, mock_pulsars, mock_metapulsar):
        metapulsar = mock_metapulsar(mock_pulsars, combination_strategy="per_pta")
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

    def test_flag_fallback_fills_only_missing_rows(self, mock_pulsars, mock_metapulsar):
        mock_pulsars["test_pta1"]._flag_dict["pta"] = np.array(
            ["release_pta", "", "release_pta"] + [""] * 47
        )

        metapulsar = mock_metapulsar(mock_pulsars, combination_strategy="per_pta")

        assert metapulsar._flags["pta"][0] == "release_pta"
        assert metapulsar._flags["pta"][1] == "test_pta1"
        assert metapulsar._flags["pta"][2] == "release_pta"

    def test_pta_slice_calculation(self, mock_pulsars, mock_metapulsar):
        metapulsar = mock_metapulsar(mock_pulsars, combination_strategy="per_pta")
        slices = metapulsar._get_pta_slices()
        assert "test_pta1" in slices
        assert "test_pta2" in slices
        assert slices["test_pta1"] == slice(0, 50)
        assert slices["test_pta2"] == slice(50, 100)

    def test_timing_data_combination_empty_pulsars(self):
        with pytest.raises(
            ValueError, match="MetaPulsar requires at least one PTA input"
        ):
            MetaPulsar({}, combination_strategy="per_pta")

    def test_timing_data_combination_single_pulsar(self, mock_metapulsar):
        mock_psr = create_mock_libstempo(
            n_toas=25,
            name="J1857+0943",
            telescope="single_pta",
            include_astrometry=True,
            include_spin=True,
            seed=7,
        )
        metapulsar = mock_metapulsar(
            {"single_pta": mock_psr}, combination_strategy="per_pta"
        )
        assert len(metapulsar._toas) == 25
        assert len(metapulsar._residuals) == 25

    def test_timing_data_combination_different_sizes(self, mock_metapulsar):
        pulsars = {
            "small_pta": create_mock_libstempo(
                n_toas=30, name="J1857+0943", telescope="small_pta", seed=1
            ),
            "large_pta": create_mock_libstempo(
                n_toas=70, name="J1857+0943", telescope="large_pta", seed=2
            ),
        }
        metapulsar = mock_metapulsar(pulsars, combination_strategy="per_pta")
        assert len(metapulsar._toas) == 100
        slices = metapulsar._get_pta_slices()
        assert slices["small_pta"] == slice(0, 30)
        assert slices["large_pta"] == slice(30, 100)

    def test_parameter_mapping_preserves_timing_package_identity(self):
        metapulsar = object.__new__(MetaPulsar)
        pint_model = MagicMock()
        tempo2_pulsar = MagicMock()
        metapulsar._unpack_pulsar_data = MagicMock(
            return_value=(
                {"pint_pta": pint_model},
                {},
                {"tempo2_pta": tempo2_pulsar},
            )
        )
        # Both engines now read one source: the retained per-PTA par content.
        metapulsar._parfile_content_for_pta = MagicMock(return_value="PSR J1857+0943\n")
        metapulsar._setup_canonical_parameters = MagicMock()
        metapulsar.combine_components = []
        metapulsar.add_dm_derivatives = False
        metapulsar.exclude_from_shared = ()

        mapping = SimpleNamespace(fitparameters={}, setparameters={})
        with patch("metapulsar.metapulsar.ParameterManager") as manager_class:
            manager_class.return_value.build_parameter_mappings.return_value = mapping
            metapulsar._setup_parameters()

        file_data = manager_class.call_args.kwargs["file_data"]
        assert file_data["pint_pta"]["timing_package"] == "pint"
        assert file_data["tempo2_pta"]["timing_package"] == "tempo2"

    def test_parameter_mapping_reads_retained_par_for_both_engines(self, tmp_path):
        """Retained session pars, not an as_parfile() re-serialization.

        ``MODE 1`` is the marker: PINT's writer drops it in its default dialect,
        so seeing the retained text proves no round trip ran.
        """
        retained = "PSR J1857+0943\nF0 186.494\nMODE 1\nTRACK -2\n"
        pint_par = tmp_path / "pint_pta.par"
        tempo2_par = tmp_path / "tempo2_pta.par"
        tim = tmp_path / "any.tim"
        for path in (pint_par, tempo2_par):
            path.write_text(retained)
        tim.write_text("FORMAT 1\n")

        metapulsar = object.__new__(MetaPulsar)
        pint_model = MagicMock()
        pint_model.as_parfile.return_value = "PSR J1857+0943\n"  # must NOT be used
        tempo2_pulsar = MagicMock()
        metapulsar._unpack_pulsar_data = MagicMock(
            return_value=({"pint_pta": pint_model}, {}, {"tempo2_pta": tempo2_pulsar})
        )
        metapulsar._pulsars = {
            "pint_pta": (pint_model, None),
            "tempo2_pta": tempo2_pulsar,
        }
        metapulsar._pta_files = {
            "pint_pta": PtaFiles(pint_par, tim, "pint"),
            "tempo2_pta": PtaFiles(tempo2_par, tim, "tempo2"),
        }
        metapulsar._setup_canonical_parameters = MagicMock()
        metapulsar.combine_components = []
        metapulsar.add_dm_derivatives = False
        metapulsar.exclude_from_shared = ()

        mapping = SimpleNamespace(fitparameters={}, setparameters={})
        with patch("metapulsar.metapulsar.ParameterManager") as manager_class:
            manager_class.return_value.build_parameter_mappings.return_value = mapping
            metapulsar._setup_parameters()

        file_data = manager_class.call_args.kwargs["file_data"]
        assert file_data["pint_pta"]["par_content"] == retained
        assert file_data["tempo2_pta"]["par_content"] == retained
        pint_model.as_parfile.assert_not_called()

    def test_construction_requires_pta_files_for_every_pta(self, mock_pulsars):
        """No retained par, no MetaPulsar: there is no second source to fall back to.

        This is what makes the tempo2 ``savepar()`` footgun structurally
        impossible -- the dump that could carry ``TZRMJD -nan`` is never taken.
        """
        with pytest.raises(ValueError, match="requires pta_files for every PTA"):
            MetaPulsar(mock_pulsars, combination_strategy="per_pta")

    def test_construction_rejects_partial_pta_files(self, mock_pulsars, tmp_path):
        """One entry per PTA -- a partial mapping names the PTAs it is missing."""
        from metapulsar.mockpulsar import write_mock_pta_files

        pta_files = write_mock_pta_files(mock_pulsars, tmp_path / "files")
        pta_files.pop("test_pta2")

        with pytest.raises(ValueError, match=r"missing: \['test_pta2'\]"):
            MetaPulsar(
                mock_pulsars,
                combination_strategy="per_pta",
                pta_files=pta_files,
            )

    def test_missing_retained_par_file_is_reported(self, mock_pulsars, tmp_path):
        """A retained par that vanished fails loudly, naming the path."""
        pulsar = object.__new__(MetaPulsar)
        pulsar._pta_files = {
            "test_pta1": PtaFiles(tmp_path / "gone.par", tmp_path / "x.tim", "tempo2")
        }
        with pytest.raises(FileNotFoundError, match="gone.par"):
            pulsar._parfile_content_for_pta("test_pta1")
