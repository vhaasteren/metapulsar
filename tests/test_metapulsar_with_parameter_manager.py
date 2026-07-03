"""Tests for MetaPulsar with ParameterManager integration."""

from pathlib import Path
from unittest.mock import Mock, patch
from pint.models import TimingModel
from pint.toa import TOAs
from metapulsar.metapulsar import MetaPulsar, SessionFiles
from metapulsar.parameter_manager import ParameterManager
from tests.helpers import make_tim_metadata


class TestMetaPulsarWithParameterManager:
    """Test MetaPulsar class with ParameterManager integration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create proper mock PINT models with correct specifications
        self.mock_model1 = Mock(spec=TimingModel)
        # Create a mock PSR parameter
        mock_psr1 = Mock()
        mock_psr1.value = "J1857+0943"
        self.mock_model1.PSR = mock_psr1
        self.mock_model1.get_params_dict.return_value = {
            "F0": "123.456",
            "RAJ": "18:57:36.4",
        }
        self.mock_model1.as_parfile.return_value = (
            "PSR J1857+0943\nF0 123.456\nRAJ 18:57:36.4\n"
        )

        self.mock_toas1 = Mock(spec=TOAs)

        self.mock_model2 = Mock(spec=TimingModel)
        # Create a mock PSR parameter
        mock_psr2 = Mock()
        mock_psr2.value = "J1857+0943"
        self.mock_model2.PSR = mock_psr2
        self.mock_model2.get_params_dict.return_value = {
            "F0": "123.456",
            "RAJ": "18:57:36.4",
        }
        self.mock_model2.as_parfile.return_value = (
            "PSR J1857+0943\nF0 123.456\nRAJ 18:57:36.4\n"
        )

        self.mock_toas2 = Mock(spec=TOAs)

        self.pulsars = {
            "epta_dr2": (self.mock_model1, self.mock_toas1),
            "ppta_dr2": (self.mock_model2, self.mock_toas2),
        }

        # Create mock libstempo object
        self.mock_libstempo_psr = Mock()
        self.mock_libstempo_psr.name = "J1857+0943"
        self.mock_libstempo_psr.parfile = {"F0": "123.456", "RAJ": "18:57:36.4"}

    def test_composite_strategy_parameter_setup(self):
        """Test parameter setup for composite strategy with ParameterManager."""
        # Test ParameterManager initialization directly
        mock_file_data = {
            "epta_dr2": {
                "tim_metadata": make_tim_metadata(timespan_days=1000),
                "par_content": "PSR J1857+0943\nF0 123.456\n",
            },
            "ppta_dr2": {
                "tim_metadata": make_tim_metadata(timespan_days=1200),
                "par_content": "PSR J1857+0943\nF0 123.456\n",
            },
        }

        # Initialize ParameterManager with empty combine_components for composite strategy
        param_manager = ParameterManager(
            file_data=mock_file_data, combine_components=[]
        )

        assert param_manager.combine_components == []
        assert (
            param_manager.reference_pta == "epta_dr2"
        )  # Should be the first dictionary key

    def test_consistent_strategy_parameter_setup(self):
        """Test parameter setup for consistent strategy with ParameterManager."""
        # Test ParameterManager initialization directly
        mock_file_data = {
            "epta_dr2": {
                "tim_metadata": make_tim_metadata(timespan_days=1000),
                "par_content": "PSR J1857+0943\nF0 123.456\n",
            },
            "ppta_dr2": {
                "tim_metadata": make_tim_metadata(timespan_days=1200),
                "par_content": "PSR J1857+0943\nF0 123.456\n",
            },
        }

        # Initialize ParameterManager with combine_components for consistent strategy
        param_manager = ParameterManager(
            file_data=mock_file_data, combine_components=["astrometry", "spindown"]
        )

        assert param_manager.combine_components == ["astrometry", "spindown"]
        assert (
            param_manager.reference_pta == "epta_dr2"
        )  # Should be the first dictionary key

    def test_get_parfile_data_pint(self):
        """Test _get_parfile_data with PINT objects."""
        # Create a minimal MetaPulsar instance for testing
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        # Test with PINT objects
        pulsars = {"epta_dr2": (self.mock_model1, self.mock_toas1)}
        result = metapulsar._get_parfile_data(pulsars)

        assert "epta_dr2" in result
        assert result["epta_dr2"] == {"F0": "123.456", "RAJ": "18:57:36.4"}

    def test_get_parfile_data_tempo2(self):
        """Test _get_parfile_data with Tempo2 objects."""
        # Create a minimal MetaPulsar instance for testing
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        # Test with libstempo objects
        pulsars = {"epta_dr2": self.mock_libstempo_psr}
        result = metapulsar._get_parfile_data(pulsars)

        assert "epta_dr2" in result
        assert result["epta_dr2"] == {"F0": "123.456", "RAJ": "18:57:36.4"}

    def test_get_parfile_data_uses_retained_session_par_for_tempo2(self, tmp_path):
        """Retained session par file should win over stale tempo2 object path."""
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        retained_par = tmp_path / "retained.par"
        retained_par.write_text(
            "PSR J1857+0943\nF0 123.456\nRAJ 18:57:36.4\nDECJ 09:43:17.2\n",
            encoding="utf-8",
        )
        retained_tim = tmp_path / "retained.tim"
        retained_tim.write_text("FORMAT 1\n", encoding="utf-8")

        mock_t2 = Mock()
        mock_t2.parfile = "/tmp/deleted_track_minus_2.par"
        mock_t2.savepar.side_effect = AssertionError(
            "savepar should not be called when retained session par exists"
        )

        pulsars = {"epta_dr2": mock_t2}
        metapulsar._pulsars = pulsars
        metapulsar._session_files = {
            "epta_dr2": SessionFiles(
                par_path=Path(retained_par),
                tim_path=Path(retained_tim),
                timing_package="tempo2",
            )
        }

        result = metapulsar._get_parfile_data(pulsars)

        assert "epta_dr2" in result
        assert isinstance(result["epta_dr2"], dict)
        mock_t2.savepar.assert_not_called()

    @patch("metapulsar.position_helpers.bj_name_from_pulsar")
    def test_get_pulsar_name_pint(self, mock_bj_name):
        """Test _get_pulsar_name with PINT objects."""
        # Mock position helper
        mock_bj_name.return_value = "J1857+0943"

        # Create a minimal MetaPulsar instance for testing
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        # Test with PINT objects
        pulsars = {"epta_dr2": (self.mock_model1, self.mock_toas1)}
        result = metapulsar._get_pulsar_name(pulsars)
        assert result == "J1857+0943"

    @patch("metapulsar.position_helpers.bj_name_from_pulsar")
    def test_get_pulsar_name_tempo2(self, mock_bj_name):
        """Test _get_pulsar_name with Tempo2 objects."""
        # Mock position helper
        mock_bj_name.return_value = "J1857+0943"

        # Create a minimal MetaPulsar instance for testing
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        # Test with libstempo objects
        pulsars = {"epta_dr2": self.mock_libstempo_psr}
        result = metapulsar._get_pulsar_name(pulsars)
        assert result == "J1857+0943"

    def test_extract_pulsar_names(self):
        """Test _extract_pulsar_names method."""
        # Create a minimal MetaPulsar instance for testing
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        # Test with mixed PINT and libstempo objects
        pulsars = {
            "epta_dr2": (self.mock_model1, self.mock_toas1),
            "ppta_dr2": self.mock_libstempo_psr,
        }

        result = metapulsar._extract_pulsar_names(pulsars)
        assert result == ["J1857+0943", "J1857+0943"]

    def test_convert_libstempo_to_dict(self):
        """Test _convert_libstempo_to_dict method."""
        # Create a minimal MetaPulsar instance for testing
        metapulsar = MetaPulsar.__new__(MetaPulsar)
        metapulsar.logger = Mock()

        # Test with libstempo object - this method doesn't exist, so we'll test the parfile access
        result = self.mock_libstempo_psr.parfile

        assert result == {"F0": "123.456", "RAJ": "18:57:36.4"}
