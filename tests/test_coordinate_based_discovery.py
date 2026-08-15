"""
Comprehensive tests for coordinate-based pulsar discovery system.

Tests the coordinate-based pulsar identification, B/J name generation,
canonical naming, and MetaPulsarFactory integration.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

import astropy.units as u
from astropy.coordinates import Angle
from pint.models.model_builder import ModelBuilder

from metapulsar.file_discovery import FileDiscovery
from metapulsar.position_helpers import (
    bj_name_from_pulsar,
    discover_pulsars_by_position,
    discover_pulsars_by_coordinates_optimized,
)
from metapulsar.metapulsar import MetaPulsar
from tests.helpers import make_tim_metadata


# === FIXTURES ===


@pytest.fixture
def mock_file_discovery():
    """Mock FileDiscovery with test configurations."""
    service = FileDiscovery(pta_data_releases={})  # Start with empty config
    service.add_data_release(
        "test_data_release1",
        {
            "base_dir": "/test/data1",
            "par_pattern": r"([JB]\d{4}[+-]\d{2,4}[A-Z]?)\.par",
            "tim_pattern": r"([JB]\d{4}[+-]\d{2,4}[A-Z]?)\.tim",
            "timing_package": "pint",
        },
    )
    service.add_data_release(
        "test_data_release2",
        {
            "base_dir": "/test/data2",
            "par_pattern": r"([JB]\d{4}[+-]\d{2,4}[A-Z]?)\.par",
            "tim_pattern": r"([JB]\d{4}[+-]\d{2,4}[A-Z]?)\.tim",
            "timing_package": "pint",
        },
    )
    return service


@pytest.fixture
def mock_parfile_content():
    """Mock parfile content for testing."""
    return """
PSR J1857+0943
F0 123.456 1
F1 -1.23e-15 1
RAJ 18:57:36.3906121
DECJ +09:43:17.20714
PEPOCH 55000.0
DM 13.3
"""


@pytest.fixture
def mock_pint_model(mock_parfile_content):
    """Mock PINT model for testing."""
    mb = ModelBuilder()
    return mb(StringIO(mock_parfile_content), allow_tcb=True, allow_T2=True)


@pytest.fixture
def mock_file_system(tmp_path):
    """Create mock file system with par/tim files."""
    # Create test data directories
    data1 = tmp_path / "data1"
    data2 = tmp_path / "data2"
    data1.mkdir()
    data2.mkdir()

    # Create par files with actual J1857+0943 coordinates
    (data1 / "J1857+0943.par").write_text(
        "PSR J1857+0943\nF0 123.456 1\nRAJ 18:57:36.3906121\nDECJ +09:43:17.20714"
    )
    (data1 / "J1857+0943.tim").write_text("# Mock tim file")
    (data2 / "B1855+09.par").write_text(
        "PSR B1855+09\nF0 123.456 1\nRAJ 18:57:36.3906121\nDECJ +09:43:17.20714"
    )
    (data2 / "B1855+09.tim").write_text("# Mock tim file")

    return tmp_path


# === TEST CLASSES ===


class TestBJNameGeneration:
    """Test B/J name generation functionality."""

    def test_j_name_generation(self, mock_pint_model):
        """Test J-name generation from coordinates."""
        j_name = bj_name_from_pulsar(mock_pint_model, "J")
        assert j_name == "J1857+0943"

    def test_b_name_generation(self, mock_pint_model):
        """Test B-name generation from coordinates."""
        b_name = bj_name_from_pulsar(mock_pint_model, "B")
        assert b_name == "B1855+09"

    def test_default_name_type(self, mock_pint_model):
        """Test default name type is J."""
        name = bj_name_from_pulsar(mock_pint_model)
        assert name == "J1857+0943"

    def test_case_insensitive_name_type(self, mock_pint_model):
        """Test that name type is case insensitive."""
        j_name = bj_name_from_pulsar(mock_pint_model, "j")
        b_name = bj_name_from_pulsar(mock_pint_model, "b")
        assert j_name == "J1857+0943"
        assert b_name == "B1855+09"

    def test_invalid_name_type_raises_error(self, mock_pint_model):
        """Test that invalid name type raises ValueError."""
        with pytest.raises(ValueError):
            bj_name_from_pulsar(mock_pint_model, "X")


class TestCoordinateBasedDiscovery:
    """Test coordinate-based pulsar discovery."""

    def _create_mock_model_with_components(self):
        """Create a mock PINT model with proper component structure."""
        mock_model = Mock()

        # Add coordinate parameters
        ra_angle = Angle(18.960109, unit=u.hourangle)  # 18:57:36.3906121
        dec_angle = Angle(9.721446, unit=u.deg)  # +09:43:17.20714
        mock_model.RAJ = type(
            "obj", (object,), {"quantity": ra_angle, "value": ra_angle.value}
        )()
        mock_model.DECJ = type(
            "obj", (object,), {"quantity": dec_angle, "value": dec_angle.value}
        )()

        # Add PSR and UNITS parameters
        mock_model.PSR = type("obj", (object,), {"value": "J1857+0943"})()
        mock_model.UNITS = type("obj", (object,), {"value": "TDB"})()

        # Create mock components with proper structure
        mock_astrometry = Mock()
        mock_astrometry.category = "astrometry"
        mock_astrometry.params = ["RAJ", "DECJ", "PMRA", "PMDEC", "PEPOCH"]

        mock_spindown = Mock()
        mock_spindown.category = "spindown"
        mock_spindown.params = ["F0", "F1", "F2"]

        # Mock components.values() to return our mock components
        mock_model.components = Mock()
        mock_model.components.values.return_value = [mock_astrometry, mock_spindown]

        return mock_model

    @patch("pint.models.model_builder.parse_parfile")
    @patch("pint.models.model_builder.ModelBuilder")
    def test_discover_pulsars_by_coordinates(
        self,
        mock_model_builder_class,
        mock_parse_parfile,
        mock_file_discovery,
        mock_file_system,
    ):
        """Test position-based pulsar discovery merges B/J catalog aliases."""
        file_data = {
            "test_data_release1": [
                {
                    "par": mock_file_system / "data1" / "J1857+0943.par",
                    "tim": mock_file_system / "data1" / "J1857+0943.tim",
                    "par_content": (
                        mock_file_system / "data1" / "J1857+0943.par"
                    ).read_text(),
                    "timing_package": "pint",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ],
            "test_data_release2": [
                {
                    "par": mock_file_system / "data2" / "B1855+09.par",
                    "tim": mock_file_system / "data2" / "B1855+09.tim",
                    "par_content": (
                        mock_file_system / "data2" / "B1855+09.par"
                    ).read_text(),
                    "timing_package": "pint",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ],
        }
        coordinate_map = discover_pulsars_by_position(file_data)

        assert "B1855+09" in coordinate_map
        pulsar_info = coordinate_map["B1855+09"]
        assert "test_data_release1" in pulsar_info
        assert "test_data_release2" in pulsar_info
        assert len(pulsar_info["test_data_release1"]) > 0
        assert len(pulsar_info["test_data_release2"]) > 0

    def test_create_metapulsar_with_canonical_name(
        self, mock_file_discovery, mock_file_system, mock_metapulsar
    ):
        """Test MetaPulsar creation includes canonical name."""
        from metapulsar.mockpulsar import create_mock_libstempo

        adapted_pulsars = {
            "test_data_release1": create_mock_libstempo(
                n_toas=50,
                name="J1857+0943",
                telescope="test_data_release1",
                include_astrometry=True,
                include_spin=True,
                seed=10,
            ),
            "test_data_release2": create_mock_libstempo(
                n_toas=50,
                name="J1857+0943",
                telescope="test_data_release2",
                include_astrometry=True,
                include_spin=True,
                seed=20,
            ),
        }
        metapulsar = mock_metapulsar(
            adapted_pulsars,
            combination_strategy="per_pta",
        )

        assert isinstance(metapulsar, MetaPulsar)
        assert hasattr(metapulsar, "name")
        # The name will be determined from the pulsar data, not from a parameter
        assert hasattr(metapulsar, "_pulsars")
        assert len(metapulsar._pulsars) == 2

    def test_discover_files_coordinate_matching(self, mock_file_discovery):
        """Test file discovery with coordinate matching using FileDiscovery."""
        # Test that FileDiscovery can discover files for a pulsar
        files = mock_file_discovery.discover_files(
            ["test_data_release1", "test_data_release2"]
        )

        # Should return file data for both PTAs
        assert "test_data_release1" in files
        assert "test_data_release2" in files

    def test_discover_files_pulsar_not_found(self, mock_file_discovery):
        """Test file discovery when pulsar not found using FileDiscovery."""
        # Test that FileDiscovery handles unknown PTAs gracefully
        with pytest.raises(KeyError):
            mock_file_discovery.discover_files(["unknown_pta"])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_coordinate_discovery_with_malformed_parfile(
        self, mock_file_discovery, mock_file_system
    ):
        """Test coordinate discovery handles malformed parfiles gracefully."""
        # Create malformed parfile
        malformed_par = mock_file_system / "data1" / "malformed.par"
        malformed_par.write_text("This is not a valid parfile")

        # Update the mock service with the test directory
        mock_file_discovery.data_releases["test_data_release1"]["base_dir"] = str(
            mock_file_system / "data1"
        )

        # factory = MetaPulsarFactory()

        with patch("pint.models.model_builder.parse_parfile") as mock_parse:
            mock_parse.side_effect = Exception("Parse error")

            # Create file_data format expected by the method
            file_data = {
                "test_data_release1": [
                    {
                        "par": mock_file_system / "data1" / "malformed.par",
                        "tim": mock_file_system / "data1" / "malformed.tim",
                        "par_content": "This is not a valid parfile",
                        "timing_package": "pint",
                        "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    }
                ]
            }

            # Should not raise exception, just log warning
            coordinate_map = discover_pulsars_by_coordinates_optimized(file_data)
            assert coordinate_map == {}

    def test_bj_name_from_pulsar_with_invalid_object(self):
        """Test bj_name_from_pulsar with invalid object."""
        invalid_obj = "not a pulsar object"

        with pytest.raises(ValueError):
            bj_name_from_pulsar(invalid_obj, "J")


# === INTEGRATION TESTS ===


def test_discover_pulsars_same_j2000_position_merge(load_parfile_text):
    """Two PTAs with identical parfiles merge into one catalog-named group."""
    parfile = load_parfile_text("test_same_pulsar_epoch1_54500.par")

    file_data = {
        "PTA1": [
            {
                "par": "test1.par",
                "par_content": parfile,
                "tim": "test1.tim",
            }
        ],
        "PTA2": [
            {
                "par": "test2.par",
                "par_content": parfile,
                "tim": "test2.tim",
            }
        ],
    }

    coordinate_map = discover_pulsars_by_position(file_data)
    assert len(coordinate_map) == 1
    group_name = next(iter(coordinate_map))
    assert len(coordinate_map[group_name]) == 2


def test_discover_pulsars_same_catalog_name_over_tolerance_raises(
    load_parfile_text,
):
    """Same PSRJ on files with J2000 separation > match tolerance is a hard error."""
    parfile_epoch1 = load_parfile_text("test_same_pulsar_epoch1_54500.par")
    parfile_epoch2 = load_parfile_text("test_same_pulsar_epoch2_56000.par")

    file_data = {
        "PTA1": [
            {
                "par": "test1.par",
                "par_content": parfile_epoch1,
                "tim": "test1.tim",
            }
        ],
        "PTA2": [
            {
                "par": "test2.par",
                "par_content": parfile_epoch2,
                "tim": "test2.tim",
            }
        ],
    }

    with pytest.raises(ValueError, match="distinct sky positions"):
        discover_pulsars_by_position(file_data)
