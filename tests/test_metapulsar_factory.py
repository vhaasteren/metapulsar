"""Tests for Meta-Pulsar Factory."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from metapulsar.metapulsar_factory import MetaPulsarFactory
from metapulsar.file_discovery_service import FileDiscoveryService
from tests.helpers import make_tim_metadata


class TestParfileContentValidation:
    """Test cases for parfile content validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MetaPulsarFactory()

    @pytest.mark.requires_ipta_data
    def test_ensure_parfile_content_with_missing_content(self):
        """Test validation when par_content is missing."""
        # Create file data without par_content
        file_data = {
            "test_pta": [
                {
                    "par": "data/ipta-dr2/PPTA_dr1dr2/par/J1857+0943_dr1dr2.par",
                    "tim": "data/ipta-dr2/PPTA_dr1dr2/tim/J1857+0943_dr1dr2.tim",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        # Validate should add par_content
        validated = self.factory._ensure_parfile_content(file_data)

        assert "test_pta" in validated
        assert "par_content" in validated["test_pta"][0]
        assert len(validated["test_pta"][0]["par_content"]) > 0
        assert "PSR" in validated["test_pta"][0]["par_content"]

    def test_ensure_parfile_content_with_existing_content(self):
        """Test validation when par_content already exists."""
        # Create file data with existing par_content
        file_data = {
            "test_pta": [
                {
                    "par": "data/ipta-dr2/PPTA_dr1dr2/par/J1857+0943_dr1dr2.par",
                    "tim": "data/ipta-dr2/PPTA_dr1dr2/tim/J1857+0943_dr1dr2.tim",
                    "par_content": "PSR J1857+0943\nF0 186.494081\n",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        # Validate should not modify existing content
        validated = self.factory._ensure_parfile_content(file_data)

        assert "test_pta" in validated
        assert "par_content" in validated["test_pta"][0]
        assert (
            validated["test_pta"][0]["par_content"] == "PSR J1857+0943\nF0 186.494081\n"
        )

    def test_ensure_parfile_content_missing_par_path(self):
        """Test validation when par file path is missing."""
        # Create file data without par path
        file_data = {
            "test_pta": [
                {
                    "tim": "data/ipta-dr2/PPTA_dr1dr2/tim/J1857+0943_dr1dr2.tim",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing 'par' file path"):
            self.factory._ensure_parfile_content(file_data)

    def test_ensure_parfile_content_file_not_found(self):
        """Test validation when par file doesn't exist."""
        # Create file data with non-existent par file
        file_data = {
            "test_pta": [
                {
                    "par": "non_existent_file.par",
                    "tim": "data/ipta-dr2/PPTA_dr1dr2/tim/J1857+0943_dr1dr2.tim",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "timing_package": "tempo2",
                }
            ]
        }

        # Should raise ValueError
        with pytest.raises(ValueError, match="Parfile not found"):
            self.factory._ensure_parfile_content(file_data)


class TestMetaPulsarFactory:
    """Test MetaPulsarFactory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MetaPulsarFactory()
        self.discovery_service = FileDiscoveryService(working_dir="../../data/ipta-dr2")

    def test_initialization(self):
        """Test factory initialization without ParFileManager."""
        factory = MetaPulsarFactory()
        assert factory.logger is not None
        assert not hasattr(factory, "parfile_manager")

    @patch("metapulsar.position_helpers.bj_name_from_pulsar")
    def test_create_metapulsar_success(self, mock_bj_name):
        """Test successful MetaPulsar creation using MockLibstempo directly."""
        # Mock position helper
        mock_bj_name.return_value = "J1857+0943"
        from metapulsar.mockpulsar import create_mock_libstempo

        mock_psr = create_mock_libstempo(
            n_toas=50,
            name="J1857+0943",
            telescope="test_pta",
            include_astrometry=True,
            include_spin=True,
            seed=42,
        )

        # Create MetaPulsar with raw MockLibstempo
        from metapulsar.metapulsar import MetaPulsar

        pulsars = {"test_pta": mock_psr}
        metapulsar = MetaPulsar(pulsars=pulsars, combination_strategy="composite")

        assert metapulsar is not None
        assert hasattr(metapulsar, "_pulsars")
        assert len(metapulsar._pulsars) == 1
        assert metapulsar.name == "J1857+0943"

    def test_validate_single_pulsar_data_empty(self):
        """Test validation with empty file data."""
        empty_file_data = {}

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized",
            return_value={},
        ):
            with pytest.raises(ValueError, match="No valid pulsar files found"):
                self.factory._validate_single_pulsar_data(empty_file_data)

    def test_validate_single_pulsar_data_multiple_pulsars(self):
        """Test validation with multiple pulsars in file data."""
        # Mock file data with multiple pulsars
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("/data/epta/J1857+0943.par"),
                    "tim": Path("/data/epta/J1857+0943.tim"),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("/data/ppta/J1909-3744.par"),
                    "tim": Path("/data/ppta/J1909-3744.tim"),
                }
            ],
        }

        # Mock coordinate discovery to return multiple pulsars
        mock_pulsar_groups = {
            "J1857+0943": {"epta_dr2": [file_data["epta_dr2"][0]]},
            "J1909-3744": {"ppta_dr2": [file_data["ppta_dr2"][0]]},
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized",
            return_value=mock_pulsar_groups,
        ):
            with pytest.raises(ValueError, match="Multiple pulsars detected"):
                self.factory._validate_single_pulsar_data(file_data)

    def test_validate_single_pulsar_data_single_pulsar(self):
        """Test validation with single pulsar in file data."""
        # Mock file data with single pulsar
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("/data/epta/J1857+0943.par"),
                    "tim": Path("/data/epta/J1857+0943.tim"),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("/data/ppta/J1857+0943.par"),
                    "tim": Path("/data/ppta/J1857+0943.tim"),
                }
            ],
        }

        # Mock coordinate discovery to return single pulsar
        mock_pulsar_groups = {
            "J1857+0943": {
                "epta_dr2": [file_data["epta_dr2"][0]],
                "ppta_dr2": [file_data["ppta_dr2"][0]],
            }
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized",
            return_value=mock_pulsar_groups,
        ):
            # Should not raise an exception
            self.factory._validate_single_pulsar_data(file_data)

    def test_group_files_by_pulsar_empty(self):
        """Test grouping with empty file data."""
        empty_file_data = {}

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized",
            return_value={},
        ):
            with pytest.raises(ValueError, match="No valid pulsar files found"):
                self.factory.group_files_by_pulsar(empty_file_data)

    def test_group_files_by_pulsar_success(self):
        """Test successful grouping of files by pulsar."""
        # Mock file data with multiple pulsars
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("/data/epta/J1857+0943.par"),
                    "tim": Path("/data/epta/J1857+0943.tim"),
                },
                {
                    "par": Path("/data/epta/J1909-3744.par"),
                    "tim": Path("/data/epta/J1909-3744.tim"),
                },
            ],
            "ppta_dr2": [
                {
                    "par": Path("/data/ppta/J1857+0943.par"),
                    "tim": Path("/data/ppta/J1857+0943.tim"),
                },
                {
                    "par": Path("/data/ppta/J1909-3744.par"),
                    "tim": Path("/data/ppta/J1909-3744.tim"),
                },
            ],
        }

        # Mock coordinate discovery to return grouped pulsars
        expected_groups = {
            "J1857+0943": {
                "epta_dr2": [file_data["epta_dr2"][0]],
                "ppta_dr2": [file_data["ppta_dr2"][0]],
            },
            "J1909-3744": {
                "epta_dr2": [file_data["epta_dr2"][1]],
                "ppta_dr2": [file_data["ppta_dr2"][1]],
            },
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized",
            return_value=expected_groups,
        ):
            result = self.factory.group_files_by_pulsar(file_data)

            assert result == expected_groups
            assert len(result) == 2
            assert "J1857+0943" in result
            assert "J1909-3744" in result
            assert "epta_dr2" in result["J1857+0943"]
            assert "ppta_dr2" in result["J1857+0943"]

    def test_create_metapulsar_with_validation_multiple_pulsars(self):
        """Test create_metapulsar with validation fails for multiple pulsars."""
        # Mock file data with multiple pulsars
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("/data/epta/J1857+0943.par"),
                    "tim": Path("/data/epta/J1857+0943.tim"),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("/data/ppta/J1909-3744.par"),
                    "tim": Path("/data/ppta/J1909-3744.tim"),
                }
            ],
        }

        # Mock coordinate discovery to return multiple pulsars
        mock_pulsar_groups = {
            "J1857+0943": {"epta_dr2": [file_data["epta_dr2"][0]]},
            "J1909-3744": {"ppta_dr2": [file_data["ppta_dr2"][0]]},
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized",
            return_value=mock_pulsar_groups,
        ):
            with patch.object(self.factory, "_ensure_parfile_content") as mock_ensure:
                mock_ensure.return_value = file_data
                with pytest.raises(ValueError, match="Multiple pulsars detected"):
                    self.factory.create_metapulsar(file_data)

    def test_file_discovery_service_integration(self):
        """Test integration with FileDiscoveryService."""
        # Test that FileDiscoveryService can be used independently
        assert self.discovery_service is not None
        assert hasattr(self.discovery_service, "discover_files")
        assert hasattr(self.discovery_service, "list_data_releases")

        # Test listing PTAs
        data_releases = self.discovery_service.list_data_releases()
        assert isinstance(data_releases, list)
        assert len(data_releases) > 0

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_create_metapulsar_with_consistent_strategy(self, mock_param_manager):
        """Test create_metapulsar with consistent strategy using ParameterManager."""
        # Mock ParameterManager
        mock_manager_instance = Mock()
        mock_manager_instance.make_parfiles_consistent.return_value = {
            "epta_dr2": Path("/tmp/consistent_epta_dr2.par")
        }
        mock_param_manager.return_value = mock_manager_instance

        # Create mock file data
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("/data/epta/J1857+0943.par"),
                    "tim": Path("/data/epta/J1857+0943.tim"),
                    "timing_package": "pint",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "par_content": "PSR J1857+0943\nF0 123.456\nRAJ 18:57:36.4\nDECJ 9:43:17.2\n",
                }
            ]
        }

        # Mock the pulsar creation and MetaPulsar creation to avoid complex setup
        with patch.object(
            self.factory, "_create_pulsar_objects"
        ) as mock_create_pulsars:
            with patch(
                "metapulsar.metapulsar_factory.MetaPulsar"
            ) as mock_metapulsar_class:
                mock_metapulsar = Mock()
                mock_metapulsar_class.return_value = mock_metapulsar
                mock_create_pulsars.return_value = {"epta_dr2": Mock()}

                # Test the ParameterManager integration
                result = self.factory.create_metapulsar(
                    file_data,
                    combination_strategy="consistent",
                    combine_components=["astrometry", "spindown"],
                )

                mock_param_manager.assert_called_once()
                call_args = mock_param_manager.call_args
                assert call_args[1]["combine_components"] == ["astrometry", "spindown"]

                # Verify the result
                assert result == mock_metapulsar

    def test_create_pulsar_objects_pint(self):
        """Test _create_pulsar_objects with PINT timing package."""
        file_pairs = {
            "epta_dr2": (
                Path("/data/epta/J1857+0943.par"),
                Path("/data/epta/J1857+0943.tim"),
            )
        }
        file_data = {
            "epta_dr2": {
                "par": Path("/data/epta/J1857+0943.par"),
                "tim": Path("/data/epta/J1857+0943.tim"),
                "timing_package": "pint",
                "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                "par_content": "PSR J1857+0943\nF0 123.456\n",
            }
        }

        with patch(
            "metapulsar.metapulsar_factory.get_model_and_toas"
        ) as mock_get_model:
            mock_model = Mock()
            mock_toas = Mock()
            mock_get_model.return_value = (mock_model, mock_toas)

            result = self.factory._create_pulsar_objects(
                file_pairs, file_data, use_pulse_numbers="no"
            )

            assert "epta_dr2" in result
            assert result["epta_dr2"] == (mock_model, mock_toas)
            mock_get_model.assert_called_once_with(
                str(file_pairs["epta_dr2"][0]),
                str(file_pairs["epta_dr2"][1]),
                planets=True,
                allow_T2=True,
            )

    def test_create_pulsar_objects_tempo2(self):
        """Test _create_pulsar_objects with Tempo2 timing package."""
        file_pairs = {
            "epta_dr2": (
                Path("/data/epta/J1857+0943.par"),
                Path("/data/epta/J1857+0943.tim"),
            )
        }
        file_data = {
            "epta_dr2": {
                "par": Path("/data/epta/J1857+0943.par"),
                "tim": Path("/data/epta/J1857+0943.tim"),
                "timing_package": "tempo2",
                "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                "par_content": "PSR J1857+0943\nF0 123.456\n",
            }
        }

        with patch("metapulsar.metapulsar_factory.tempopulsar") as mock_tempopulsar:
            mock_psr = Mock()
            mock_tempopulsar.return_value = mock_psr

            result = self.factory._create_pulsar_objects(
                file_pairs, file_data, use_pulse_numbers="no"
            )

            assert "epta_dr2" in result
            assert result["epta_dr2"] == mock_psr
            mock_tempopulsar.assert_called_once_with(
                parfile=str(file_pairs["epta_dr2"][0]),
                timfile=str(file_pairs["epta_dr2"][1]),
                dofit=False,
            )

    def test_create_pulsar_objects_tempo2_yes_uses_track_minus_2(self, tmp_path):
        """Tempo2 yes mode wraps par with TRACK -2."""
        par_path = tmp_path / "test.par"
        tim_path = tmp_path / "test.tim"
        par_path.write_text("PSR J1857+0943\nF0 123.456\n", encoding="utf-8")
        tim_path.write_text("FORMAT 1\nMODE 1\n", encoding="utf-8")

        file_pairs = {"epta_dr2": (par_path, tim_path)}
        file_data = {
            "epta_dr2": {
                "par": par_path,
                "tim": tim_path,
                "timing_package": "tempo2",
                "par_content": par_path.read_text(encoding="utf-8"),
            }
        }

        with (
            patch(
                "metapulsar.metapulsar_factory.resolved_tim_for_pulse_numbers"
            ) as mock_resolved,
            patch(
                "metapulsar.metapulsar_factory.temporary_par_with_track_minus_2"
            ) as mock_track_par,
            patch("metapulsar.metapulsar_factory.tempopulsar") as mock_tempopulsar,
        ):
            mock_resolved.return_value.__enter__.return_value = str(tim_path)
            mock_track_par.return_value.__enter__.return_value = "/tmp/track.par"
            mock_tempopulsar.return_value = Mock()

            self.factory._create_pulsar_objects(
                file_pairs, file_data, use_pulse_numbers="yes"
            )

            mock_track_par.assert_called_once()
            mock_tempopulsar.assert_called_once_with(
                parfile="/tmp/track.par",
                timfile=str(tim_path),
                dofit=False,
            )

    def test_validate_pulse_number_mode_rejects_bool(self):
        with pytest.raises(ValueError, match="must be one of"):
            self.factory.create_metapulsar(
                {
                    "pta": [
                        {
                            "par": Path("x.par"),
                            "tim": Path("x.tim"),
                            "timing_package": "pint",
                            "par_content": "PSR J0000+0000\nF0 1\n",
                        }
                    ]
                },
                use_pulse_numbers=True,  # type: ignore[arg-type]
            )


class TestPerPulsarOrdering:
    """Test cases for per-pulsar reference PTA ordering functionality."""

    def test_group_files_by_pulsar_with_ordering_auto_selection(self):
        """Test automatic reference PTA selection by timespan."""
        factory = MetaPulsarFactory()

        # Mock file data with different timespans
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=2000.0),
                }
            ],
        }

        # Mock the coordinate-based discovery to return grouped data
        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized"
        ) as mock_discover:
            mock_discover.return_value = {
                "J1857+0943": {
                    "epta_dr2": file_data["epta_dr2"],
                    "ppta_dr2": file_data["ppta_dr2"],
                }
            }

            result = factory._group_files_by_pulsar_with_ordering(
                file_data, reference_pta=None
            )

            # PPTA should be first (longer timespan)
            assert list(result["J1857+0943"].keys())[0] == "ppta_dr2"

    def test_group_files_by_pulsar_with_ordering_specified_reference(self):
        """Test specified reference PTA ordering."""
        factory = MetaPulsarFactory()

        file_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=2000.0),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ],
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized"
        ) as mock_discover:
            mock_discover.return_value = {
                "J1857+0943": {
                    "epta_dr2": file_data["epta_dr2"],
                    "ppta_dr2": file_data["ppta_dr2"],
                }
            }

            result = factory._group_files_by_pulsar_with_ordering(
                file_data, reference_pta="epta_dr2"
            )

            # EPTA should be first (specified reference)
            assert list(result["J1857+0943"].keys())[0] == "epta_dr2"

    def test_group_files_by_pulsar_with_ordering_fallback(self):
        """Test fallback to auto-selection when specified PTA not available."""
        factory = MetaPulsarFactory()

        file_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=2000.0),
                }
            ],
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized"
        ) as mock_discover:
            mock_discover.return_value = {
                "J1857+0943": {
                    "epta_dr2": file_data["epta_dr2"],
                    "ppta_dr2": file_data["ppta_dr2"],
                }
            }

            # Specify a PTA that doesn't exist for this pulsar
            result = factory._group_files_by_pulsar_with_ordering(
                file_data, reference_pta="nanograv_12y"
            )

            # Should fallback to PPTA (longer timespan)
            assert list(result["J1857+0943"].keys())[0] == "ppta_dr2"

    def test_find_best_reference_pta_by_timespan(self):
        """Test finding best reference PTA by timespan."""
        factory = MetaPulsarFactory()

        pulsar_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=2000.0),
                }
            ],
            "nanograv_12y": [
                {
                    "par": Path("test3.par"),
                    "tim": Path("test3.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=1500.0),
                }
            ],
        }

        result = factory._find_best_reference_pta_by_timespan(pulsar_data)
        assert result == "ppta_dr2"  # Longest timespan

    def test_find_best_reference_pta_by_timespan_empty_files(self):
        """Test finding best reference PTA with empty file lists."""
        factory = MetaPulsarFactory()

        pulsar_data = {
            "epta_dr2": [],  # Empty files
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "tim_metadata": make_tim_metadata(timespan_days=2000.0),
                }
            ],
        }

        result = factory._find_best_reference_pta_by_timespan(pulsar_data)
        assert result == "ppta_dr2"  # Only non-empty PTA

    def test_reorder_ptas_for_pulsar(self):
        """Test reordering PTAs for a specific pulsar."""
        from metapulsar.metapulsar_factory import reorder_ptas_for_pulsar

        pulsar_data = {
            "epta_dr2": [{"par": Path("test1.par"), "tim": Path("test1.tim")}],
            "ppta_dr2": [{"par": Path("test2.par"), "tim": Path("test2.tim")}],
            "nanograv_12y": [{"par": Path("test3.par"), "tim": Path("test3.tim")}],
        }

        result = reorder_ptas_for_pulsar(pulsar_data, "ppta_dr2")

        # PPTA should be first
        assert list(result.keys())[0] == "ppta_dr2"
        # All PTAs should still be present
        assert len(result) == 3
        assert "epta_dr2" in result
        assert "nanograv_12y" in result

    def test_reorder_ptas_for_pulsar_invalid_reference(self):
        """Test reordering with invalid reference PTA."""
        from metapulsar.metapulsar_factory import reorder_ptas_for_pulsar

        pulsar_data = {
            "epta_dr2": [{"par": Path("test1.par"), "tim": Path("test1.tim")}],
            "ppta_dr2": [{"par": Path("test2.par"), "tim": Path("test2.tim")}],
        }

        with pytest.raises(ValueError, match="Reference PTA 'nanograv_12y' not found"):
            reorder_ptas_for_pulsar(pulsar_data, "nanograv_12y")

    def test_create_all_metapulsars_with_ordering(self):
        """Test create_all_metapulsars with new ordering logic."""
        factory = MetaPulsarFactory()

        file_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "tim_metadata": make_tim_metadata(
                        timespan_days=1000.0, toa_count=500
                    ),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "tim_metadata": make_tim_metadata(
                        timespan_days=2000.0, toa_count=1000
                    ),
                }
            ],
        }

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized"
        ) as mock_discover:
            mock_discover.return_value = {
                "J1857+0943": {
                    "epta_dr2": file_data["epta_dr2"],
                    "ppta_dr2": file_data["ppta_dr2"],
                }
            }

            # Mock the internal methods to avoid actual file processing
            with patch.object(factory, "_ensure_parfile_content") as mock_ensure:
                with patch.object(factory, "create_metapulsar") as mock_create:
                    mock_ensure.return_value = file_data
                    mock_metapulsar = Mock()
                    mock_metapulsar.name = "J1857+0943"
                    mock_create.return_value = mock_metapulsar

                    result = factory.create_all_metapulsars(
                        file_data, reference_pta=None
                    )

                    # Should create MetaPulsar for the pulsar
                    assert "J1857+0943" in result

                    # Should call create_metapulsar with PPTA data first (longer timespan)
                    mock_create.assert_called_once()
                    call_args = mock_create.call_args
                    # Check that the file_data passed to create_metapulsar has PPTA first
                    file_data_passed = call_args[1]["file_data"]
                    assert list(file_data_passed.keys())[0] == "ppta_dr2"


class TestPtaSummary:
    """Tests for PTA summary display including pulse-number coverage."""

    def test_pta_summary_shows_pn_status(self, capsys):
        factory = MetaPulsarFactory()
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.4\nDECJ 09:43:17.1\n",
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(
                        timespan_days=1000.0,
                        toa_count=100,
                        pn_status="complete",
                        pn_with_count=100,
                        pn_without_count=0,
                    ),
                }
            ],
            "ppta_dr2": [
                {
                    "par": Path("test2.par"),
                    "tim": Path("test2.tim"),
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.4\nDECJ 09:43:17.1\n",
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(
                        timespan_days=2000.0,
                        toa_count=200,
                        pn_status="mixed",
                        pn_with_count=120,
                        pn_without_count=80,
                    ),
                }
            ],
        }

        with (
            patch(
                "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized"
            ) as mock_discover,
            patch.object(
                factory, "_get_display_name_for_pulsar", return_value="J1857+0943"
            ),
        ):
            mock_discover.return_value = {
                "J1857+0943": {
                    "epta_dr2": file_data["epta_dr2"],
                    "ppta_dr2": file_data["ppta_dr2"],
                }
            }
            factory.pta_summary(file_data)

        captured = capsys.readouterr().out
        assert "pn=complete (100/100)" in captured
        assert "pn=mixed (120/200)" in captured

    def test_pta_summary_aggregates_pn_across_files(self, capsys):
        factory = MetaPulsarFactory()
        file_data = {
            "epta_dr2": [
                {
                    "par": Path("test1.par"),
                    "tim": Path("test1.tim"),
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.4\nDECJ 09:43:17.1\n",
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(
                        timespan_days=1000.0,
                        toa_count=100,
                        pn_status="complete",
                        pn_with_count=100,
                        pn_without_count=0,
                    ),
                },
                {
                    "par": Path("test1b.par"),
                    "tim": Path("test1b.tim"),
                    "par_content": "PSR J1857+0943\nRAJ 18:57:36.4\nDECJ 09:43:17.1\n",
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(
                        timespan_days=800.0,
                        toa_count=50,
                        pn_status="none",
                        pn_with_count=0,
                        pn_without_count=50,
                    ),
                },
            ],
        }

        with (
            patch(
                "metapulsar.metapulsar_factory.discover_pulsars_by_coordinates_optimized"
            ) as mock_discover,
            patch.object(
                factory, "_get_display_name_for_pulsar", return_value="J1857+0943"
            ),
        ):
            mock_discover.return_value = {
                "J1857+0943": {"epta_dr2": file_data["epta_dr2"]},
            }
            factory.pta_summary(file_data)

        captured = capsys.readouterr().out
        assert "150 TOAs" in captured
        assert "pn=mixed (100/150)" in captured
