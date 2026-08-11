"""Tests for Meta-Pulsar Factory."""

import pytest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch
from metapulsar.metapulsar_factory import (
    MetaPulsarFactory,
    _par_content_has_dmx,
    _safe_pta_filename,
    _SINGLE_PTA_SHARED_DMX_WARNING,
)
from metapulsar.file_discovery import FileDiscovery
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
        self.discovery_service = FileDiscovery(working_dir="../../data/ipta-dr2")

    def test_initialization(self):
        """Test factory initialization without ParFileManager."""
        factory = MetaPulsarFactory()
        assert factory.logger is not None
        assert not hasattr(factory, "parfile_manager")

    def test_safe_pta_filename_is_injective_for_punctuation(self):
        slash = _safe_pta_filename("pta/a")
        question = _safe_pta_filename("pta?a")
        underscore = _safe_pta_filename("pta_a")

        assert len({slash, question, underscore}) == 3
        assert "/" not in slash

    def test_create_metapulsar_success(self):
        """Test successful MetaPulsar creation using MockLibstempo directly."""
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
        metapulsar = MetaPulsar(pulsars=pulsars, combination_strategy="per_pta")

        assert metapulsar is not None
        assert hasattr(metapulsar, "_pulsars")
        assert len(metapulsar._pulsars) == 1
        assert metapulsar.name == "J1857+0943"

    def test_validate_single_pulsar_data_empty(self):
        """Test validation with empty file data."""
        empty_file_data = {}

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_position",
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position",
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position",
            return_value=mock_pulsar_groups,
        ):
            # Should not raise an exception
            self.factory._validate_single_pulsar_data(file_data)

    def test_group_files_by_pulsar_empty(self):
        """Test grouping with empty file data."""
        empty_file_data = {}

        with patch(
            "metapulsar.metapulsar_factory.discover_pulsars_by_position",
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position",
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position",
            return_value=mock_pulsar_groups,
        ):
            with patch.object(self.factory, "_ensure_parfile_content") as mock_ensure:
                mock_ensure.return_value = file_data
                with pytest.raises(ValueError, match="Multiple pulsars detected"):
                    self.factory.create_metapulsar(file_data)

    def test_file_discovery_integration(self):
        """Test integration with FileDiscovery."""
        # Test that FileDiscovery can be used independently
        assert self.discovery_service is not None
        assert hasattr(self.discovery_service, "discover_files")
        assert hasattr(self.discovery_service, "list_data_releases")

        # Test listing PTAs
        data_releases = self.discovery_service.list_data_releases()
        assert isinstance(data_releases, list)
        assert len(data_releases) > 0

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_create_metapulsar_with_shared_strategy(self, mock_param_manager):
        """Test create_metapulsar with shared strategy using ParameterManager."""
        # Mock ParameterManager
        mock_manager_instance = Mock()
        mock_manager_instance.make_parfiles_shared.return_value = {
            "epta_dr2": Path("/tmp/shared_epta_dr2.par")
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

        # Mock pulsar creation / MODE transfer / MetaPulsar — fixtures use
        # placeholder .tim paths that are not on disk.
        with (
            patch.object(self.factory, "_create_pulsar_objects") as mock_create_pulsars,
            patch.object(
                self.factory,
                "_apply_tim_mode_transfer",
                side_effect=lambda **kw: (kw["engine_pars"], set()),
            ),
            patch("metapulsar.metapulsar_factory.MetaPulsar") as mock_metapulsar_class,
        ):
            mock_metapulsar = Mock()
            mock_metapulsar_class.return_value = mock_metapulsar
            mock_create_pulsars.return_value = {"epta_dr2": Mock()}

            result = self.factory.create_metapulsar(
                file_data,
                combination_strategy="shared",
                combine_components=["astrometry", "spindown"],
            )

            mock_param_manager.assert_called_once()
            call_args = mock_param_manager.call_args
            assert call_args[1]["combine_components"] == ["astrometry", "spindown"]
            assert result == mock_metapulsar

    @staticmethod
    def _write_session_inputs(tmp_path, timing_package):
        par_path = tmp_path / "J1857+0943.par"
        tim_path = tmp_path / "J1857+0943.tim"
        par_path.write_text("PSR J1857+0943\nF0 123.456\n", encoding="utf-8")
        tim_path.write_text(
            "FORMAT 1\n obs1 1400.0 58000.0 1.0 g -sys foo\n", encoding="utf-8"
        )
        file_pairs = {"epta_dr2": (par_path, tim_path)}
        file_data = {
            "epta_dr2": {
                "par": par_path,
                "tim": tim_path,
                "timing_package": timing_package,
                "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                "par_content": par_path.read_text(encoding="utf-8"),
            }
        }
        return par_path, tim_path, file_pairs, file_data

    def test_create_pulsar_objects_pint_loads_canonical_tim(self, tmp_path):
        """PINT loads the canonical stamped tim, not the release tim."""
        par_path, tim_path, file_pairs, file_data = self._write_session_inputs(
            tmp_path, "pint"
        )
        session_dir = tmp_path / "session"

        with patch(
            "metapulsar.metapulsar_factory.get_model_and_toas"
        ) as mock_get_model:
            mock_model = Mock()
            mock_toas = Mock()
            mock_get_model.return_value = (mock_model, mock_toas)

            result = self.factory._create_pulsar_objects(
                file_pairs,
                file_data,
                use_pulse_numbers="no",
                pta_file_dir=session_dir,
                canonicalize_tim=True,
            )

        canonical = session_dir / "epta_dr2.tim"
        assert result["epta_dr2"] == (mock_model, mock_toas)
        mock_get_model.assert_called_once_with(
            str(par_path),
            str(canonical),
            planets=True,
            allow_T2=True,
            ell1h_shapiro="full",
        )
        text = canonical.read_text(encoding="utf-8")
        assert "-pta epta_dr2" in text
        assert "-pta_dataset epta_dr2" in text
        assert "-timing_package pint" in text
        assert tim_path.read_text(encoding="utf-8") == (
            "FORMAT 1\n obs1 1400.0 58000.0 1.0 g -sys foo\n"
        )

    def test_create_pulsar_objects_tempo2_loads_canonical_tim(self, tmp_path):
        """Tempo2 loads the canonical stamped tim, not the release tim."""
        par_path, _, file_pairs, file_data = self._write_session_inputs(
            tmp_path, "tempo2"
        )
        session_dir = tmp_path / "session"

        with patch("metapulsar.metapulsar_factory.tempopulsar") as mock_tempopulsar:
            mock_psr = Mock()
            mock_tempopulsar.return_value = mock_psr

            result = self.factory._create_pulsar_objects(
                file_pairs,
                file_data,
                use_pulse_numbers="no",
                pta_file_dir=session_dir,
                canonicalize_tim=True,
            )

        canonical = session_dir / "epta_dr2.tim"
        assert result["epta_dr2"] == mock_psr
        mock_tempopulsar.assert_called_once_with(
            parfile=str(par_path),
            timfile=str(canonical),
            dofit=False,
        )
        assert "-timing_package tempo2" in canonical.read_text(encoding="utf-8")

    def test_create_pulsar_objects_exports_canonical_tim(self, tmp_path):
        """timfile_output_dir receives the exact file the engine consumed."""
        _, _, file_pairs, file_data = self._write_session_inputs(tmp_path, "pint")
        session_dir = tmp_path / "session"
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        with patch(
            "metapulsar.metapulsar_factory.get_model_and_toas"
        ) as mock_get_model:
            mock_get_model.return_value = (Mock(), Mock())
            _, pta_files = self.factory._create_pulsar_objects(
                file_pairs,
                file_data,
                use_pulse_numbers="no",
                pta_file_dir=session_dir,
                return_pta_files=True,
                canonicalize_tim=True,
            )

        self.factory._write_canonical_timfiles(pta_files, export_dir, "J1857+0943")
        exported = export_dir / "J1857+0943_epta_dr2.tim"
        canonical = session_dir / "epta_dr2.tim"
        assert pta_files["epta_dr2"]["tim_path"] == canonical
        assert exported.read_text(encoding="utf-8") == canonical.read_text(
            encoding="utf-8"
        )

    def test_create_pulsar_objects_tempo2_yes_uses_track_minus_2(self, tmp_path):
        """Tempo2 derives from canonical tim and wraps par with TRACK -2."""
        par_path = tmp_path / "test.par"
        tim_path = tmp_path / "test.tim"
        par_path.write_text("PSR J1857+0943\nF0 123.456\n", encoding="utf-8")
        tim_path.write_text(
            "FORMAT 1\nMODE 1\n obs1 1400.0 58000.0 1.0 g\n", encoding="utf-8"
        )
        derived_path = tmp_path / "derived.tim"
        derived_path.write_text(
            "FORMAT 1\n toa00001 1400.0 58000.0 1.0 g -pn 42\n",
            encoding="utf-8",
        )

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
            mock_resolved.return_value.__enter__.return_value = str(derived_path)
            mock_track_par.return_value.__enter__.return_value = "/tmp/track.par"
            mock_tempopulsar.return_value = Mock()

            session_dir = tmp_path / "session"
            self.factory._create_pulsar_objects(
                file_pairs,
                file_data,
                use_pulse_numbers="yes",
                pta_file_dir=session_dir,
                canonicalize_tim=True,
            )

            mock_track_par.assert_called_once()
            resolver_args = mock_resolved.call_args
            canonical = session_dir / "epta_dr2.tim"
            assert resolver_args.args[2] == canonical
            assert resolver_args.kwargs["tim_metadata"].pn_status == "none"
            mock_tempopulsar.assert_called_once_with(
                parfile="/tmp/track.par",
                timfile=str(canonical),
                dofit=False,
            )
            assert "-pn 42" in canonical.read_text(encoding="utf-8")

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

    def test_create_pulsar_objects_skips_canonical_when_gated_off(self, tmp_path):
        """canonicalize_tim=False loads a session copy of the release .tim."""
        par_path, tim_path, file_pairs, file_data = self._write_session_inputs(
            tmp_path, "pint"
        )
        session_dir = tmp_path / "session"
        release_text = tim_path.read_text(encoding="utf-8")

        with (
            patch(
                "metapulsar.metapulsar_factory.write_canonical_tim"
            ) as mock_canonical,
            patch("metapulsar.metapulsar_factory.get_model_and_toas") as mock_get_model,
        ):
            mock_get_model.return_value = (Mock(), Mock())
            self.factory._create_pulsar_objects(
                file_pairs,
                file_data,
                use_pulse_numbers="no",
                pta_file_dir=session_dir,
                canonicalize_tim=False,
            )

        mock_canonical.assert_not_called()
        engine_tim = session_dir / "epta_dr2.tim"
        mock_get_model.assert_called_once_with(
            str(par_path),
            str(engine_tim),
            planets=True,
            allow_T2=True,
            ell1h_shapiro="full",
        )
        assert engine_tim.read_text(encoding="utf-8") == release_text
        assert "-pta epta_dr2" not in release_text

    def test_convert_jump_mjd_requires_canonicalize_tim(self):
        with pytest.raises(ValueError, match="canonicalize_tim=True"):
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
                convert_jump_mjd=True,
                canonicalize_tim=False,
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position"
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position"
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position"
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
            "metapulsar.metapulsar_factory.discover_pulsars_by_position"
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
                "metapulsar.metapulsar_factory.discover_pulsars_by_position"
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
                "metapulsar.metapulsar_factory.discover_pulsars_by_position"
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


class TestSinglePtaSharedDmxWarning:
    """Warn when shared strategy would strip DMX from a single-PTA pulsar."""

    def test_par_content_has_dmx(self):
        assert _par_content_has_dmx("PSR J1640+2224\nDMX_0001 0.01 1\n")
        assert _par_content_has_dmx("DMX 12\n")
        assert _par_content_has_dmx("DMXR1_0002 55000\n")
        assert not _par_content_has_dmx("PSR J1640+2224\nDM 18.4 1\n")
        assert not _par_content_has_dmx("DMXTHING 18.4\n")

    def test_warns_for_single_pta_shared_with_dmx(self):
        factory = MetaPulsarFactory()
        single = {
            "ng12": {
                "par": Path("/tmp/fake.par"),
                "par_content": "PSR J1640+2224\nDMX_0001 0.01 1\nDM 18.4 1\n",
            }
        }
        with pytest.warns(UserWarning, match="strips DMX"):
            factory._warn_single_pta_shared_dmx_strip(
                single, combine_components=["astrometry", "dispersion"]
            )

    def test_no_warn_for_multi_pta_or_per_pta_path(self):
        factory = MetaPulsarFactory()
        multi = {
            "a": {"par_content": "DMX_0001 0.01 1\n"},
            "b": {"par_content": "DMX_0001 0.02 1\n"},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            factory._warn_single_pta_shared_dmx_strip(
                multi, combine_components=["dispersion"]
            )
            factory._warn_single_pta_shared_dmx_strip(
                {"a": {"par_content": "DMX_0001 0.01 1\n"}},
                combine_components=["astrometry"],  # no dispersion → no strip
            )
            factory._warn_single_pta_shared_dmx_strip(
                {"a": {"par_content": "DM 18.4 1\n"}},
                combine_components=["dispersion"],
            )
        assert not any(_SINGLE_PTA_SHARED_DMX_WARNING in str(w.message) for w in caught)

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_create_metapulsar_shared_single_pta_dmx_emits_warning(
        self, mock_param_manager
    ):
        factory = MetaPulsarFactory()
        mock_manager = Mock()
        mock_manager.make_parfiles_shared.return_value = {
            "ng12": Path("/tmp/shared_ng12.par")
        }
        mock_param_manager.return_value = mock_manager

        file_data = {
            "ng12": [
                {
                    "par": Path("/tmp/J1640.par"),
                    "tim": Path("/tmp/J1640.tim"),
                    "timing_package": "tempo2",
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "par_content": "PSR J1640+2224\nDMX_0001 0.01 1\nRAJ 16:00\nDECJ 22:00\n",
                }
            ]
        }
        with (
            patch.object(factory, "_ensure_parfile_content", return_value=file_data),
            patch.object(factory, "_ensure_tim_metadata", return_value=file_data),
            patch.object(factory, "_validate_single_pulsar_data"),
            patch(
                "metapulsar.metapulsar_factory.discover_pulsars_by_position",
                return_value={"J1640+2224": {"ng12": file_data["ng12"]}},
            ),
            patch.object(
                factory, "_create_pulsar_objects", return_value={"ng12": Mock()}
            ),
            patch.object(
                factory,
                "_apply_tim_mode_transfer",
                side_effect=lambda **kw: (kw["engine_pars"], set()),
            ),
            patch("metapulsar.metapulsar_factory.MetaPulsar") as mock_mp,
        ):
            mock_mp.return_value = Mock()
            with pytest.warns(UserWarning, match="strips DMX"):
                factory.create_metapulsar(
                    file_data,
                    combination_strategy="shared",
                    combine_components=["dispersion"],
                )


class TestAlignmentPolicyForwarding:
    """The factory's only new user-facing argument (section 5 of the plan)."""

    def setup_method(self):
        self.factory = MetaPulsarFactory()

    @staticmethod
    def _file_data(timing_packages=("pint",)):
        body = (
            "PSR J1857+0943\n"
            "PEPOCH 55000\n"
            "F0 186.494081 1\n"
            "F1 -6.2e-16 1\n"
            "RAJ 18:57:36.3937 1\n"
            "DECJ +09:43:17.291 1\n"
            "POSEPOCH 55000\n"
            "DM 13.299 1\n"
            "DMEPOCH 55000\n"
            "EPHEM DE440\n"
            "CLK TT(BIPM2019)\n"
            "UNITS TDB\n"
        )
        return {
            f"pta{index}": [
                {
                    "par": Path(f"/data/pta{index}/J1857+0943.par"),
                    "tim": Path(f"/data/pta{index}/J1857+0943.tim"),
                    "timing_package": package,
                    "tim_metadata": make_tim_metadata(timespan_days=1000.0),
                    "par_content": body,
                }
            ]
            for index, package in enumerate(timing_packages)
        }

    def _run(self, mock_param_manager, **kwargs):
        mock_manager_instance = Mock()
        mock_manager_instance.make_parfiles_shared.return_value = {}
        mock_manager_instance.ell1h_shapiro = "full"
        mock_param_manager.return_value = mock_manager_instance

        with (
            patch.object(self.factory, "_create_pulsar_objects") as mock_create,
            patch("metapulsar.metapulsar_factory.MetaPulsar"),
        ):
            mock_create.return_value = {"pta0": Mock()}
            self.factory.create_metapulsar(self._file_data(), **kwargs)
        return mock_param_manager.call_args, mock_create.call_args

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_default_policy_is_none_and_manager_supplies_the_default(
        self, mock_param_manager
    ):
        manager_call, _ = self._run(mock_param_manager)
        assert manager_call[1]["alignment_policy"] is None

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_policy_is_forwarded_to_parameter_manager(self, mock_param_manager):
        from metapulsar import AlignmentPolicy

        policy = AlignmentPolicy(unsupported="error", ephem="DE421", ne_sw=4.0)
        manager_call, _ = self._run(mock_param_manager, alignment_policy=policy)
        assert manager_call[1]["alignment_policy"] is policy

    def test_policy_with_per_pta_strategy_raises(self):
        from metapulsar import AlignmentPolicy

        with pytest.raises(ValueError, match="only applies to.*shared"):
            self.factory.create_metapulsar(
                self._file_data(),
                combination_strategy="per_pta",
                alignment_policy=AlignmentPolicy(),
            )

    def test_create_all_rejects_policy_with_per_pta_before_processing(self):
        from metapulsar import AlignmentPolicy

        with (
            patch.object(self.factory, "_ensure_parfile_content") as ensure_content,
            pytest.raises(ValueError, match="only applies to.*shared"),
        ):
            self.factory.create_all_metapulsars(
                self._file_data(),
                combination_strategy="per_pta",
                alignment_policy=AlignmentPolicy(),
            )

        ensure_content.assert_not_called()

    def test_per_pta_strategy_without_policy_is_accepted(self):
        with (
            patch.object(self.factory, "_create_pulsar_objects") as mock_create,
            patch.object(
                self.factory,
                "_apply_tim_mode_transfer",
                side_effect=lambda **kw: (kw["engine_pars"], set()),
            ),
            patch("metapulsar.metapulsar_factory.MetaPulsar"),
        ):
            mock_create.return_value = {"pta0": Mock()}
            self.factory.create_metapulsar(
                self._file_data(), combination_strategy="per_pta"
            )
        assert mock_create.call_args[1]["ell1h_shapiro"] == "full"

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_mixed_engine_shared_materialization_uses_absorbed(
        self, mock_param_manager
    ):
        mock_manager_instance = Mock()
        mock_manager_instance.make_parfiles_shared.return_value = {}
        mock_manager_instance.ell1h_shapiro = "absorbed"
        mock_param_manager.return_value = mock_manager_instance

        with (
            patch.object(self.factory, "_create_pulsar_objects") as mock_create,
            patch("metapulsar.metapulsar_factory.MetaPulsar"),
        ):
            mock_create.return_value = {"pta0": Mock()}
            self.factory.create_metapulsar(self._file_data(("tempo2", "pint")))

        assert mock_create.call_args[1]["ell1h_shapiro"] == "absorbed"

    @patch("metapulsar.metapulsar_factory.ParameterManager")
    def test_pint_only_shared_materialization_uses_full(self, mock_param_manager):
        _, create_call = self._run(mock_param_manager)
        assert create_call[1]["ell1h_shapiro"] == "full"

    def test_module_level_helpers_accept_alignment_policy(self):
        import inspect

        from metapulsar.metapulsar_factory import (
            create_all_metapulsars,
            create_metapulsar,
        )

        for func in (create_metapulsar, create_all_metapulsars):
            assert "alignment_policy" in inspect.signature(func).parameters
            assert "combination_output_dir" in inspect.signature(func).parameters


class TestCombinationOutputDir:
    """Factory guards and smoke for ``combination_output_dir``."""

    def setup_method(self):
        self.factory = MetaPulsarFactory()

    def test_requires_shared_strategy(self, tmp_path):
        with pytest.raises(ValueError, match="combination_strategy='shared'"):
            self.factory.create_metapulsar(
                file_data={
                    "pta_a": [
                        {
                            "par": tmp_path / "a.par",
                            "tim": tmp_path / "a.tim",
                            "par_content": "PSR J0000+0000\n",
                            "timing_package": "pint",
                        }
                    ]
                },
                combination_strategy="per_pta",
                canonicalize_tim=True,
                combination_output_dir=tmp_path / "out",
            )

    def test_requires_canonicalize_tim(self, tmp_path):
        with pytest.raises(ValueError, match="canonicalize_tim=True"):
            self.factory.create_metapulsar(
                file_data={
                    "pta_a": [
                        {
                            "par": tmp_path / "a.par",
                            "tim": tmp_path / "a.tim",
                            "par_content": "PSR J0000+0000\n",
                            "timing_package": "pint",
                        }
                    ]
                },
                combination_strategy="shared",
                canonicalize_tim=False,
                combination_output_dir=tmp_path / "out",
            )

    @pytest.mark.unit
    def test_combination_output_dir_writes_self_contained_tree(self, tmp_path):
        import re

        from metapulsar.parameter_manager import AlignmentPolicy
        from metapulsar.metapulsar_factory import create_metapulsar

        fixture_par = (
            Path(__file__).parent / "fixtures" / "sample_parfiles" / "simple.par"
        )
        par_text = fixture_par.read_text(encoding="utf-8") + (
            "JUMP -sys TEST 0 1\n" "FD1 1.2D-03 1 3.4D-04\n"
        )
        par_path = tmp_path / "pta_a.par"
        tim_path = tmp_path / "pta_a.tim"
        par_path.write_text(par_text, encoding="utf-8")
        tim_path.write_text(
            "FORMAT 1\n"
            "test1 1400.0 54510.0 1.5 g -sys TEST\n"
            "test2 1400.0 54520.0 1.5 g -sys TEST\n",
            encoding="utf-8",
        )
        out = tmp_path / "combined"
        mp = create_metapulsar(
            file_data={
                "pta_a": [
                    {
                        "par": par_path,
                        "tim": tim_path,
                        "par_content": par_text,
                        "timing_package": "pint",
                    }
                ]
            },
            combination_strategy="shared",
            canonicalize_tim=True,
            use_pulse_numbers="no",
            alignment_policy=AlignmentPolicy(
                convention_profile="always", binary_conversion="off"
            ),
            combination_output_dir=out,
        )
        assert mp.combination_write_result is not None
        result = mp.combination_write_result
        assert result.par_path.is_file()
        assert result.tim_path.is_file()
        assert result.pn_stats is None
        stem = result.par_path.stem
        assert result.par_path == out / "par" / f"{stem}.par"
        assert result.tim_path == out / "tim" / f"{stem}.tim"
        leg_dir = out / "tim" / stem
        assert leg_dir.is_dir()
        include_targets = list(leg_dir.glob("*.tim"))
        assert len(include_targets) == 1
        comb_tim = result.tim_path.read_text(encoding="utf-8")
        assert "FORMAT 1" in comb_tim
        assert "INCLUDE" in comb_tim
        assert f"{stem}/" in comb_tim
        assert "../" not in comb_tim
        comb_par = result.par_path.read_text(encoding="utf-8")
        assert "FDJUMP1 -pta pta_a" in comb_par
        assert "1.2E-03" in comb_par
        assert not re.search(r"^JUMP\s+-pta\b", comb_par, re.M)
        assert not re.search(r"^TRACK\b", comb_par, re.M)
        # use_pulse_numbers="no" must not invent a combination PN ladder.
        include_text = include_targets[0].read_text(encoding="utf-8")
        assert "-pn" not in include_text

    @pytest.mark.unit
    def test_combination_output_dir_pulse_numbers_yes_aligns_tzr_and_pn(self, tmp_path):
        import re

        from metapulsar.parameter_manager import AlignmentPolicy
        from metapulsar.metapulsar_factory import create_metapulsar

        fixture_par = (
            Path(__file__).parent / "fixtures" / "sample_parfiles" / "simple.par"
        )
        par_text = fixture_par.read_text(encoding="utf-8") + (
            "JUMP -sys TEST 0 1\n" "FD1 1.2D-03 1 3.4D-04\n"
        )
        par_path = tmp_path / "pta_a.par"
        tim_path = tmp_path / "pta_a.tim"
        par_path.write_text(par_text, encoding="utf-8")
        tim_path.write_text(
            "FORMAT 1\n"
            "test1 1400.0 54510.0 1.5 g -sys TEST\n"
            "test2 1400.0 54520.0 1.5 g -sys TEST\n",
            encoding="utf-8",
        )
        out = tmp_path / "combined"
        mp = create_metapulsar(
            file_data={
                "pta_a": [
                    {
                        "par": par_path,
                        "tim": tim_path,
                        "par_content": par_text,
                        "timing_package": "pint",
                    }
                ]
            },
            combination_strategy="shared",
            canonicalize_tim=True,
            use_pulse_numbers="yes",
            alignment_policy=AlignmentPolicy(
                convention_profile="always", binary_conversion="off"
            ),
            combination_output_dir=out,
        )
        result = mp.combination_write_result
        assert result is not None
        assert result.pn_stats is not None
        comb_par = result.par_path.read_text(encoding="utf-8")
        assert re.search(r"^TRACK\s+-2\b", comb_par, re.M)
        assert re.search(r"^TZRMJD\b", comb_par, re.M)
        include_targets = sorted((out / "tim" / result.par_path.stem).glob("*.tim"))
        assert len(include_targets) == 1
        first_data = next(
            ln
            for ln in include_targets[0].read_text(encoding="utf-8").splitlines()
            if ln.strip()
            and not ln.strip().upper().startswith("FORMAT")
            and not ln.strip().startswith("#")
            and not ln.strip().upper().startswith("C ")
        )
        assert re.search(r"-pn\s+0(?:\s|$)", first_data)

    @pytest.mark.unit
    def test_always_profile_single_tempo2_exported_par_has_tdb(self, tmp_path):
        """Single tempo2 PTA with convention_profile=always → UNITS TDB."""
        import re

        from metapulsar.parameter_manager import AlignmentPolicy
        from metapulsar.metapulsar_factory import create_metapulsar

        body = (
            Path(__file__).parent / "fixtures" / "sample_parfiles" / "simple.par"
        ).read_text(encoding="utf-8")
        par_path = tmp_path / "pta_a.par"
        tim_path = tmp_path / "pta_a.tim"
        par_path.write_text(body, encoding="utf-8")
        tim_path.write_text(
            "FORMAT 1\ntest1 1400.0 54510.0 1.5 g -sys TEST\n",
            encoding="utf-8",
        )
        file_data = {
            "pta_a": [
                {
                    "par": par_path,
                    "tim": tim_path,
                    "par_content": body,
                    "timing_package": "tempo2",
                }
            ]
        }
        par_out = tmp_path / "par"

        with (
            patch.object(
                MetaPulsarFactory,
                "_create_pulsar_objects",
                return_value=({"pta_a": Mock()}, {}),
            ),
            patch("metapulsar.metapulsar_factory.MetaPulsar") as mock_mp,
        ):
            mock_mp.return_value = Mock(
                binary_conversion_report=None, combination_write_result=None
            )
            create_metapulsar(
                file_data=file_data,
                combination_strategy="shared",
                alignment_policy=AlignmentPolicy(
                    convention_profile="always", binary_conversion="off"
                ),
                canonicalize_tim=True,
                use_pulse_numbers="no",
                parfile_output_dir=par_out,
            )

        shared = next(par_out.glob("*_shared_*.par"))
        text = shared.read_text(encoding="utf-8")
        assert re.search(r"^UNITS\s+TDB\b", text, re.M)
        assert re.search(r"^TIMEEPH\s+FB90\b", text, re.M)
        assert re.search(r"^T2CMETHOD\s+IAU2000B\b", text, re.M)
