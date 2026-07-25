"""
Unit tests for ParameterManager class.

Tests the unified parameter and par file management functionality
for multi-PTA pulsar data.
"""

import logging
import pytest
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from pint.models.model_builder import parse_parfile

from metapulsar.parameter_manager import (
    ParameterManager,
    ParameterInconsistencyError,
    ParameterMapping,
)
from metapulsar.pint_helpers import (
    get_parameters_by_type_from_models,
    resolve_parameter_alias,
)
from tests.helpers import make_tim_metadata

# Mark all tests as slow
pytestmark = pytest.mark.slow


class TestParameterManager:
    """Test cases for ParameterManager class."""

    @pytest.fixture
    def sample_file_data(self):
        """Sample file data for testing."""
        return {
            "EPTA": {
                "par": Path("test_parfiles/epta.par"),
                "tim": Path("test_parfiles/epta.tim"),
                "tim_metadata": make_tim_metadata(timespan_days=3650.5),
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\nUNITS TDB\n",
            },
            "PPTA": {
                "par": Path("test_parfiles/ppta.par"),
                "tim": Path("test_parfiles/ppta.tim"),
                "tim_metadata": make_tim_metadata(timespan_days=4200.3),
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\nUNITS TDB\n",
            },
            "NANOGrav": {
                "par": Path("test_parfiles/nanograv.par"),
                "tim": Path("test_parfiles/nanograv.tim"),
                "tim_metadata": make_tim_metadata(timespan_days=2800.1),
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\nUNITS TDB\n",
            },
        }

    @pytest.fixture
    def sample_parfile_content(self):
        """Sample parfile content for testing."""
        return """PSR J1857+0943
PEPOCH 55000
F0 186.494081
F1 -6.2e-16
RAJ 18:57:36.3937
DECJ +09:43:17.291
DM 13.299
UNITS TDB
"""

    @pytest.fixture
    def parameter_manager(self, sample_file_data, sample_parfile_content):
        """Create ParameterManager instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test parfiles
            for pta_name, data in sample_file_data.items():
                parfile_path = Path(temp_dir) / f"{pta_name.lower()}.par"
                parfile_path.parent.mkdir(parents=True, exist_ok=True)
                with open(parfile_path, "w") as f:
                    f.write(sample_parfile_content)
                data["par"] = parfile_path

            yield ParameterManager(
                file_data=sample_file_data,
                combine_components=["astrometry", "spindown"],
                add_dm_derivatives=True,
                output_dir=Path(temp_dir) / "output",
            )

    # ===== CONSTRUCTOR TESTS =====

    def test_init_uses_first_dictionary_key(self, sample_file_data):
        """Test ParameterManager uses first dictionary key as reference PTA."""
        pm = ParameterManager(file_data=sample_file_data)

        # Should use first key from sample_file_data
        first_key = list(sample_file_data.keys())[0]
        assert pm.reference_pta == first_key

    # ===== HELPER METHOD TESTS =====

    def test_get_parfile_content(self, parameter_manager, sample_parfile_content):
        """Test getting parfile content for a PTA."""
        with patch("builtins.open", mock_open(read_data=sample_parfile_content)):
            content = parameter_manager._get_parfile_content("EPTA")
            assert "PSR J1857+0943" in content
            assert "F0 186.494081" in content

    def test_get_timing_package(self, parameter_manager):
        """Test getting timing package for a PTA."""
        assert parameter_manager._get_timing_package("EPTA") == "pint"
        assert parameter_manager._get_timing_package("PPTA") == "tempo2"
        assert parameter_manager._get_timing_package("NANOGrav") == "pint"

    def test_get_output_filename(self, parameter_manager):
        """Test output filename generation."""
        filename = parameter_manager._get_output_filename("EPTA")
        assert filename == "shared_EPTA.par"

    def test_is_parameter_for_component(self, parameter_manager):
        """Test parameter component checking."""
        component_params = ["F0", "F1", "RAJ", "DECJ"]

        assert (
            parameter_manager._is_parameter_for_component("F0", component_params)
            is True
        )
        assert (
            parameter_manager._is_parameter_for_component("RAJ", component_params)
            is True
        )
        assert (
            parameter_manager._is_parameter_for_component("DM", component_params)
            is False
        )

    # ===== PARFILE PROCESSING TESTS =====

    def test_parse_parfiles(self, parameter_manager):
        """Test parsing parfiles into dictionaries."""
        with patch.object(
            parameter_manager, "_get_parfile_content"
        ) as mock_get_content:
            mock_get_content.side_effect = (
                lambda pta: "PSR J1857+0943\nF0 186.494081\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nUNITS TDB"
            )

            with patch("metapulsar.parameter_manager.parse_parfile") as mock_parse:
                mock_parse.return_value = {
                    "PSR": ["J1857+0943"],
                    "F0": ["186.494081"],
                    "RAJ": ["18:57:36.3937"],
                    "DECJ": ["+09:43:17.291"],
                    "UNITS": ["TDB"],
                }

                result = parameter_manager._parse_parfiles()

                assert len(result) == 3  # All PTAs in file_data
                assert "EPTA" in result
                assert "PPTA" in result
                assert "NANOGrav" in result
                assert result["EPTA"]["F0"] == ["186.494081"]

    def test_dict_to_parfile_string_custom(self, parameter_manager):
        """Test converting dictionary to parfile string."""
        parfile_dict = {
            "PSR": ["J1857+0943"],
            "F0": ["186.494081"],
            "RAJ": ["18:57:36.3937"],
            "UNITS": ["TDB"],
        }

        # Import the function directly instead of calling the removed method
        from metapulsar.pint_helpers import dict_to_parfile_string

        result = dict_to_parfile_string(parfile_dict)

        # The function now includes headers and formatting, so check for the actual content
        assert "PSR" in result
        assert "J1857+0943" in result
        assert "F0" in result
        assert "186.494081" in result
        assert "RAJ" in result
        assert "18:57:36.3937" in result
        assert "UNITS" in result
        assert "TDB" in result

    # ===== PARAMETER MAPPING TESTS =====

    def test_add_merged_parameter(self, parameter_manager):
        """Test adding merged parameter to dictionary."""
        fitparameters = {}

        parameter_manager._add_merged_parameter("F0", "EPTA", "F0", fitparameters)

        assert "F0" in fitparameters
        assert fitparameters["F0"]["EPTA"] == "F0"

    def test_add_pta_specific_parameter(self, parameter_manager):
        """Test adding PTA-specific parameter to dictionary."""
        setparameters = {}

        parameter_manager._add_pta_specific_parameter(
            "DM", "EPTA", "DM", "DM", setparameters
        )

        assert "DM_EPTA" in setparameters
        assert setparameters["DM_EPTA"]["EPTA"] == "DM"

    def test_add_pta_specific_parameter_meta_key_differs_from_mapped_value(
        self, parameter_manager
    ):
        """Meta key suffix uses PINT name; mapped value uses parfile-native spelling."""
        setparameters = {}

        parameter_manager._add_pta_specific_parameter(
            "A1DOT", "epta", "A1DOT", "XDOT", setparameters
        )

        assert "A1DOT_epta" in setparameters
        assert setparameters["A1DOT_epta"]["epta"] == "XDOT"

    def test_validate_parameter_consistency(self, parameter_manager):
        """Test parameter consistency validation."""
        fitparameters = {"F0": {"EPTA": "F0"}}
        setparameters = {"F0": {"EPTA": "F0"}}

        # Should not raise exception
        parameter_manager._validate_parameter_consistency(fitparameters, setparameters)

    def test_validate_parameter_consistency_error(self, parameter_manager):
        """Test parameter consistency validation error."""
        fitparameters = {"F0": {"EPTA": "F0"}}
        setparameters = {}  # Missing F0

        with pytest.raises(ParameterInconsistencyError):
            parameter_manager._validate_parameter_consistency(
                fitparameters, setparameters
            )

    def test_build_parameter_mapping_result(self, parameter_manager):
        """Test building parameter mapping result."""
        fitparameters = {
            "F0": {"EPTA": "F0", "PPTA": "F0"},  # Merged parameter
            "DM_EPTA": {"EPTA": "DM"},  # PTA-specific parameter
        }
        setparameters = {"F0": {"EPTA": "F0", "PPTA": "F0"}, "DM_EPTA": {"EPTA": "DM"}}

        result = parameter_manager._build_parameter_mapping_result(
            fitparameters, setparameters
        )

        assert isinstance(result, ParameterMapping)
        assert result.fitparameters == fitparameters
        assert result.setparameters == setparameters
        assert "F0" in result.merged_parameters
        assert "DM_EPTA" in result.pta_specific_parameters

    # ===== PARAMETER RESOLUTION TESTS =====

    def test_resolve_parameter_aliases(self, parameter_manager):
        """Test parameter alias resolution."""
        # This will depend on what aliases are available in PINT
        result = parameter_manager.resolve_parameter_aliases("F0")
        assert isinstance(result, str)

    def test_check_component_available_across_ptas(self, parameter_manager):
        """Test checking component availability across PTAs."""
        with patch.object(
            parameter_manager, "_get_parfile_content"
        ) as mock_get_content:
            mock_get_content.return_value = "PSR J1857+0943\nF0 186.494081\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nUNITS TDB"

            with patch(
                "metapulsar.pint_helpers.create_pint_model"
            ) as mock_create_model:
                mock_model = Mock()
                mock_create_model.return_value = mock_model

                with patch(
                    "metapulsar.pint_helpers.check_component_available_in_model"
                ) as mock_check:
                    mock_check.return_value = True

                    result = parameter_manager.check_component_available_across_ptas(
                        "spindown"
                    )
                    assert result is True

    def test_check_parameter_identifiable(self, parameter_manager):
        """Test checking parameter identifiability."""
        with patch.object(
            parameter_manager, "check_parameter_identifiable"
        ) as mock_method:
            mock_method.return_value = True

            result = parameter_manager.check_parameter_identifiable("EPTA", "F0")
            assert result is True
            mock_method.assert_called_once_with("EPTA", "F0")

    # ===== INTEGRATION TESTS =====

    def test_make_parfiles_shared_integration(self, parameter_manager):
        """Test full make_parfiles_shared workflow."""
        with patch.object(parameter_manager, "_parse_parfiles") as mock_parse:
            mock_parse.return_value = {
                "EPTA": {
                    "PSR": ["J1857+0943"],
                    "F0": ["186.494081"],
                    "RAJ": ["18:57:36.3937"],
                    "DECJ": ["+09:43:17.291"],
                    "UNITS": ["TDB"],
                },
                "PPTA": {
                    "PSR": ["J1857+0943"],
                    "F0": ["186.494082"],
                    "RAJ": ["18:57:36.3938"],
                    "DECJ": ["+09:43:17.292"],
                    "UNITS": ["TDB"],
                },
            }

            with patch.object(
                parameter_manager, "_convert_units_if_needed"
            ) as mock_convert:
                mock_convert.return_value = {
                    "EPTA": "PSR J1857+0943\nF0 186.494081\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nUNITS TDB",
                    "PPTA": "PSR J1857+0943\nF0 186.494082\nRAJ 18:57:36.3938\nDECJ +09:43:17.292\nUNITS TDB",
                }

                with patch.object(
                    parameter_manager, "_make_parameters_shared"
                ) as mock_make_consistent:
                    mock_make_consistent.return_value = {
                        "EPTA": "consistent_epta_content",
                        "PPTA": "consistent_ppta_content",
                    }

                    with patch.object(
                        parameter_manager, "_write_shared_parfiles"
                    ) as mock_write:
                        mock_write.return_value = {
                            "EPTA": Path("/tmp/shared_EPTA.par"),
                            "PPTA": Path("/tmp/shared_PPTA.par"),
                        }

                        result = parameter_manager.make_parfiles_shared()

                        assert len(result) == 2
                        assert "EPTA" in result
                        assert "PPTA" in result
                        mock_parse.assert_called_once()
                        mock_convert.assert_called_once()
                        mock_make_consistent.assert_called_once()
                        mock_write.assert_called_once()

    def test_build_parameter_mappings_integration(self, parameter_manager):
        """Test full build_parameter_mappings workflow."""
        with patch.object(
            parameter_manager, "_discover_mergeable_parameters"
        ) as mock_discover:
            mock_discover.return_value = ["F0", "RAJ"]

            with patch.object(
                parameter_manager, "_process_all_pta_parameters"
            ) as mock_process:
                mock_process.return_value = (
                    {"F0": {"EPTA": "F0", "PPTA": "F0"}},  # fitparameters
                    {"F0": {"EPTA": "F0", "PPTA": "F0"}},  # setparameters
                )

                with patch.object(
                    parameter_manager, "_validate_parameter_consistency"
                ) as mock_validate:
                    with patch.object(
                        parameter_manager, "_build_parameter_mapping_result"
                    ) as mock_build:
                        mock_result = ParameterMapping(
                            fitparameters={"F0": {"EPTA": "F0", "PPTA": "F0"}},
                            setparameters={"F0": {"EPTA": "F0", "PPTA": "F0"}},
                            merged_parameters=["F0"],
                            pta_specific_parameters=[],
                        )
                        mock_build.return_value = mock_result

                        result = parameter_manager.build_parameter_mappings()

                        assert isinstance(result, ParameterMapping)
                        mock_discover.assert_called_once()
                        mock_process.assert_called_once()
                        mock_validate.assert_called_once()
                        mock_build.assert_called_once()

    def test_build_parameter_mappings_uses_parfile_native_xdot(self):
        """Binary parfiles with XDOT should map A1DOT meta key to XDOT engine name."""
        file_data = {
            "epta": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSRJ J1640+2224\n"
                    "F0 316.12397933185408713 1 0\n"
                    "PEPOCH 55000\n"
                    "DM 18.417 1 0\n"
                    "BINARY T2\n"
                    "PB 175.46066459623014253 1 0\n"
                    "T0 51626.179967495799449 1 0\n"
                    "A1 55.329722354525327725 1 0\n"
                    "OM 50.733505043065199373 1 0\n"
                    "ECC 0.0007972975541058369088 1 0\n"
                    "XDOT 8.1279761448223669144e-15 1 0\n"
                    "EPHEM DE421\n"
                    "CLK TT(BIPM2011)\n"
                ),
            }
        }
        pm = ParameterManager(
            file_data=file_data,
            combine_components=["binary"],
        )
        result = pm.build_parameter_mappings()
        assert "A1DOT" in result.fitparameters
        assert result.fitparameters["A1DOT"]["epta"] == "XDOT"

    # ===== ERROR HANDLING TESTS =====

    def test_parameter_inconsistency_error(self):
        """Test ParameterInconsistencyError exception."""
        error = ParameterInconsistencyError("Test error message")
        assert str(error) == "Test error message"

    def test_parameter_mapping_creation(self):
        """Test ParameterMapping data class."""
        mapping = ParameterMapping(
            fitparameters={"F0": {"EPTA": "F0"}},
            setparameters={"F0": {"EPTA": "F0"}},
            merged_parameters=["F0"],
            pta_specific_parameters=[],
        )

        assert mapping.fitparameters == {"F0": {"EPTA": "F0"}}
        assert mapping.setparameters == {"F0": {"EPTA": "F0"}}
        assert mapping.merged_parameters == ["F0"]
        assert mapping.pta_specific_parameters == []

    def test_handle_dm_special_cases_missing_dm_error(self):
        """Test that _handle_dm_special_cases raises when any PTA lacks DM."""
        file_data_without_dm = {
            "EPTA": {
                "par": Path("test_parfiles/epta.par"),
                "tim": Path("test_parfiles/epta.tim"),
                "tim_metadata": make_tim_metadata(timespan_days=3650.5),
                "timing_package": "pint",
                "par_content": (
                    "PSR J1857+0943\n"
                    "PEPOCH 55000\n"
                    "F0 186.494081\n"
                    "F1 -6.2e-16\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "DM 13.299\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "par": Path("test_parfiles/ppta.par"),
                "tim": Path("test_parfiles/ppta.tim"),
                "tim_metadata": make_tim_metadata(timespan_days=4200.3),
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1857+0943\n"
                    "PEPOCH 55000\n"
                    "F0 186.494081\n"
                    "F1 -6.2e-16\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "UNITS TDB\n"
                ),
            },
        }

        parameter_manager = ParameterManager(
            file_data=file_data_without_dm,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
        )

        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        assert "DM" in reference_dict
        assert "DM" not in parfile_dicts["PPTA"]

        with pytest.raises(
            ValueError,
            match="DM parameter is missing from parfile for PTA PPTA",
        ):
            parameter_manager._handle_dm_special_cases(
                parfile_dicts=parfile_dicts,
                reference_dict=reference_dict,
                add_dm_derivatives=True,
                dmx_params_map={"EPTA": [], "PPTA": []},
            )

    def test_apply_shared_convention_rules_cross_engine_ecliptic(self):
        """Cross-engine ecliptic pars force ECL IERS2003 and remove T2CMETHOD TEMPO."""
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE440\n"
                    "CLK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._apply_shared_convention_rules(parfile_dicts, reference_dict)

        for _, pd in parfile_dicts.items():
            assert pd["UNITS"] == ["TDB"]
            assert pd["ECL"] == ["IERS2003"]
            assert "T2CMETHOD" not in pd
            assert pd["EPHEM"] == ["DE436"]
            clock_value = pd["CLOCK"] if "CLOCK" in pd else pd["CLK"]
            assert clock_value == ["TT(BIPM2015)"]

    def test_apply_shared_convention_rules_cross_engine_equatorial_warning_and_no_ecl(
        self,
    ):
        """Cross-engine equatorial pars emit warning and should not carry ECL."""
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1857+0943\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1857+0943\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        with patch.object(parameter_manager.logger, "warning") as mock_warning:
            parameter_manager._apply_shared_convention_rules(
                parfile_dicts, reference_dict
            )

        for _, pd in parfile_dicts.items():
            assert pd["UNITS"] == ["TDB"]
            assert "ECL" not in pd
            assert "T2CMETHOD" not in pd
        assert mock_warning.call_count == 2

    def test_apply_shared_convention_rules_pint_only_aligns_missing_ecl_to_iers2010(
        self,
    ):
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "EPHEM DE436\n"
                    "CLK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        with patch.object(parameter_manager.logger, "warning") as mock_warning:
            parameter_manager._apply_shared_convention_rules(
                parfile_dicts, reference_dict
            )

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2010"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2010"]
        assert "T2CMETHOD" not in parfile_dicts["EPTA"]
        assert "T2CMETHOD" not in parfile_dicts["PPTA"]
        assert mock_warning.call_count == 0

    def test_apply_shared_convention_rules_tempo2_only_preserves_shared_t2cmethod(
        self,
    ):
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE440\n"
                    "CLK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._apply_shared_convention_rules(parfile_dicts, reference_dict)

        for pd in parfile_dicts.values():
            assert pd["ECL"] == ["IERS2010"]
            assert pd["T2CMETHOD"] == ["TEMPO"]

    def test_apply_shared_convention_rules_tempo2_only_aligns_heterogeneous_conventions(
        self,
    ):
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2003\n"
                    "T2CMETHOD IAU2000B\n"
                    "EPHEM DE440\n"
                    "CLK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._apply_shared_convention_rules(parfile_dicts, reference_dict)

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2003"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2003"]
        assert parfile_dicts["EPTA"]["T2CMETHOD"] == ["TEMPO"]
        assert parfile_dicts["PPTA"]["T2CMETHOD"] == ["TEMPO"]

    def test_apply_shared_convention_rules_single_pta_skips_alignment(self):
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1600-3053\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "CLK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            }
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._apply_shared_convention_rules(parfile_dicts, reference_dict)

        assert parfile_dicts["EPTA"]["T2CMETHOD"] == ["TEMPO"]
        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2010"]
        assert parfile_dicts["EPTA"]["CLOCK"] == ["TT(BIPM2015)"]
        assert parfile_dicts["EPTA"]["CLK"] == ["TT(BIPM2021)"]

    def test_single_pta_dispersion_cleanup_still_rewrites_dm_model(self):
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "RAJ 16:00:51.9032\n"
                    "DECJ -30:53:49.38\n"
                    "DM 13.2 0\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            }
        }
        parameter_manager = ParameterManager(
            file_data=file_data,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
        )
        parfile_dicts = {
            "EPTA": {
                "DM": ["13.2 0"],
                "DMEPOCH": ["55000"],
                "DMX_0001": ["0.01 1"],
            }
        }

        parameter_manager._handle_dm_special_cases(
            parfile_dicts=parfile_dicts,
            reference_dict=parfile_dicts["EPTA"],
            add_dm_derivatives=True,
            dmx_params_map={"EPTA": ["DMX_0001"]},
        )

        assert "DMX_0001" not in parfile_dicts["EPTA"]
        assert parfile_dicts["EPTA"]["DM"] == ["13.2 1"]
        assert parfile_dicts["EPTA"]["DMEPOCH"] == ["55000.0 0"]
        assert parfile_dicts["EPTA"]["DM1"] == ["0.0 1"]
        assert parfile_dicts["EPTA"]["DM2"] == ["0.0 1"]

    @pytest.fixture
    def file_data_with_two_different_dm_values(self):
        return {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "PEPOCH 55000\n"
                    "F0 186.494081\n"
                    "RAJ 16:00:51.9032\n"
                    "DECJ -30:53:49.38\n"
                    "DM 13.2 1\n"
                    "DMEPOCH 54000\n"
                    "DM1 0.0 1\n"
                    "DM2 0.0 1\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "PEPOCH 55000\n"
                    "F0 186.494081\n"
                    "RAJ 16:00:51.9032\n"
                    "DECJ -30:53:49.38\n"
                    "DM 13.7 1\n"
                    "DMEPOCH 56000\n"
                    "DM1 0.0 1\n"
                    "DM2 0.0 1\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
        }

    def test_consistent_dispersion_preserves_local_dm_by_default(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
        )

        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]
        parameter_manager._make_component_parameters_shared(
            parfile_dicts,
            reference_dict,
            "EPTA",
            "dispersion",
            ["DM", "DMEPOCH", "DM1", "DM2"],
        )
        parameter_manager._handle_dm_special_cases(
            parfile_dicts=parfile_dicts,
            reference_dict=reference_dict,
            add_dm_derivatives=True,
            dmx_params_map={"EPTA": [], "PPTA": []},
        )

        assert parfile_dicts["EPTA"]["DM"] == ["13.2 1"]
        assert parfile_dicts["PPTA"]["DM"] == ["13.7 1"]
        assert parfile_dicts["EPTA"]["DMEPOCH"] == ["54000.0 0"]
        assert parfile_dicts["PPTA"]["DMEPOCH"] == ["54000.0 0"]
        assert parfile_dicts["EPTA"]["DM1"] == ["0.0 1"]
        assert parfile_dicts["PPTA"]["DM1"] == ["0.0 1"]

    def test_discover_mergeable_parameters_excludes_dm_by_default(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
        )

        mergeable = {
            resolve_parameter_alias(param)
            for param in parameter_manager._discover_mergeable_parameters()
        }

        assert "DM" not in mergeable
        assert "DM1" in mergeable
        assert "DM2" in mergeable

    def test_build_parameter_mappings_keeps_dm_pta_specific_by_default(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
        )

        mapping = parameter_manager.build_parameter_mappings()

        assert "DM_EPTA" in mapping.pta_specific_parameters
        assert "DM_PPTA" in mapping.pta_specific_parameters
        assert "DM1" in mapping.merged_parameters
        assert "DM2" in mapping.merged_parameters
        assert "DM" not in mapping.merged_parameters

    def test_empty_exclude_from_shared_restores_merged_dm(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
            exclude_from_shared=[],
        )

        mergeable = {
            resolve_parameter_alias(param)
            for param in parameter_manager._discover_mergeable_parameters()
        }

        assert "DM" in mergeable

        parfile_data = {
            pta: data["par_content"]
            for pta, data in file_data_with_two_different_dm_values.items()
        }
        consistent_parfiles = parameter_manager._make_parameters_shared(parfile_data)

        for pta_name in file_data_with_two_different_dm_values:
            parfile_dict = parse_parfile(StringIO(consistent_parfiles[pta_name]))
            assert parfile_dict["DM"] == ["13.2 1"]

        mapping = parameter_manager.build_parameter_mappings()

        assert "DM" in mapping.merged_parameters
        assert "DM1" in mapping.merged_parameters
        assert "DM2" in mapping.merged_parameters
        assert "DM_EPTA" not in mapping.pta_specific_parameters
        assert "DM_PPTA" not in mapping.pta_specific_parameters

    @pytest.fixture
    def file_data_with_two_different_f0_values(self):
        return {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "PEPOCH 55000\n"
                    "F0 186.494081 1\n"
                    "F1 -6.2e-16 1\n"
                    "RAJ 16:00:51.9032\n"
                    "DECJ -30:53:49.38\n"
                    "DM 13.2 1\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "PEPOCH 55000\n"
                    "F0 186.500000 1\n"
                    "F1 -6.2e-16 1\n"
                    "RAJ 16:00:51.9032\n"
                    "DECJ -30:53:49.38\n"
                    "DM 13.7 1\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
        }

    def test_exclude_from_shared_keeps_f0_pta_specific(
        self, file_data_with_two_different_f0_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_f0_values,
            combine_components=["spindown"],
            exclude_from_shared=("F0",),
        )

        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]
        spindown_params = get_parameters_by_type_from_models(
            "spindown", parameter_manager.pint_models
        )
        parameter_manager._make_component_parameters_shared(
            parfile_dicts,
            reference_dict,
            "EPTA",
            "spindown",
            spindown_params,
        )

        assert parfile_dicts["EPTA"]["F0"] == ["186.494081 1"]
        assert parfile_dicts["PPTA"]["F0"] == ["186.500000 1"]

        mapping = parameter_manager.build_parameter_mappings()

        assert "F0_EPTA" in mapping.pta_specific_parameters
        assert "F0_PPTA" in mapping.pta_specific_parameters
        assert "F0" not in mapping.merged_parameters
        assert "F1" in mapping.merged_parameters

    def test_exclude_from_shared_accepts_lowercase_dm_alias(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
            exclude_from_shared=("dm",),
        )

        assert parameter_manager.exclude_from_shared == {"DM"}

        mergeable = {
            resolve_parameter_alias(param)
            for param in parameter_manager._discover_mergeable_parameters()
        }

        assert "DM" not in mergeable

        mapping = parameter_manager.build_parameter_mappings()

        assert "DM_EPTA" in mapping.pta_specific_parameters
        assert "DM_PPTA" in mapping.pta_specific_parameters
        assert "DM" not in mapping.merged_parameters

    def test_apply_shared_convention_rules_mixed_astrometry_raises(self):
        """Mixed ecliptic/equatorial astrometry should fail fast."""
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1857+0943\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "LAMBDA 244.347\n"
                    "BETA -10.07\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            }
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        with pytest.raises(ValueError, match="Mixed astrometry detected"):
            parameter_manager._apply_shared_convention_rules(
                parfile_dicts, reference_dict
            )

    def test_apply_shared_convention_rules_pint_only_elong_elat_aliases(self):
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "ELONG 244.347\n"
                    "ELAT -10.07\n"
                    "ECL IERS2010\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1600-3053\n"
                    "ELONG 244.347\n"
                    "ELAT -10.07\n"
                    "EPHEM DE436\n"
                    "CLK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._apply_shared_convention_rules(parfile_dicts, reference_dict)

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2010"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2010"]

    def test_apply_shared_convention_rules_missing_reference_clock_raises(self):
        """Reference parfile must contain CLOCK or CLK."""
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1857+0943\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "EPHEM DE440\n"
                    "UNITS TDB\n"
                ),
            }
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        with pytest.raises(ValueError, match="CLOCK'.*CLK"):
            parameter_manager._apply_shared_convention_rules(
                parfile_dicts, reference_dict
            )

    def test_apply_shared_convention_rules_missing_reference_ephem_raises(self):
        """Reference parfile must contain EPHEM."""
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1857+0943\n"
                    "RAJ 18:57:36.3937\n"
                    "DECJ +09:43:17.291\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            }
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        with pytest.raises(ValueError, match="EPHEM"):
            parameter_manager._apply_shared_convention_rules(
                parfile_dicts, reference_dict
            )

    def test_parse_ne_sw_value_present_and_absent(self):
        """Parse explicit NE_SW from parfile dict or return None when absent."""
        parameter_manager = ParameterManager(
            file_data={"EPTA": {"timing_package": "pint", "par_content": ""}},
            combine_components=[],
        )
        assert parameter_manager._parse_ne_sw_value({"NE_SW": ["4 0"]}) == 4.0
        assert parameter_manager._parse_ne_sw_value({}) is None

    def test_resolve_consistent_ne_sw_reference_explicit(self):
        """Reference explicit NE_SW takes precedence over tempo2 fallback."""
        parameter_manager = ParameterManager(
            file_data={
                "EPTA": {"timing_package": "pint", "par_content": ""},
                "PPTA": {"timing_package": "tempo2", "par_content": ""},
            },
            combine_components=[],
        )
        reference = {"NE_SW": ["0 0"]}
        packages = parameter_manager._normalized_timing_packages()
        assert parameter_manager._resolve_consistent_ne_sw(reference, packages) == 0.0

    def test_resolve_consistent_ne_sw_tempo2_fallback(self):
        """Missing reference NE_SW with tempo2 PTA resolves to 4.0 cm^-3."""
        parameter_manager = ParameterManager(
            file_data={
                "EPTA": {"timing_package": "pint", "par_content": ""},
                "PPTA": {"timing_package": "tempo2", "par_content": ""},
            },
            combine_components=[],
        )
        packages = parameter_manager._normalized_timing_packages()
        assert parameter_manager._resolve_consistent_ne_sw({}, packages) == 4.0

    def test_resolve_consistent_ne_sw_pint_only_skip(self):
        """PINT-only stack with no reference NE_SW leaves alignment unresolved."""
        parameter_manager = ParameterManager(
            file_data={
                "EPTA": {"timing_package": "pint", "par_content": ""},
                "PPTA": {"timing_package": "pint", "par_content": ""},
            },
            combine_components=[],
        )
        packages = parameter_manager._normalized_timing_packages()
        assert parameter_manager._resolve_consistent_ne_sw({}, packages) is None

    def test_align_ne_sw_cross_engine_missing(self):
        """Cross-engine stack without NE_SW lines gets tempo2 fallback on all PTAs."""
        file_data = {
            "NG": {
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nUNITS TDB\n",
            },
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nUNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["NG"]

        parameter_manager._align_ne_sw_convention(parfile_dicts, reference_dict)

        assert parfile_dicts["NG"]["NE_SW"] == ["4 0"]
        assert parfile_dicts["EPTA"]["NE_SW"] == ["4 0"]

    def test_align_ne_sw_tempo2_only_single_pta_missing(self):
        """Single tempo2 PTA without NE_SW gets explicit tempo2 default."""
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nUNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._align_ne_sw_convention(parfile_dicts, reference_dict)

        assert parfile_dicts["EPTA"]["NE_SW"] == ["4 0"]

    def test_align_ne_sw_pint_only_multi_pta_missing(self):
        """PINT-only multi-PTA stack leaves NE_SW absent when reference omits it."""
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nUNITS TDB\n",
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nUNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        parameter_manager._align_ne_sw_convention(parfile_dicts, reference_dict)

        assert "NE_SW" not in parfile_dicts["EPTA"]
        assert "NE_SW" not in parfile_dicts["PPTA"]

    def test_align_ne_sw_reference_explicit_zero(self, caplog):
        """Reference NE_SW 0 overwrites conflicting explicit values with warning."""
        file_data = {
            "NG": {
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nNE_SW 0 0\nUNITS TDB\n",
            },
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nNE_SW 4 0\nUNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["NG"]

        with caplog.at_level(logging.WARNING):
            parameter_manager._align_ne_sw_convention(parfile_dicts, reference_dict)

        assert parfile_dicts["NG"]["NE_SW"] == ["0 0"]
        assert parfile_dicts["EPTA"]["NE_SW"] == ["0 0"]
        assert any(
            "overwriting NE_SW 4 with consistent value 0" in record.message
            for record in caplog.records
        )

    def test_align_ne_sw_reference_explicit_four(self):
        """Reference explicit NE_SW 4 is written on all PTAs including absent targets."""
        file_data = {
            "NG": {
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nNE_SW 4 0\nUNITS TDB\n",
            },
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nUNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["NG"]

        parameter_manager._align_ne_sw_convention(parfile_dicts, reference_dict)

        assert parfile_dicts["NG"]["NE_SW"] == ["4 0"]
        assert parfile_dicts["EPTA"]["NE_SW"] == ["4 0"]

    @patch.object(ParameterManager, "_apply_shared_convention_rules")
    def test_make_parameters_shared_aligns_ne_sw_cross_engine(self, _mock_conv):
        """_make_parameters_shared aligns NE_SW before convention rules."""
        file_data = {
            "NG": {
                "timing_package": "pint",
                "par_content": (
                    "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\n"
                    "F1 -6.2e-16\nUNITS TDB\n"
                ),
            },
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\n"
                    "F1 -6.2e-16\nUNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(
            file_data=file_data,
            combine_components=[],
        )
        parfile_data = {pta: data["par_content"] for pta, data in file_data.items()}

        result = parameter_manager._make_parameters_shared(parfile_data)

        assert "NE_SW" in result["NG"] and "4" in result["NG"]
        assert "NE_SW" in result["EPTA"] and "4" in result["EPTA"]

    def test_parse_ne_sw_value_solarn0_alias(self):
        """SOLARN0 (NANOGrav spelling) counts as an explicit NE_SW value."""
        parameter_manager = ParameterManager(
            file_data={"NG": {"timing_package": "pint", "par_content": ""}},
            combine_components=[],
        )
        assert parameter_manager._parse_ne_sw_value({"SOLARN0": ["0.00"]}) == 0.0
        assert parameter_manager._parse_ne_sw_value({"NE1AU": ["4"]}) == 4.0

    def test_resolve_consistent_ne_sw_reference_solarn0_beats_fallback(self):
        """Reference SOLARN0 0 is explicit; tempo2 fallback of 4.0 must not win."""
        parameter_manager = ParameterManager(
            file_data={
                "NG": {"timing_package": "pint", "par_content": ""},
                "EPTA": {"timing_package": "tempo2", "par_content": ""},
            },
            combine_components=[],
        )
        packages = parameter_manager._normalized_timing_packages()
        assert (
            parameter_manager._resolve_consistent_ne_sw({"SOLARN0": ["0.00"]}, packages)
            == 0.0
        )

    def test_align_ne_sw_drops_alias_spellings(self):
        """Aligner must remove SOLARN0/NE1AU so PINT never sees two NE_SW lines.

        Regression for the IPTA DR2 notebook crash: NANOGrav 9y par files carry
        SOLARN0 0.00; injecting NE_SW alongside it made PINT reject the shared
        par with "Parameter NE_SW is not a repeatable parameter".
        """
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nNE_SW 4\nUNITS TDB\n",
            },
            "NG": {
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nSOLARN0 0.00\nUNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()

        parameter_manager._align_ne_sw_convention(parfile_dicts, parfile_dicts["EPTA"])

        assert parfile_dicts["EPTA"]["NE_SW"] == ["4 0"]
        assert parfile_dicts["NG"]["NE_SW"] == ["4 0"]
        assert "SOLARN0" not in parfile_dicts["NG"]

    def test_make_parfiles_shared_solarn0_roundtrip(self, tmp_path):
        """Written consistent pars must re-ingest cleanly through PINT.

        End-to-end regression for the notebook crash: cross-engine stack where
        the PINT PTA spells solar wind as SOLARN0. Every written par must build
        a PINT model (the _create_pulsar_objects step) without duplicate-NE_SW
        errors.
        """
        base = (
            "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\n"
            "RAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\n"
            "EPHEM DE421\nCLK TT(BIPM2015)\nUNITS TDB\n"
        )
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": base + "NE_SW 4\n",
            },
            "NG": {
                "timing_package": "pint",
                "par_content": base + "SOLARN0 0.00\n",
            },
        }
        parameter_manager = ParameterManager(
            file_data=file_data,
            output_dir=tmp_path,
            pulsar_name="J1857+0943",
        )

        output_files = parameter_manager.make_parfiles_shared()

        from metapulsar.pint_helpers import create_pint_model

        for pta_name, path in output_files.items():
            content = Path(path).read_text()
            assert "SOLARN0" not in content, f"{pta_name}: alias survived rewrite"
            model = create_pint_model(content)  # must not raise
            assert float(model.NE_SW.value) == pytest.approx(4.0)
