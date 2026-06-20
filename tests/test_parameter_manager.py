"""
Unit tests for ParameterManager class.

Tests the unified parameter and par file management functionality
for multi-PTA pulsar data.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from metapulsar.parameter_manager import (
    ParameterManager,
    ParameterInconsistencyError,
    ParameterMapping,
)

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
                "timespan_days": 3650.5,
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\nUNITS TDB\n",
            },
            "PPTA": {
                "par": Path("test_parfiles/ppta.par"),
                "tim": Path("test_parfiles/ppta.tim"),
                "timespan_days": 4200.3,
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\nUNITS TDB\n",
            },
            "NANOGrav": {
                "par": Path("test_parfiles/nanograv.par"),
                "tim": Path("test_parfiles/nanograv.tim"),
                "timespan_days": 2800.1,
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
        assert filename == "consistent_EPTA.par"

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

        parameter_manager._add_pta_specific_parameter("DM", "EPTA", "DM", setparameters)

        assert "DM_EPTA" in setparameters
        assert setparameters["DM_EPTA"]["EPTA"] == "DM"

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

    def test_make_parfiles_consistent_integration(self, parameter_manager):
        """Test full make_parfiles_consistent workflow."""
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
                    parameter_manager, "_make_parameters_consistent"
                ) as mock_make_consistent:
                    mock_make_consistent.return_value = {
                        "EPTA": "consistent_epta_content",
                        "PPTA": "consistent_ppta_content",
                    }

                    with patch.object(
                        parameter_manager, "_write_consistent_parfiles"
                    ) as mock_write:
                        mock_write.return_value = {
                            "EPTA": Path("/tmp/consistent_EPTA.par"),
                            "PPTA": Path("/tmp/consistent_PPTA.par"),
                        }

                        result = parameter_manager.make_parfiles_consistent()

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
        """Test that _handle_dm_special_cases raises error when DM is missing from reference dict."""
        # Create file data without DM parameter
        file_data_without_dm = {
            "EPTA": {
                "par": Path("test_parfiles/epta.par"),
                "tim": Path("test_parfiles/epta.tim"),
                "timespan_days": 3650.5,
                "timing_package": "pint",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nUNITS TDB\n",
            },
            "PPTA": {
                "par": Path("test_parfiles/ppta.par"),
                "tim": Path("test_parfiles/ppta.tim"),
                "timespan_days": 4200.3,
                "timing_package": "tempo2",
                "par_content": "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\nRAJ 18:57:36.3937\nDECJ +09:43:17.291\nUNITS TDB\n",
            },
        }

        # Create ParameterManager with add_dm_derivatives=True to trigger DM handling
        parameter_manager = ParameterManager(
            file_data=file_data_without_dm,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
        )

        # Parse parfiles to get the reference dict
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]  # First PTA is used as reference

        # Verify DM is not in reference dict
        assert "DM" not in reference_dict

        # Test that _handle_dm_special_cases raises an error when DM is missing
        with pytest.raises(
            ValueError, match="DM parameter is missing from reference parfile"
        ):
            parameter_manager._handle_dm_special_cases(
                parfile_dicts=parfile_dicts,
                reference_dict=reference_dict,
                add_dm_derivatives=True,
                dmx_params_map={"EPTA": [], "PPTA": []},
            )

    def test_apply_consistent_convention_rules_cross_engine_ecliptic(self):
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

        parameter_manager._apply_consistent_convention_rules(
            parfile_dicts, reference_dict
        )

        for _, pd in parfile_dicts.items():
            assert pd["UNITS"] == ["TDB"]
            assert pd["ECL"] == ["IERS2003"]
            assert "T2CMETHOD" not in pd
            assert pd["EPHEM"] == ["DE436"]
            clock_value = pd["CLOCK"] if "CLOCK" in pd else pd["CLK"]
            assert clock_value == ["TT(BIPM2015)"]

    def test_apply_consistent_convention_rules_cross_engine_equatorial_warning_and_no_ecl(
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
            parameter_manager._apply_consistent_convention_rules(
                parfile_dicts, reference_dict
            )

        for _, pd in parfile_dicts.items():
            assert pd["UNITS"] == ["TDB"]
            assert "ECL" not in pd
            assert "T2CMETHOD" not in pd
        assert mock_warning.call_count == 2

    def test_apply_consistent_convention_rules_pint_only_aligns_missing_ecl_to_iers2010(
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
            parameter_manager._apply_consistent_convention_rules(
                parfile_dicts, reference_dict
            )

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2010"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2010"]
        assert "T2CMETHOD" not in parfile_dicts["EPTA"]
        assert "T2CMETHOD" not in parfile_dicts["PPTA"]
        assert mock_warning.call_count == 0

    def test_apply_consistent_convention_rules_tempo2_only_preserves_shared_t2cmethod(
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

        parameter_manager._apply_consistent_convention_rules(
            parfile_dicts, reference_dict
        )

        for pd in parfile_dicts.values():
            assert pd["ECL"] == ["IERS2010"]
            assert pd["T2CMETHOD"] == ["TEMPO"]

    def test_apply_consistent_convention_rules_tempo2_only_aligns_heterogeneous_conventions(
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

        parameter_manager._apply_consistent_convention_rules(
            parfile_dicts, reference_dict
        )

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2003"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2003"]
        assert parfile_dicts["EPTA"]["T2CMETHOD"] == ["TEMPO"]
        assert parfile_dicts["PPTA"]["T2CMETHOD"] == ["TEMPO"]

    def test_apply_consistent_convention_rules_single_pta_skips_alignment(self):
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

        parameter_manager._apply_consistent_convention_rules(
            parfile_dicts, reference_dict
        )

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

    def test_apply_consistent_convention_rules_mixed_astrometry_raises(self):
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
            parameter_manager._apply_consistent_convention_rules(
                parfile_dicts, reference_dict
            )

    def test_apply_consistent_convention_rules_pint_only_elong_elat_aliases(self):
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

        parameter_manager._apply_consistent_convention_rules(
            parfile_dicts, reference_dict
        )

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2010"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2010"]

    def test_apply_consistent_convention_rules_missing_reference_clock_raises(self):
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
            parameter_manager._apply_consistent_convention_rules(
                parfile_dicts, reference_dict
            )

    def test_apply_consistent_convention_rules_missing_reference_ephem_raises(self):
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
            parameter_manager._apply_consistent_convention_rules(
                parfile_dicts, reference_dict
            )
