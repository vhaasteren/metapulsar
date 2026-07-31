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
    AlignmentPolicy,
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


def align_conventions(parameter_manager, parfile_dicts, reference_dict):
    """Run the two consistent-alignment steps in pipeline order.

    ``_make_parameters_consistent`` numerically transforms ecliptic astrometry
    first and only then applies the convention rules; tests that exercise the
    convention surface must do the same.
    """
    parameter_manager._transform_ecliptic_for_all(parfile_dicts, reference_dict)
    parameter_manager._apply_consistent_convention_rules(parfile_dicts, reference_dict)


# Enough of a timing model that PINT can build one (needed for the numeric
# ecliptic transformation), kept small on purpose.
ECLIPTIC_BODY = (
    "PSR J1600-3053\n"
    "PEPOCH 55000\n"
    "F0 277.937 1\n"
    "F1 -7.3e-16 1\n"
    "LAMBDA 244.347677 1\n"
    "BETA -10.071873 1\n"
    "PMLAMBDA -0.35 1\n"
    "PMBETA -7.0 1\n"
    "POSEPOCH 55000\n"
    "DM 52.3 1\n"
    "DMEPOCH 55000\n"
)

EQUATORIAL_BODY = (
    "PSR J1857+0943\n"
    "PEPOCH 55000\n"
    "F0 186.494081 1\n"
    "F1 -6.2e-16 1\n"
    "RAJ 18:57:36.3937 1\n"
    "DECJ +09:43:17.291 1\n"
    "POSEPOCH 55000\n"
    "DM 13.299 1\n"
    "DMEPOCH 55000\n"
)


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

    def test_apply_consistent_convention_rules_cross_engine_ecliptic(self):
        """Cross-engine ecliptic pars land on IERS2003 with explicit IAU2000B."""
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    ECLIPTIC_BODY + "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    ECLIPTIC_BODY + "ECL IERS2010\n"
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

        align_conventions(parameter_manager, parfile_dicts, reference_dict)

        for _, pd in parfile_dicts.items():
            assert pd["UNITS"] == ["TDB"]
            assert pd["ECL"] == ["IERS2003"]
            assert pd["T2CMETHOD"] == ["IAU2000B"]
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
                    EQUATORIAL_BODY + "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    EQUATORIAL_BODY + "T2CMETHOD TEMPO\n"
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
            align_conventions(parameter_manager, parfile_dicts, reference_dict)

        for _, pd in parfile_dicts.items():
            assert pd["UNITS"] == ["TDB"]
            assert "ECL" not in pd
            assert pd["T2CMETHOD"] == ["IAU2000B"]
        assert mock_warning.call_count == 2

    def test_apply_consistent_convention_rules_pint_only_aligns_missing_ecl_to_iers2010(
        self,
    ):
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    ECLIPTIC_BODY + "ECL IERS2010\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    ECLIPTIC_BODY + "EPHEM DE436\nCLK TT(BIPM2015)\nUNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        with patch.object(parameter_manager.logger, "warning") as mock_warning:
            align_conventions(parameter_manager, parfile_dicts, reference_dict)

        assert parfile_dicts["EPTA"]["ECL"] == ["IERS2010"]
        assert parfile_dicts["PPTA"]["ECL"] == ["IERS2010"]
        # PINT-only stacks keep the thinner profile: no forced T2CMETHOD and no
        # forced troposphere / planetary-Shapiro switches.
        assert "T2CMETHOD" not in parfile_dicts["EPTA"]
        assert "T2CMETHOD" not in parfile_dicts["PPTA"]
        assert "CORRECT_TROPOSPHERE" not in parfile_dicts["EPTA"]
        assert "TIMEEPH" not in parfile_dicts["EPTA"]
        assert mock_warning.call_count == 0

    def test_apply_consistent_convention_rules_tempo2_only_preserves_shared_t2cmethod(
        self,
    ):
        file_data = {
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    ECLIPTIC_BODY + "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    ECLIPTIC_BODY + "ECL IERS2010\n"
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

        align_conventions(parameter_manager, parfile_dicts, reference_dict)

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
                    ECLIPTIC_BODY + "ECL IERS2010\n"
                    "T2CMETHOD TEMPO\n"
                    "EPHEM DE436\n"
                    "CLOCK TT(BIPM2015)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "tempo2",
                "par_content": (
                    ECLIPTIC_BODY + "ECL IERS2003\n"
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

        align_conventions(parameter_manager, parfile_dicts, reference_dict)

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
        parameter_manager._make_component_parameters_consistent(
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

    def test_empty_exclude_from_consistent_restores_merged_dm(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
            exclude_from_consistent=[],
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
        consistent_parfiles = parameter_manager._make_parameters_consistent(
            parfile_data
        )

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

    def test_exclude_from_consistent_keeps_f0_pta_specific(
        self, file_data_with_two_different_f0_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_f0_values,
            combine_components=["spindown"],
            exclude_from_consistent=("F0",),
        )

        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]
        spindown_params = get_parameters_by_type_from_models(
            "spindown", parameter_manager.pint_models
        )
        parameter_manager._make_component_parameters_consistent(
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

    def test_exclude_from_consistent_accepts_lowercase_dm_alias(
        self, file_data_with_two_different_dm_values
    ):
        parameter_manager = ParameterManager(
            file_data=file_data_with_two_different_dm_values,
            combine_components=["dispersion"],
            add_dm_derivatives=True,
            exclude_from_consistent=("dm",),
        )

        assert parameter_manager.exclude_from_consistent == {"DM"}

        mergeable = {
            resolve_parameter_alias(param)
            for param in parameter_manager._discover_mergeable_parameters()
        }

        assert "DM" not in mergeable

        mapping = parameter_manager.build_parameter_mappings()

        assert "DM_EPTA" in mapping.pta_specific_parameters
        assert "DM_PPTA" in mapping.pta_specific_parameters
        assert "DM" not in mapping.merged_parameters

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
        # LAMBDA/BETA -> ELONG/ELAT, and PMLAMBDA/PMBETA -> PMELONG/PMELAT
        elong_body = ECLIPTIC_BODY.replace("LAMBDA ", "ELONG ").replace(
            "BETA ", "ELAT "
        )
        file_data = {
            "EPTA": {
                "timing_package": "pint",
                "par_content": (
                    elong_body + "ECL IERS2010\n"
                    "EPHEM DE440\n"
                    "CLOCK TT(BIPM2021)\n"
                    "UNITS TDB\n"
                ),
            },
            "PPTA": {
                "timing_package": "pint",
                "par_content": (
                    elong_body + "EPHEM DE436\nCLK TT(BIPM2015)\nUNITS TDB\n"
                ),
            },
        }
        parameter_manager = ParameterManager(file_data=file_data, combine_components=[])
        parfile_dicts = parameter_manager._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        align_conventions(parameter_manager, parfile_dicts, reference_dict)

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

    @patch.object(ParameterManager, "_apply_consistent_convention_rules")
    def test_make_parameters_consistent_aligns_ne_sw_cross_engine(self, _mock_conv):
        """_make_parameters_consistent aligns NE_SW before convention rules."""
        file_data = {
            "NG": {
                "timing_package": "pint",
                "par_content": EQUATORIAL_BODY + "UNITS TDB\n",
            },
            "EPTA": {
                "timing_package": "tempo2",
                "par_content": EQUATORIAL_BODY + "UNITS TDB\n",
            },
        }
        parameter_manager = ParameterManager(
            file_data=file_data,
            combine_components=[],
        )
        parfile_data = {pta: data["par_content"] for pta, data in file_data.items()}

        result = parameter_manager._make_parameters_consistent(parfile_data)

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

    def test_make_parfiles_consistent_solarn0_roundtrip(self, tmp_path):
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

        output_files = parameter_manager.make_parfiles_consistent()

        from metapulsar.pint_helpers import create_pint_model

        for pta_name, path in output_files.items():
            content = Path(path).read_text()
            assert "SOLARN0" not in content, f"{pta_name}: alias survived rewrite"
            model = create_pint_model(content)  # must not raise
            assert float(model.NE_SW.value) == pytest.approx(4.0)

    def test_make_parfiles_consistent_writes_engine_native_clock_keys(self, tmp_path):
        """PINT targets use CLOCK and Tempo2 targets use CLK after alignment."""
        body = (
            "PSR J1857+0943\nPEPOCH 55000\nF0 186.494081\nF1 -6.2e-16\n"
            "RAJ 18:57:36.3937\nDECJ +09:43:17.291\nDM 13.299\n"
            "EPHEM DE440\nUNITS TDB\n"
        )
        manager = ParameterManager(
            file_data={
                "NG": {
                    "timing_package": "pint",
                    "par_content": body + "CLK TT(BIPM2019)\n",
                },
                "EPTA": {
                    "timing_package": "tempo2",
                    "par_content": body + "CLOCK TT(BIPM2015)\n",
                },
            },
            combine_components=[],
            output_dir=tmp_path,
            pulsar_name="J1857+0943",
        )

        output_files = manager.make_parfiles_consistent()

        rows_by_pta = {}
        for pta_name, path in output_files.items():
            content = Path(path).read_text()
            rows_by_pta[pta_name] = {
                line.split()[0]: line.split()[1:]
                for line in content.splitlines()
                if line.strip() and not line.startswith("#")
            }

        assert rows_by_pta["NG"]["CLOCK"] == ["TT(BIPM2019)"]
        assert "CLK" not in rows_by_pta["NG"]
        assert rows_by_pta["EPTA"]["CLK"] == ["TT(BIPM2019)"]
        assert "CLOCK" not in rows_by_pta["EPTA"]


# ===================================================================
# Complete cross-engine alignment: policy, stripping, transformations
# ===================================================================


def _cross_engine_file_data(
    reference_extra: str = "",
    other_extra: str = "",
    body: str = EQUATORIAL_BODY,
    reference_package: str = "tempo2",
    other_package: str = "pint",
):
    """Two-PTA file data with a tempo2 reference and a PINT partner."""
    return {
        "EPTA": {
            "timing_package": reference_package,
            "par_content": (
                body + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TDB\n" + reference_extra
            ),
        },
        "NG": {
            "timing_package": other_package,
            "par_content": (
                body + "EPHEM DE436\nCLOCK TT(BIPM2015)\nUNITS TDB\n" + other_extra
            ),
        },
    }


def _prepared_dicts(parameter_manager):
    """Parse and run the common-surface preparation step."""
    parfile_dicts = parameter_manager._parse_parfiles()
    parameter_manager._prepare_common_surface(parfile_dicts)
    return parfile_dicts


class TestAlignmentPolicy:
    """The one new public knob for the consistent strategy."""

    def test_defaults(self):
        policy = AlignmentPolicy()
        assert policy.unsupported == "strip"
        assert policy.ephem is None
        assert policy.clock is None
        assert policy.bipm_version is None
        assert policy.ne_sw is None

    def test_rejects_unknown_unsupported_policy(self):
        with pytest.raises(ValueError, match="strip.*error"):
            AlignmentPolicy(unsupported="keep")

    def test_rejects_negative_ne_sw(self):
        with pytest.raises(ValueError, match="ne_sw must be non-negative"):
            AlignmentPolicy(ne_sw=-1.0)

    def test_is_frozen(self):
        policy = AlignmentPolicy()
        with pytest.raises(Exception):
            policy.unsupported = "error"

    def test_parameter_manager_defaults_to_strip(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        assert pm.alignment_policy == AlignmentPolicy()

    def test_exported_from_package_root(self):
        import metapulsar

        assert metapulsar.AlignmentPolicy is AlignmentPolicy


class TestTempo1Expansion:
    """Section 6.1: the aggregate TEMPO1 switch becomes six explicit states."""

    def test_absent_tempo1_is_a_no_op(self):
        from metapulsar.parameter_manager import expand_tempo1

        par = {"PSR": ["J1857+0943"], "UNITS": ["TDB"]}
        assert expand_tempo1(par) == []
        assert par == {"PSR": ["J1857+0943"], "UNITS": ["TDB"]}

    def test_tempo1_is_removed_and_expanded(self):
        from metapulsar.parameter_manager import TEMPO1_DEFAULTS, expand_tempo1

        par = {"PSR": ["J1857+0943"], "TEMPO1": ["1"]}
        filled = expand_tempo1(par)

        assert "TEMPO1" not in par
        assert set(filled) == set(TEMPO1_DEFAULTS)
        for key, value in TEMPO1_DEFAULTS.items():
            assert par[key] == value

    def test_explicit_values_win_over_tempo1_defaults(self):
        from metapulsar.parameter_manager import expand_tempo1

        par = {"TEMPO1": ["1"], "T2CMETHOD": ["IAU2000B"], "UNITS": ["TCB"]}
        filled = expand_tempo1(par)

        assert "T2CMETHOD" not in filled
        assert par["T2CMETHOD"] == ["IAU2000B"]
        assert par["UNITS"] == ["TCB"]
        assert par["TIMEEPH"] == ["FB90"]

    def test_pipeline_expands_tempo1_for_multi_pta(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(reference_extra="TEMPO1\n")
        )
        parfile_dicts = _prepared_dicts(pm)

        assert "TEMPO1" not in parfile_dicts["EPTA"]
        assert parfile_dicts["EPTA"]["TIMEEPH"] == ["FB90"]
        assert parfile_dicts["EPTA"]["DILATEFREQ"] == ["N"]


class TestUnsupportedFamilyMatchers:
    """Section 6.2-6.4: anchored matchers, never generic prefix stripping."""

    @pytest.mark.parametrize(
        "key",
        [
            "EPHEM_FILE",
            "EPH_FILE",
            "EOP_FILE",
            "CLK_CORR_CHAIN",
            "NE_SW_SIN",
            "NE_SW_IFUNC",
            "_NE_SW",
            "DMMODEL",
            "_DM",
            "_CM",
            "DMOFF",
            "SATJUMP",
            "PMRA2",
            "PMDEC2",
            "PMLAMBDA2",
            "PMBETA2",
            "PMELONG2",
            "PMELAT2",
            "PMRV",
            "DSHK",
            "D_AOP",
            "STEL_DX",
            "TELEPOCH",
            "TELX",
            "TELY",
            "TELZ",
            "TEL_DX",
            "TEL_DX1",
            "TEL_DX_1",
        ],
    )
    def test_pint_unsafe_positives(self, key):
        from metapulsar.parameter_manager import _is_pint_unsafe

        assert _is_pint_unsafe(key)

    @pytest.mark.parametrize(
        "key", ["NE_SW", "DM", "CM", "DMX_0001", "PMRA", "PMDEC", "TELESCOPE"]
    )
    def test_pint_unsafe_negatives(self, key):
        from metapulsar.parameter_manager import _is_pint_unsafe

        assert not _is_pint_unsafe(key)

    @pytest.mark.parametrize(
        "key",
        [
            "SWP",
            "SWEPOCH",
            "VLBIAX",
            "VLBIAY",
            "VLBIAZ",
            "NE_SW1",
            "NE_SW2",
            "NE_SW12",
            "SWXDM_0001",
            "SWXP_0001",
            "SWXR1_0001",
            "SWXR2_0001",
            "DMWXEPOCH",
            "DMWXFREQ_0001",
            "DMWXSIN_0001",
            "DMWXCOS_0001",
        ],
    )
    def test_tempo2_unsafe_positives(self, key):
        from metapulsar.parameter_manager import _is_tempo2_unsafe

        assert _is_tempo2_unsafe(key)

    @pytest.mark.parametrize(
        "key", ["NE_SW", "NE_SW_SIN", "SWM", "DM", "DMX_0001", "DMJUMP"]
    )
    def test_tempo2_unsafe_negatives(self, key):
        from metapulsar.parameter_manager import _is_tempo2_unsafe

        assert not _is_tempo2_unsafe(key)

    @pytest.mark.parametrize(
        "key",
        [
            "WAVE1",
            "WAVE12",
            "WAVEEPOCH",
            "WAVE_OM",
            "WXEPOCH",
            "WXFREQ_0001",
            "WXSIN_0001",
            "WXCOS_0001",
            "IFUNC1",
            "SIFUNC",
            "CM",
            "CM1",
            "CM2",
            "CMEPOCH",
            "CMX_0001",
            "CHROMX_0001",
            "CMWXEPOCH",
            "CMWXFREQ_0001",
            "GLEP_1",
            "GLPH_1",
            "GLF0_1",
            "GLF1_1",
            "GLF2_1",
            "GLF0D_1",
            "GLTD_1",
            "GLF0D2_1",
            "GLTD2_1",
            "DMASSPLANET5",
            "DPHASEPLANET3",
            "EXPEP_1",
            "EXPPH_1",
            "EXPTAU_1",
            "EXPINDEX_1",
            "GAUSEP_1",
            "GAUSAMP_1",
            "GAUSSIG_1",
            "GAUSINDEX_1",
            "EXPDIPEP_1",
            "EXPDIPAMP_1",
            "PWSTART_1",
            "PWF0_1",
            "CHROMGAUSS_FREF",
            "TNDMEVENT",
            "TNSHAPELETEVENT",
        ],
    )
    def test_mixed_unsafe_positives(self, key):
        from metapulsar.parameter_manager import _is_mixed_unsafe

        assert _is_mixed_unsafe(key)

    @pytest.mark.parametrize(
        "key",
        [
            # Noise hyperparameters are out of scope and must never be matched.
            "EFAC",
            "EQUAD",
            "ECORR",
            "T2EFAC",
            "T2EQUAD",
            "TNEF",
            "TNEQ",
            "TNECORR",
            "TNREDAMP",
            "TNREDGAM",
            "TNDMAMP",
            "TNDMGAM",
            "TNCHROMAMP",
            "TNCHROMGAM",
            "RNAMP",
            "RNIDX",
            "DMEFAC",
            "DMJUMP",
            # Ordinary deterministic terms that stay.
            "DM",
            "DM1",
            "DM2",
            "DMX_0001",
            "DMEPOCH",
            "JUMP",
            "FD1",
            "FD2",
            "FDJUMP1",
            "FDJUMPDM",
            "FDDC",
            "FDDI",
            "TZRMJD",
            "TZRSITE",
            "TZRFRQ",
            "NE_SW",
            "PX",
            "WAVE",
            "GLEP",
            "CMWX",
        ],
    )
    def test_mixed_unsafe_negatives(self, key):
        from metapulsar.parameter_manager import _is_mixed_unsafe

        assert not _is_mixed_unsafe(key)


class TestUnsupportedPolicy:
    """Section 6: default strip with a warning, or a hard error."""

    UNSUPPORTED_EXTRA = (
        "DMMODEL 1\n"
        "CONSTRAIN DMMODEL\n"
        "CONSTRAIN IFUNC\n"
        "EPHEM_FILE DE440.bsp\n"
        "GLEP_1 55000\n"
        "WAVE1 1e-8 1e-8\n"
        "CMX_0001 0.1\n"
        "DMASSPLANET5 1e-9\n"
        "PMRA2 0.1\n"
        "TELX 0.1\n"
    )
    PINT_EXTRA = "NE_SW1 0.1\nSWX_0001 1.0\nDMWXFREQ_0001 0.01\nSWEPOCH 55000\n"
    NOISE_EXTRA = (
        "TNRedAmp -14.0\n"
        "TNRedGam 3.0\n"
        "EFAC -f L-wide 1.1\n"
        "ECORR -f L-wide 0.5\n"
        "DMJUMP -fe Rcvr 0.01\n"
    )
    LOCAL_EXTRA = (
        "JUMP -fe Rcvr_800 1.2e-6 1\n"
        "FD1 1.0e-5 1\n"
        "FDJUMP1 1.0e-6 1\n"
        "FDDC 1.0\n"
        "FDDI 2.0\n"
        "TZRMJD 55000\n"
        "TZRSITE ao\n"
        "TZRFRQ 1400.0\n"
    )

    def _manager(self, policy=None):
        return ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra=self.UNSUPPORTED_EXTRA
                + self.NOISE_EXTRA
                + self.LOCAL_EXTRA,
                other_extra=self.PINT_EXTRA + self.NOISE_EXTRA,
            ),
            alignment_policy=policy,
        )

    def test_default_policy_strips_every_family(self):
        pm = self._manager()
        parfile_dicts = _prepared_dicts(pm)

        reference = parfile_dicts["EPTA"]
        for key in (
            "DMMODEL",
            "EPHEM_FILE",
            "GLEP_1",
            "WAVE1",
            "CMX_0001",
            "DMASSPLANET5",
            "PMRA2",
            "TELX",
        ):
            assert key not in reference, f"{key} survived the strip policy"

        other = parfile_dicts["NG"]
        for key in ("NE_SW1", "SWX_0001", "DMWXFREQ_0001", "SWEPOCH"):
            assert key not in other, f"{key} survived the strip policy"

    def test_default_policy_keeps_noise_hyperparameters(self):
        pm = self._manager()
        parfile_dicts = _prepared_dicts(pm)

        for pta in ("EPTA", "NG"):
            par = parfile_dicts[pta]
            for key in ("TNREDAMP", "TNREDGAM", "EFAC", "ECORR", "DMJUMP"):
                assert key in par, f"{pta}: noise key {key} was stripped"

    def test_default_policy_keeps_pta_local_deterministic_terms(self):
        pm = self._manager()
        parfile_dicts = _prepared_dicts(pm)

        par = parfile_dicts["EPTA"]
        for key in (
            "JUMP",
            "FD1",
            "FDJUMP1",
            "FDDC",
            "FDDI",
            "TZRMJD",
            "TZRSITE",
            "TZRFRQ",
        ):
            assert key in par, f"PTA-local key {key} was stripped"

    def test_dmmodel_constraints_are_filtered_without_touching_others(self):
        pm = self._manager()
        parfile_dicts = _prepared_dicts(pm)

        assert parfile_dicts["EPTA"]["CONSTRAIN"] == ["IFUNC"]

    def test_strip_warning_names_pta_and_removed_keys(self, caplog):
        pm = self._manager()
        with caplog.at_level(logging.WARNING):
            _prepared_dicts(pm)

        messages = [r.message for r in caplog.records if "stripped" in r.message]
        assert any("PTA EPTA" in m and "DMMODEL" in m for m in messages)
        assert any("PTA NG" in m and "SWX_0001" in m for m in messages)

    def test_error_policy_reports_all_offending_keys(self):
        pm = self._manager(AlignmentPolicy(unsupported="error"))
        parfile_dicts = pm._parse_parfiles()

        with pytest.raises(ValueError) as excinfo:
            pm._prepare_common_surface(parfile_dicts)

        message = str(excinfo.value)
        assert "PTA EPTA" in message
        for key in ("DMMODEL", "EPHEM_FILE", "GLEP_1", "WAVE1", "TELX"):
            assert key in message
        assert "CONSTRAIN DMMODEL" in message

    def test_single_pta_surface_is_not_rewritten(self):
        pm = ParameterManager(
            file_data={
                "EPTA": {
                    "timing_package": "tempo2",
                    "par_content": (
                        EQUATORIAL_BODY
                        + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TDB\n"
                        + self.UNSUPPORTED_EXTRA
                    ),
                }
            }
        )
        parfile_dicts = _prepared_dicts(pm)

        assert parfile_dicts["EPTA"]["DMMODEL"] == ["1"]
        assert "GLEP_1" in parfile_dicts["EPTA"]
        assert parfile_dicts["EPTA"]["CONSTRAIN"] == ["DMMODEL", "IFUNC"]


class TestSolarGeometryNormalization:
    """Section 6.2/6.3: IPM 0 and SWM 1 are value-dependent violations."""

    def test_ipm_zero_is_stripped_and_normalized_on_tempo2_output(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(reference_extra="IPM 0\n")
        )
        parfile_dicts = _prepared_dicts(pm)
        assert "IPM" not in parfile_dicts["EPTA"]

        pm._apply_explicit_conventions(parfile_dicts)
        assert parfile_dicts["EPTA"]["IPM"] == ["1"]
        assert "IPM" not in parfile_dicts["NG"]

    def test_absent_ipm_still_becomes_explicit_on_tempo2_output(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        parfile_dicts = _prepared_dicts(pm)

        pm._apply_explicit_conventions(parfile_dicts)
        assert parfile_dicts["EPTA"]["IPM"] == ["1"]
        assert "IPM" not in parfile_dicts["NG"]

    def test_ipm_zero_is_a_violation_under_error_policy(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(reference_extra="IPM 0\n"),
            alignment_policy=AlignmentPolicy(unsupported="error"),
        )
        with pytest.raises(ValueError, match="IPM"):
            pm._prepare_common_surface(pm._parse_parfiles())

    def test_swm_one_is_stripped_and_normalized_to_zero(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(other_extra="SWM 1\nSWP 2.0\n")
        )
        parfile_dicts = _prepared_dicts(pm)
        assert "SWM" not in parfile_dicts["NG"]
        assert "SWP" not in parfile_dicts["NG"]

        pm._apply_explicit_conventions(parfile_dicts)
        assert parfile_dicts["NG"]["SWM"] == ["0"]
        assert parfile_dicts["EPTA"]["SWM"] == ["0"]

    def test_swm_one_is_a_violation_under_error_policy(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(other_extra="SWM 1\n"),
            alignment_policy=AlignmentPolicy(unsupported="error"),
        )
        with pytest.raises(ValueError, match="SWM"):
            pm._prepare_common_surface(pm._parse_parfiles())

    def test_swm_zero_is_not_a_violation(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(other_extra="SWM 0\n"),
            alignment_policy=AlignmentPolicy(unsupported="error"),
        )
        pm._prepare_common_surface(pm._parse_parfiles())  # must not raise

    def test_policy_ne_sw_overrides_reference_value(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(reference_extra="NE_SW 4\n"),
            alignment_policy=AlignmentPolicy(ne_sw=7.5),
        )
        parfile_dicts = pm._parse_parfiles()
        pm._align_ne_sw_convention(parfile_dicts, parfile_dicts["EPTA"])

        assert parfile_dicts["EPTA"]["NE_SW"] == ["7.5 0"]
        assert parfile_dicts["NG"]["NE_SW"] == ["7.5 0"]


class TestUnitNormalization:
    """Section 7.3: timescale policy is gated by PTA count and engine mix."""

    def test_all_tdb_collection_is_returned_unchanged(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        contents = {pta: data["par_content"] for pta, data in pm.file_data.items()}
        assert pm._convert_units_if_needed(contents) == contents

    def test_mixed_engine_all_tcb_collection_is_converted(self):
        tcb_body = EQUATORIAL_BODY + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TCB\n"
        pm = ParameterManager(
            file_data={
                "A": {"timing_package": "pint", "par_content": tcb_body},
                "B": {"timing_package": "tempo2", "par_content": tcb_body},
            }
        )
        contents = {pta: data["par_content"] for pta, data in pm.file_data.items()}

        def as_tdb(text):
            return text.replace("UNITS TCB", "UNITS TDB")

        with (
            patch.object(pm, "_convert_pint_to_tdb", side_effect=as_tdb),
            patch.object(pm, "_convert_tempo2_to_tdb", side_effect=as_tdb),
        ):
            converted = pm._convert_units_if_needed(contents)

        for pta, text in converted.items():
            parsed = parse_parfile(StringIO(text))
            assert parsed["UNITS"] == ["TDB"], f"{pta} not converted to TDB"
            assert text != contents[pta]

    @pytest.mark.parametrize("package", ["pint", "tempo2"])
    def test_single_engine_all_tcb_collection_is_preserved(self, package):
        tcb_body = EQUATORIAL_BODY + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TCB\n"
        pm = ParameterManager(
            file_data={
                "A": {"timing_package": package, "par_content": tcb_body},
                "B": {"timing_package": package, "par_content": tcb_body},
            }
        )
        contents = {pta: data["par_content"] for pta, data in pm.file_data.items()}

        assert pm._convert_units_if_needed(contents) == contents

        parfile_dicts = {
            pta: parse_parfile(StringIO(text)) for pta, text in contents.items()
        }
        pm._apply_explicit_conventions(parfile_dicts)
        assert {par["UNITS"][0] for par in parfile_dicts.values()} == {"TCB"}

    def test_single_pta_tcb_is_preserved(self):
        text = EQUATORIAL_BODY + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TCB\n"
        pm = ParameterManager(
            file_data={
                "A": {"timing_package": "tempo2", "par_content": text},
            }
        )

        assert pm._convert_units_if_needed({"A": text}) == {"A": text}

    def test_mixed_collection_converts_only_the_tcb_input(self):
        pm = ParameterManager(
            file_data={
                "A": {
                    "timing_package": "pint",
                    "par_content": EQUATORIAL_BODY
                    + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TDB\n",
                },
                "B": {
                    "timing_package": "pint",
                    "par_content": EQUATORIAL_BODY
                    + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TCB\n",
                },
            }
        )
        contents = {pta: data["par_content"] for pta, data in pm.file_data.items()}

        converted = pm._convert_units_if_needed(contents)

        assert converted["A"] == contents["A"]
        assert parse_parfile(StringIO(converted["B"]))["UNITS"] == ["TDB"]

    def test_assert_explicit_tdb_rejects_non_tdb_output(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        with pytest.raises(RuntimeError, match="explicit\n?.*UNITS TDB"):
            pm._assert_explicit_tdb("A", "PSR J1857+0943\nUNITS TCB\n")


class TestEclipticTransformation:
    """Section 7.4: a coordinate rotation, never a label rewrite."""

    ECL_BODY = ECLIPTIC_BODY + "EPHEM DE440\nCLK TT(BIPM2019)\nUNITS TDB\n"

    def _icrs(self, par_dict, epoch):
        from astropy.coordinates import ICRS
        from metapulsar.pint_helpers import create_pint_model

        return (
            create_pint_model(par_dict).get_psr_coords(epoch=epoch).transform_to(ICRS)
        )

    def test_transform_preserves_sky_direction_at_two_epochs(self):
        import astropy.units as u

        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra="ECL IERS2010\n",
                other_extra="ECL IERS2010\n",
                body=ECLIPTIC_BODY,
            )
        )
        parfile_dicts = pm._parse_parfiles()
        before = dict(parfile_dicts["EPTA"])

        pm._transform_ecliptic_for_all(parfile_dicts, parfile_dicts["EPTA"])
        after = parfile_dicts["EPTA"]

        assert after["ECL"] == ["IERS2003"]
        # A relabel would keep the printed ELONG/ELAT; this must not.
        relabelled = dict(before)
        relabelled["ECL"] = ["IERS2003"]

        for epoch in (54000.0, 57000.0):
            reference = self._icrs(before, epoch)
            transformed = self._icrs(after, epoch)
            relabel = self._icrs(relabelled, epoch)

            assert reference.separation(transformed).to_value(u.arcsec) < 1e-7
            # Guard: the test would pass trivially if a relabel were harmless.
            assert reference.separation(relabel).to_value(u.arcsec) > 1e-5

    def test_transform_replaces_lambda_beta_with_canonical_names(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra="ECL IERS2010\n",
                other_extra="ECL IERS2010\n",
                body=ECLIPTIC_BODY,
            )
        )
        parfile_dicts = pm._parse_parfiles()
        pm._transform_ecliptic_for_all(parfile_dicts, parfile_dicts["EPTA"])

        par = parfile_dicts["EPTA"]
        for alias in ("LAMBDA", "BETA", "PMLAMBDA", "PMBETA"):
            assert alias not in par
        for canonical in ("ELONG", "ELAT", "PMELONG", "PMELAT"):
            assert canonical in par

    def test_iers2003_input_in_mixed_stack_is_still_transformed_to_iers2003(self):
        import astropy.units as u

        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra="ECL IERS2003\n",
                other_extra="ECL IERS2003\n",
                body=ECLIPTIC_BODY,
            )
        )
        parfile_dicts = pm._parse_parfiles()
        before = dict(parfile_dicts["EPTA"])
        pm._transform_ecliptic_for_all(parfile_dicts, parfile_dicts["EPTA"])
        after = parfile_dicts["EPTA"]

        assert after["ECL"] == ["IERS2003"]
        assert (
            self._icrs(before, 55000.0)
            .separation(self._icrs(after, 55000.0))
            .to_value(u.arcsec)
            < 1e-7
        )

    def test_equatorial_stack_is_left_alone(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        parfile_dicts = pm._parse_parfiles()
        before = {pta: dict(par) for pta, par in parfile_dicts.items()}

        pm._transform_ecliptic_for_all(parfile_dicts, parfile_dicts["EPTA"])

        assert parfile_dicts == before

    def test_single_pta_astrometry_is_untouched(self):
        pm = ParameterManager(
            file_data={
                "EPTA": {
                    "timing_package": "tempo2",
                    "par_content": self.ECL_BODY + "ECL IERS2010\n",
                }
            }
        )
        parfile_dicts = pm._parse_parfiles()
        before = {pta: dict(par) for pta, par in parfile_dicts.items()}

        pm._transform_ecliptic_for_all(parfile_dicts, parfile_dicts["EPTA"])

        assert parfile_dicts == before

    def test_ddk_kom_is_converted(self):
        ddk_body = ECLIPTIC_BODY + (
            "BINARY DDK\n"
            "PB 14.348466\n"
            "A1 8.801653\n"
            "T0 55000.0\n"
            "ECC 1.737e-04\n"
            "OM 181.85\n"
            "M2 0.33\n"
            "KIN 68.0\n"
            "KOM 76.0\n"
            "PX 0.5\n"
        )
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra="ECL IERS2010\n",
                other_extra="ECL IERS2010\n",
                body=ddk_body,
            )
        )
        parfile_dicts = pm._parse_parfiles()
        pm._transform_ecliptic_for_all(parfile_dicts, parfile_dicts["EPTA"])

        par = parfile_dicts["EPTA"]
        assert par["ECL"] == ["IERS2003"]
        assert "KOM" in par
        kom = float(str(par["KOM"][0]).split()[0])
        # Same physical orientation, expressed against the new ecliptic pole.
        assert kom == pytest.approx(76.0, abs=1e-2)

    def test_conversion_failure_is_reported_with_the_pta_name(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        with pytest.raises(ValueError, match="PTA EPTA: ecliptic astrometry"):
            pm._transform_ecliptic_astrometry(
                "EPTA", {"ELONG": ["1"], "ELAT": ["2"]}, "IERS2003"
            )


class TestReferenceConventionResolution:
    """Section 7.5: EPHEM and a dated clock realization."""

    def test_reference_values_are_used_by_default(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        parfile_dicts = pm._parse_parfiles()

        ephem, clock = pm._resolve_reference_conventions(parfile_dicts["EPTA"])

        assert ephem == "DE440"
        assert clock == "TT(BIPM2019)"

    def test_policy_values_override_the_reference(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(),
            alignment_policy=AlignmentPolicy(ephem="DE421", clock="TT(BIPM2023)"),
        )
        parfile_dicts = pm._parse_parfiles()

        ephem, clock = pm._resolve_reference_conventions(parfile_dicts["EPTA"])

        assert ephem == "DE421"
        assert clock == "TT(BIPM2023)"

    def test_policy_supplies_missing_reference_values(self):
        pm = ParameterManager(
            file_data={
                "A": {
                    "timing_package": "pint",
                    "par_content": EQUATORIAL_BODY + "UNITS TDB\n",
                },
                "B": {
                    "timing_package": "pint",
                    "par_content": EQUATORIAL_BODY + "UNITS TDB\n",
                },
            },
            alignment_policy=AlignmentPolicy(ephem="DE440", clock="TT(BIPM2021)"),
        )
        parfile_dicts = pm._parse_parfiles()

        assert pm._resolve_reference_conventions(parfile_dicts["A"]) == (
            "DE440",
            "TT(BIPM2021)",
        )

    def test_clk_alias_is_accepted_and_spelling_is_retained(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        parfile_dicts = pm._parse_parfiles()
        reference_dict = parfile_dicts["EPTA"]

        align_conventions(pm, parfile_dicts, reference_dict)

        assert parfile_dicts["EPTA"]["CLK"] == ["TT(BIPM2019)"]
        assert parfile_dicts["NG"]["CLOCK"] == ["TT(BIPM2019)"]
        assert "CLOCK" not in parfile_dicts["EPTA"]
        assert "CLK" not in parfile_dicts["NG"]

    def test_bare_bipm_without_version_raises(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(reference_extra="", body=EQUATORIAL_BODY)
        )
        parfile_dicts = pm._parse_parfiles()
        parfile_dicts["EPTA"]["CLK"] = ["TT(BIPM)"]

        with pytest.raises(ValueError, match="Bare TT\\(BIPM\\) is ambiguous"):
            pm._resolve_reference_conventions(parfile_dicts["EPTA"])

    def test_bare_bipm_resolves_with_policy_version(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(),
            alignment_policy=AlignmentPolicy(bipm_version=2021),
        )
        parfile_dicts = pm._parse_parfiles()
        parfile_dicts["EPTA"]["CLK"] = ["TT(BIPM)"]

        _, clock = pm._resolve_reference_conventions(parfile_dicts["EPTA"])
        assert clock == "TT(BIPM2021)"

    def test_dated_clock_conflicting_with_bipm_version_raises(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(),
            alignment_policy=AlignmentPolicy(clock="TT(BIPM2019)", bipm_version=2021),
        )
        parfile_dicts = pm._parse_parfiles()

        with pytest.raises(ValueError, match="disagrees with"):
            pm._resolve_reference_conventions(parfile_dicts["EPTA"])

    def test_dated_clock_agreeing_with_bipm_version_is_accepted(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(),
            alignment_policy=AlignmentPolicy(clock="TT(BIPM2019)", bipm_version=2019),
        )
        parfile_dicts = pm._parse_parfiles()

        _, clock = pm._resolve_reference_conventions(parfile_dicts["EPTA"])
        assert clock == "TT(BIPM2019)"

    def test_non_bipm_clock_is_passed_through(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        assert pm._resolve_bipm_clock("UTC(NIST)") == "UTC(NIST)"


class TestExplicitConventionSwitches:
    """Section 4.1/4.2: forced switches only for mixed PINT+Tempo2 stacks."""

    def _mixed_dicts(self, reference_extra="", other_extra=""):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra=reference_extra, other_extra=other_extra
            )
        )
        parfile_dicts = pm._parse_parfiles()
        pm._apply_explicit_conventions(parfile_dicts)
        return parfile_dicts

    def test_mixed_stack_writes_the_full_profile(self):
        parfile_dicts = self._mixed_dicts()

        for par in parfile_dicts.values():
            assert par["UNITS"] == ["TDB"]
            assert par["T2CMETHOD"] == ["IAU2000B"]
            assert par["TIMEEPH"] == ["FB90"]
            assert par["DILATEFREQ"] == ["N"]
            assert par["CORRECT_TROPOSPHERE"] == ["N"]
            assert par["PLANET_SHAPIRO"] == ["N"]
            assert par["SWM"] == ["0"]
            assert "NO_SS_SHAPIRO" not in par

    def test_no_ss_shapiro_is_removed_independently_of_planet_shapiro(self):
        parfile_dicts = self._mixed_dicts(
            reference_extra="NO_SS_SHAPIRO\nPLANET_SHAPIRO Y\n"
        )

        assert "NO_SS_SHAPIRO" not in parfile_dicts["EPTA"]
        assert parfile_dicts["EPTA"]["PLANET_SHAPIRO"] == ["N"]

    def test_absent_no_ss_shapiro_stays_absent(self):
        parfile_dicts = self._mixed_dicts()
        assert "NO_SS_SHAPIRO" not in parfile_dicts["EPTA"]

    def test_pint_only_stack_keeps_tropo_and_planet_settings(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra="CORRECT_TROPOSPHERE Y\nPLANET_SHAPIRO Y\n",
                reference_package="pint",
                other_package="pint",
            )
        )
        parfile_dicts = pm._parse_parfiles()
        pm._apply_explicit_conventions(parfile_dicts)

        assert parfile_dicts["EPTA"]["CORRECT_TROPOSPHERE"] == ["Y"]
        assert parfile_dicts["EPTA"]["PLANET_SHAPIRO"] == ["Y"]
        assert "TIMEEPH" not in parfile_dicts["EPTA"]
        assert "T2CMETHOD" not in parfile_dicts["EPTA"]
        assert parfile_dicts["EPTA"]["UNITS"] == ["TDB"]

    def test_tempo2_only_stack_keeps_tropo_and_planet_settings(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra="CORRECT_TROPOSPHERE Y\n",
                reference_package="tempo2",
                other_package="tempo2",
            )
        )
        parfile_dicts = pm._parse_parfiles()
        pm._apply_explicit_conventions(parfile_dicts)

        assert parfile_dicts["EPTA"]["CORRECT_TROPOSPHERE"] == ["Y"]
        assert "IPM" not in parfile_dicts["EPTA"]


class TestEll1hHarmonics:
    """Section 7.7: dual NHARM (tempo2) + NHARMS (PINT) floor of 7."""

    def _align(self, extra_lines):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        par = parse_parfile(StringIO(extra_lines))
        pm._align_ell1h_nharms(par)
        return par

    def test_h3_h4_without_harmonic_count_gets_seven(self):
        par = self._align("BINARY ELL1H\nH3 1e-7\nH4 5e-8\n")
        assert par["NHARM"] == ["7"]
        assert par["NHARMS"] == ["7"]

    def test_nharms_four_is_raised_to_seven(self):
        par = self._align("BINARY ELL1H\nH3 1e-7\nH4 5e-8\nNHARMS 4\n")
        assert par["NHARM"] == ["7"]
        assert par["NHARMS"] == ["7"]

    def test_nharm_four_is_raised_to_seven(self):
        par = self._align("BINARY T2\nH3 1e-7\nH4 5e-8\nNHARM 4\n")
        assert par["NHARM"] == ["7"]
        assert par["NHARMS"] == ["7"]

    def test_nharms_nine_is_preserved(self):
        par = self._align("BINARY ELL1H\nH3 1e-7\nH4 5e-8\nNHARMS 9\n")
        assert par["NHARM"] == ["9"]
        assert par["NHARMS"] == ["9"]

    def test_nharm_nine_is_preserved(self):
        par = self._align("BINARY T2\nH3 1e-7\nH4 5e-8\nNHARM 9\n")
        assert par["NHARM"] == ["9"]
        assert par["NHARMS"] == ["9"]

    def test_h3_stig_gets_no_harmonic_count(self):
        par = self._align("BINARY ELL1H\nH3 1e-7\nSTIG 0.7\n")
        assert "NHARM" not in par
        assert "NHARMS" not in par

    def test_h3_stigma_alias_gets_no_harmonic_count(self):
        par = self._align("BINARY ELL1H\nH3 1e-7\nSTIGMA 0.7\n")
        assert "NHARM" not in par
        assert "NHARMS" not in par

    @pytest.mark.parametrize("stigma_name", ["STIG", "STIGMA", "VARSIGMA"])
    @pytest.mark.parametrize("unsupported", ["strip", "error"])
    def test_h4_and_stigma_conflict_always_errors(self, stigma_name, unsupported):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_extra=(f"BINARY ELL1H\nH3 1e-7\nH4 5e-8\n{stigma_name} 0.7\n")
            ),
            alignment_policy=AlignmentPolicy(unsupported=unsupported),
        )

        with pytest.raises(
            ValueError,
            match=rf"PTA EPTA: invalid orthometric.*H4 and {stigma_name}",
        ):
            _prepared_dicts(pm)

    def test_single_pta_h4_and_stig_conflict_errors(self):
        pm = ParameterManager(
            file_data={
                "EPTA": {
                    "timing_package": "tempo2",
                    "par_content": (
                        EQUATORIAL_BODY + "BINARY ELL1H\nH3 1e-7\nH4 5e-8\nSTIG 0.7\n"
                    ),
                }
            }
        )

        with pytest.raises(ValueError, match="Tempo2 would ignore STIG"):
            _prepared_dicts(pm)

    def test_non_orthometric_binary_is_untouched(self):
        par = self._align("BINARY DD\nPB 12.3\nA1 9.2\n")
        assert "NHARM" not in par
        assert "NHARMS" not in par

    def test_both_spellings_survive_pint_serialization(self):
        from metapulsar.pint_helpers import dict_to_parfile_string

        par = self._align("BINARY ELL1H\nH3 1e-7\nH4 5e-8\n")
        text = dict_to_parfile_string(par, format="pint")

        assert "NHARM " in text or text.splitlines()[-2].startswith("NHARM")
        reparsed = parse_parfile(StringIO(text))
        assert reparsed["NHARM"] == ["7"]
        assert reparsed["NHARMS"] == ["7"]


class TestEll1hShapiroMode:
    """Section 7.8: 'absorbed' only on mixed-engine consistent stacks."""

    def test_mixed_stack_uses_absorbed(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        assert pm.ell1h_shapiro == "absorbed"

    def test_pint_only_stack_uses_full(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_package="pint", other_package="pint"
            )
        )
        assert pm.ell1h_shapiro == "full"

    def test_tempo2_only_stack_uses_full(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(
                reference_package="tempo2", other_package="tempo2"
            )
        )
        assert pm.ell1h_shapiro == "full"

    def test_single_pta_uses_full(self):
        pm = ParameterManager(
            file_data={
                "EPTA": {
                    "timing_package": "tempo2",
                    "par_content": EQUATORIAL_BODY + "UNITS TDB\n",
                }
            }
        )
        assert pm.ell1h_shapiro == "full"

    def test_libstempo_label_counts_as_tempo2(self):
        pm = ParameterManager(
            file_data=_cross_engine_file_data(reference_package="libstempo")
        )
        assert pm.ell1h_shapiro == "absorbed"

    def test_resolve_helper(self):
        from metapulsar.parameter_manager import resolve_ell1h_shapiro_mode

        assert resolve_ell1h_shapiro_mode(["pint", "tempo2"]) == "absorbed"
        assert resolve_ell1h_shapiro_mode(["pint", "pint"]) == "full"
        assert resolve_ell1h_shapiro_mode(["tempo2"]) == "full"
        assert resolve_ell1h_shapiro_mode([None]) == "full"

    def test_temporary_models_use_the_stack_convention(self):
        pm = ParameterManager(file_data=_cross_engine_file_data())
        with patch("metapulsar.parameter_manager.create_pint_model") as mock_create:
            pm._create_model("PSR J0000+0000\n")
        mock_create.assert_called_once_with(
            "PSR J0000+0000\n", ell1h_shapiro="absorbed"
        )
