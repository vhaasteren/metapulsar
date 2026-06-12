"""Unit tests for PINT helper functions."""

import pytest
from unittest.mock import Mock, patch
from metapulsar.pint_helpers import (
    check_component_available_in_model,
    get_parameter_identifiability_from_model,
    get_parameters_by_type_from_parfiles,
    create_pint_model,
    PINTDiscoveryError,
)

# Test constants for better maintainability
EXPECTED_ASTROMETRY_PARAMS = {
    "RAJ",
    "DECJ",
    "PMRA",
    "PMDEC",
    "ELONG",
    "ELAT",
    "PMELONG",
    "PMELAT",
}
EXPECTED_SPINDOWN_PARAMS = {"F0", "F1", "F2", "PEPOCH"}


@pytest.fixture
def mock_astrometry_components():
    """Fixture for astrometry component mocking to avoid duplication."""
    mock_instance = Mock()
    mock_instance.category_component_map = {
        "astrometry": ["AstrometryEquatorial", "AstrometryEcliptic"]
    }

    # Mock equatorial astrometry component
    mock_astrometry_eq = Mock()
    mock_astrometry_eq.return_value.params = ["RAJ", "DECJ", "PMRA", "PMDEC"]

    # Mock ecliptic astrometry component
    mock_astrometry_ec = Mock()
    mock_astrometry_ec.return_value.params = ["ELONG", "ELAT", "PMELONG", "PMELAT"]

    mock_instance.AstrometryEquatorial = mock_astrometry_eq
    mock_instance.AstrometryEcliptic = mock_astrometry_ec

    return mock_instance


# These tests are commented out until the function is implemented as part of the refactor
#
# class TestGetParametersByTypeFromPint:
#     """Test get_parameters_by_type_from_pint function."""
#
#     def test_astrometry_parameters_discovery(self):
#         """Test discovery of astrometry parameters."""
#         result = get_parameters_by_type_from_pint("astrometry")
#
#         # Check that we get the expected astrometry parameters
#         assert EXPECTED_ASTROMETRY_PARAMS.issubset(set(result))
#
#         # Check that we get additional parameters (PINT discovery is working)
#         assert len(result) > len(
#             EXPECTED_ASTROMETRY_PARAMS
#         ), "Expected PINT to return more parameters than just the basic set"
#
#     def test_spindown_parameters_discovery(self):
#         """Test discovery of spindown parameters."""
#         result = get_parameters_by_type_from_pint("spindown")
#
#         # Check that we get the expected spindown parameters
#         assert EXPECTED_SPINDOWN_PARAMS.issubset(set(result))
#
#         # Check that we get additional parameters (PINT discovery is working)
#         assert len(result) > len(
#             EXPECTED_SPINDOWN_PARAMS
#         ), "Expected PINT to return more parameters than just the basic set"
#
#     def test_unknown_parameter_type(self):
#         """Test handling of unknown parameter type."""
#         result = get_parameters_by_type_from_pint("unknown_type")
#         assert result == []
#
#     def test_empty_parameter_type(self):
#         """Test handling of empty parameter type."""
#         result = get_parameters_by_type_from_pint("")
#         assert result == []
#
#     def test_none_parameter_type(self):
#         """Test handling of None parameter type."""
#         result = get_parameters_by_type_from_pint(None)
#         assert result == []
#
#     def test_pint_discovery_failure_raises_error(self):
#         """Test that PINT discovery failure raises PINTDiscoveryError."""
#         with patch(
#             "metapulsar.pint_helpers.AllComponents",
#             side_effect=Exception("PINT error"),
#         ):
#             with pytest.raises(PINTDiscoveryError):
#                 get_parameters_by_type_from_pint("astrometry")


class TestCheckComponentAvailableInModel:
    """Test check_component_available_in_model function."""

    def test_component_available(self):
        """Test when component is available in model."""
        mock_model = Mock()
        mock_component = Mock()
        mock_component.category = "astrometry"
        mock_model.components = {"AstrometryEquatorial": mock_component}

        result = check_component_available_in_model(mock_model, "astrometry")
        assert result is True

    def test_component_not_available(self):
        """Test when component is not available in model."""
        mock_model = Mock()
        mock_component = Mock()
        mock_component.category = "other"
        mock_model.components = {"OtherComponent": mock_component}

        result = check_component_available_in_model(mock_model, "astrometry")
        assert result is False

    def test_unknown_component_type(self):
        """Test handling of unknown component type."""
        mock_model = Mock()
        result = check_component_available_in_model(mock_model, "unknown_type")
        assert result is False

    def test_empty_components_dict(self):
        """Test handling of empty components dictionary."""
        mock_model = Mock()
        mock_model.components = {}
        result = check_component_available_in_model(mock_model, "astrometry")
        assert result is False

    def test_none_components_attribute(self):
        """Test handling of None components attribute."""
        mock_model = Mock()
        mock_model.components = None
        # The current implementation doesn't handle None components gracefully
        with pytest.raises(AttributeError):
            check_component_available_in_model(mock_model, "astrometry")


class TestGetParameterIdentifiabilityFromModel:
    """Test get_parameter_identifiability_from_model function."""

    def test_parameter_identifiable(self):
        """Test when parameter is both fittable and free."""
        mock_model = Mock()
        mock_model.fittable_params = ["F0", "F1", "RAJ"]
        mock_model.free_params = ["F0", "RAJ"]

        result = get_parameter_identifiability_from_model(mock_model, "F0")
        assert result is True

    def test_parameter_not_fittable(self):
        """Test when parameter is not fittable."""
        mock_model = Mock()
        mock_model.fittable_params = ["F0", "F1"]
        mock_model.free_params = ["F0", "F1", "RAJ"]

        result = get_parameter_identifiability_from_model(mock_model, "RAJ")
        assert result is False

    def test_parameter_not_free(self):
        """Test when parameter is not free (frozen)."""
        mock_model = Mock()
        mock_model.fittable_params = ["F0", "F1", "RAJ"]
        mock_model.free_params = ["F0", "F1"]

        result = get_parameter_identifiability_from_model(mock_model, "RAJ")
        assert result is False

    def test_parameter_neither_fittable_nor_free(self):
        """Test when parameter is neither fittable nor free."""
        mock_model = Mock()
        mock_model.fittable_params = ["F0", "F1"]
        mock_model.free_params = ["F0"]

        result = get_parameter_identifiability_from_model(mock_model, "RAJ")
        assert result is False

    def test_empty_fittable_params(self):
        """Test when fittable_params is empty."""
        mock_model = Mock()
        mock_model.fittable_params = []
        mock_model.free_params = ["F0", "F1"]

        result = get_parameter_identifiability_from_model(mock_model, "F0")
        assert result is False

    def test_empty_free_params(self):
        """Test when free_params is empty."""
        mock_model = Mock()
        mock_model.fittable_params = ["F0", "F1"]
        mock_model.free_params = []

        result = get_parameter_identifiability_from_model(mock_model, "F0")
        assert result is False

    def test_nonexistent_parameter(self):
        """Test when parameter doesn't exist in either list."""
        mock_model = Mock()
        mock_model.fittable_params = ["F0", "F1"]
        mock_model.free_params = ["F0", "F1"]

        result = get_parameter_identifiability_from_model(mock_model, "NONEXISTENT")
        assert result is False


class TestGetParametersByTypeFromParfiles:
    """Test get_parameters_by_type_from_parfiles function."""

    @pytest.fixture
    def mock_parfile_dicts(self):
        """Fixture providing realistic mock parfile dictionaries with proper models."""
        return {
            "EPTA": {
                # Basic pulsar info
                "PSR": "J1857+0943",
                # Spindown parameters with PEPOCH (required for F1)
                "F0": "123.456",
                "F1": "-1.23e-15",
                "F2": "1.0e-30",
                "PEPOCH": "55000.0",
                # Astrometry parameters with POSEPOCH (required for proper motion)
                "RAJ": "18:57:36.3906121",
                "DECJ": "+09:43:17.20714",
                "PMRA": "10.5",
                "PMDEC": "-5.2",
                "POSEPOCH": "55000.0",
                # Dispersion parameters with DMEPOCH (required for DM2)
                "DM": "13.3",
                "DM1": "0.001",
                "DM2": "0.0001",
                "DM3": "0.00001",  # Dynamic derivative
                "DMEPOCH": "55000.0",
                # Binary parameters with complete BT model
                "BINARY": "BT",
                "A1": "1.2",
                "PB": "0.357",
                "ECC": "0.1",
                "OM": "90.0",
                "T0": "55000.0",
            },
            "PPTA": {
                # Basic pulsar info
                "PSR": "J1857+0943",
                # Spindown parameters with PEPOCH
                "F0": "123.456",
                "F1": "-1.23e-15",
                "F2": "1.0e-30",
                "F3": "1.0e-45",  # Dynamic derivative
                "PEPOCH": "55000.0",
                # Astrometry parameters (ecliptic coordinates)
                "ELONG": "284.4015854",
                "ELAT": "9.7213056",
                "PMELONG": "10.5",
                "PMELAT": "-5.2",
                "POSEPOCH": "55000.0",
                # Dispersion parameters
                "DM": "13.3",
                "DM1": "0.001",
                "DM2": "0.0001",
                "DMEPOCH": "55000.0",
                # Binary parameters with complete BT model
                "BINARY": "BT",
                "A1": "1.2",
                "PB": "0.357",
                "ECC": "0.1",
                "OM": "90.0",
                "T0": "55000.0",
            },
        }

    @patch("metapulsar.pint_helpers.create_pint_model")
    @patch("metapulsar.pint_helpers.get_category_mapping_from_pint")
    def test_spindown_parameters_with_dynamic_derivatives_and_pepoch(
        self, mock_get_category, mock_create_model, mock_parfile_dicts
    ):
        """Test that function discovers dynamic derivatives like F2, F3 and includes PEPOCH."""
        # Mock category mapping
        mock_get_category.return_value = {"spindown": "spindown"}

        # Mock PINT models with spindown components
        mock_model_epta = Mock()
        mock_model_epta.components = {
            "Spindown": Mock(category="spindown", params=["F0", "F1", "F2", "PEPOCH"])
        }

        mock_model_ppta = Mock()
        mock_model_ppta.components = {
            "Spindown": Mock(
                category="spindown", params=["F0", "F1", "F2", "F3", "PEPOCH"]
            )
        }

        mock_create_model.side_effect = [mock_model_epta, mock_model_ppta]

        result = get_parameters_by_type_from_parfiles("spindown", mock_parfile_dicts)

        # Should include dynamic derivatives from both PTAs
        assert "F0" in result
        assert "F1" in result
        assert "F2" in result
        assert "F3" in result  # Dynamic derivative from PPTA
        assert "PEPOCH" in result  # Required epoch parameter

        # Should include aliases (if they exist in the mock data)
        # Note: The function returns both canonical parameters and aliases
        # The specific aliases depend on the mock alias data provided

    @patch("metapulsar.pint_helpers.create_pint_model")
    @patch("metapulsar.pint_helpers.get_category_mapping_from_pint")
    def test_dispersion_parameters_with_dynamic_derivatives_and_dmepoch(
        self, mock_get_category, mock_create_model, mock_parfile_dicts
    ):
        """Test that function discovers DM dynamic derivatives like DM3 and includes DMEPOCH."""
        # Mock category mapping
        mock_get_category.return_value = {"dispersion": "dispersion"}

        # Mock PINT models with dispersion components
        mock_model_epta = Mock()
        mock_model_epta.components = {
            "Dispersion": Mock(
                category="dispersion", params=["DM", "DM1", "DM2", "DM3", "DMEPOCH"]
            )
        }

        mock_model_ppta = Mock()
        mock_model_ppta.components = {
            "Dispersion": Mock(
                category="dispersion", params=["DM", "DM1", "DM2", "DMEPOCH"]
            )
        }

        mock_create_model.side_effect = [mock_model_epta, mock_model_ppta]

        result = get_parameters_by_type_from_parfiles("dispersion", mock_parfile_dicts)

        # Should include dynamic derivatives
        assert "DM" in result
        assert "DM1" in result
        assert "DM2" in result
        assert "DM3" in result  # Dynamic derivative from EPTA
        assert "DMEPOCH" in result  # Required epoch parameter

    @patch("metapulsar.pint_helpers.create_pint_model")
    @patch("metapulsar.pint_helpers.get_category_mapping_from_pint")
    def test_binary_parameters_with_complete_bt_model(
        self, mock_get_category, mock_create_model, mock_parfile_dicts
    ):
        """Test that function discovers complete binary parameters for BT model."""
        # Mock category mapping
        mock_get_category.return_value = {"binary": "binary"}

        # Mock PINT models with binary components
        mock_model_epta = Mock()
        mock_model_epta.components = {
            "BinaryBT": Mock(category="binary", params=["A1", "PB", "ECC", "OM", "T0"])
        }

        mock_model_ppta = Mock()
        mock_model_ppta.components = {
            "BinaryBT": Mock(category="binary", params=["A1", "PB", "ECC", "OM", "T0"])
        }

        mock_create_model.side_effect = [mock_model_epta, mock_model_ppta]

        result = get_parameters_by_type_from_parfiles("binary", mock_parfile_dicts)

        # Should include complete binary parameters
        assert "A1" in result
        assert "PB" in result
        assert "ECC" in result
        assert "OM" in result
        assert "T0" in result

        # Should include aliases (if they exist in the mock data)
        # Note: The function returns both canonical parameters and aliases
        # The specific aliases depend on the mock alias data provided
        assert "E" in result  # Alias for ECC (from mock data)

    @patch("metapulsar.pint_helpers.create_pint_model")
    @patch("metapulsar.pint_helpers.get_category_mapping_from_pint")
    def test_parfile_parsing_failure_handled(
        self, mock_get_category, mock_create_model, mock_parfile_dicts
    ):
        """Test that function handles parfile parsing failures gracefully."""
        # Mock category mapping
        mock_get_category.return_value = {"spindown": "spindown"}

        # Mock PINT model creation to fail for one PTA
        mock_model_epta = Mock()
        mock_model_epta.components = {
            "Spindown": Mock(category="spindown", params=["F0", "F1", "PEPOCH"])
        }

        mock_create_model.side_effect = [
            mock_model_epta,  # EPTA succeeds
            Exception("PINT parsing failed"),  # PPTA fails
        ]

        with patch("loguru.logger.warning") as mock_warning:
            result = get_parameters_by_type_from_parfiles(
                "spindown", mock_parfile_dicts
            )

            # Should still return parameters from successful PTAs
            assert "F0" in result
            assert "F1" in result
            assert "PEPOCH" in result

            # Should log warning for failed PTA
            mock_warning.assert_called_once()
            assert "Failed to create PINT model for PTA PPTA" in str(
                mock_warning.call_args
            )

    def test_empty_parfile_dicts(self):
        """Test handling of empty parfile dictionaries."""
        result = get_parameters_by_type_from_parfiles("spindown", {})
        assert result == []

    def test_none_parfile_dicts(self):
        """Test handling of None parfile dictionaries."""
        with pytest.raises(AttributeError):
            get_parameters_by_type_from_parfiles("spindown", None)


class TestCreatePintModel:
    """Test create_pint_model function."""

    @patch("pint.models.model_builder.ModelBuilder")
    def test_create_pint_model_with_realistic_string_input(
        self, mock_model_builder_class
    ):
        """Test model creation from realistic parfile string content."""
        # Mock ModelBuilder
        mock_builder = Mock()
        mock_model_builder_class.return_value = mock_builder

        mock_model = Mock()
        mock_builder.return_value = mock_model

        # Test with realistic parfile string
        parfile_content = """
PSR J1857+0943
F0 123.456 1
F1 -1.23e-15 1
F2 1.0e-30 1
PEPOCH 55000.0
RAJ 18:57:36.3906121
DECJ +09:43:17.20714
PMRA 10.5 1
PMDEC -5.2 1
POSEPOCH 55000.0
DM 13.3 1
DM1 0.001 1
DM2 0.0001 1
DMEPOCH 55000.0
BINARY BT
A1 1.2 1
PB 0.357 1
ECC 0.1 1
OM 90.0 1
T0 55000.0 1
"""

        result = create_pint_model(parfile_content)

        # Should call ModelBuilder with StringIO
        mock_model_builder_class.assert_called_once()
        mock_builder.assert_called_once()

        # Check that StringIO was used
        call_args = mock_builder.call_args[0]
        assert hasattr(call_args[0], "read")  # StringIO object

        assert result == mock_model

    @patch("pint.models.model_builder.ModelBuilder")
    def test_create_pint_model_with_realistic_dict_input(
        self, mock_model_builder_class
    ):
        """Test model creation from realistic parfile dictionary."""
        # Mock ModelBuilder
        mock_builder = Mock()
        mock_model_builder_class.return_value = mock_builder

        mock_model = Mock()
        mock_builder.return_value = mock_model

        # Test with realistic dict input including binary model
        parfile_dict = {
            "PSR": "J1857+0943",
            "F0": "123.456",
            "F1": "-1.23e-15",
            "F2": "1.0e-30",
            "PEPOCH": "55000.0",
            "RAJ": "18:57:36.3906121",
            "DECJ": "+09:43:17.20714",
            "PMRA": "10.5",
            "PMDEC": "-5.2",
            "POSEPOCH": "55000.0",
            "DM": "13.3",
            "DM1": "0.001",
            "DM2": "0.0001",
            "DMEPOCH": "55000.0",
            "BINARY": "BT",
            "A1": "1.2",
            "PB": "0.357",
            "ECC": "0.1",
            "OM": "90.0",
            "T0": "55000.0",
        }

        result = create_pint_model(parfile_dict)

        # Should call ModelBuilder with dict directly
        mock_model_builder_class.assert_called_once()
        mock_builder.assert_called_once_with(
            parfile_dict, allow_tcb=True, allow_T2=True
        )

        assert result == mock_model

    @patch("pint.models.model_builder.ModelBuilder")
    def test_create_pint_model_missing_binary_model_error(
        self, mock_model_builder_class
    ):
        """Test error handling when binary parameters present but no BINARY model specified."""
        from pint.exceptions import MissingParameter

        # Mock ModelBuilder to raise MissingParameter
        mock_builder = Mock()
        mock_model_builder_class.return_value = mock_builder
        mock_builder.side_effect = MissingParameter(
            "binary", "BINARY", "BINARY model required for binary parameters"
        )

        with patch("loguru.logger.error") as mock_logger:
            with pytest.raises(MissingParameter):
                # Dict with binary parameters but no BINARY model
                create_pint_model(
                    {
                        "F0": "123.456",
                        "A1": "1.2",  # Binary parameter without BINARY model
                        "PB": "0.357",
                    }
                )

            # Should log the error
            mock_logger.assert_called_once()

    @patch("pint.models.model_builder.ModelBuilder")
    def test_create_pint_model_missing_epoch_error(self, mock_model_builder_class):
        """Test error handling when epoch parameters are missing."""
        from pint.exceptions import MissingParameter

        # Mock ModelBuilder to raise MissingParameter
        mock_builder = Mock()
        mock_model_builder_class.return_value = mock_builder
        mock_builder.side_effect = MissingParameter(
            "spindown", "PEPOCH", "PEPOCH required for F1"
        )

        with patch("loguru.logger.error") as mock_logger:
            with pytest.raises(MissingParameter):
                # Dict with F1 but no PEPOCH
                create_pint_model(
                    {"F0": "123.456", "F1": "-1.23e-15"}  # F1 without PEPOCH
                )

            # Should log the error
            mock_logger.assert_called_once()

    @patch("pint.models.model_builder.ModelBuilder")
    def test_create_pint_model_component_conflict_error(self, mock_model_builder_class):
        """Test error handling for ComponentConflict."""
        from pint.exceptions import ComponentConflict

        # Mock ModelBuilder to raise ComponentConflict
        mock_builder = Mock()
        mock_model_builder_class.return_value = mock_builder
        mock_builder.side_effect = ComponentConflict("Conflicting binary models")

        with patch("loguru.logger.error") as mock_logger:
            with pytest.raises(ComponentConflict):
                # Dict with conflicting binary models
                create_pint_model(
                    {
                        "F0": "123.456",
                        "BINARY": "BT",
                        "BINARY2": "DD",  # Conflicting models
                    }
                )

            # Should log the error
            mock_logger.assert_called_once()

    def test_create_pint_model_none_input(self):
        """Test handling of None input."""
        with pytest.raises(PINTDiscoveryError):
            create_pint_model(None)

    def test_create_pint_model_empty_string(self):
        """Test handling of empty string input."""
        with patch(
            "pint.models.model_builder.ModelBuilder"
        ) as mock_model_builder_class:
            mock_builder = Mock()
            mock_model_builder_class.return_value = mock_builder

            mock_model = Mock()
            mock_builder.return_value = mock_model

            result = create_pint_model("")

            # Should still call ModelBuilder
            mock_builder.assert_called_once()
            assert result == mock_model


class TestAstrometryStyleDetection:
    """Test alias-driven astrometry style detection helpers."""

    def test_has_parameter_alias_accepts_pint_aliases(self):
        from metapulsar.pint_helpers import has_parameter_alias

        parfile_dict = {"LAMBDA": ["244.347"], "BETA": ["-10.07"]}
        assert has_parameter_alias(parfile_dict, "ELONG")
        assert has_parameter_alias(parfile_dict, "ELAT")
        assert not has_parameter_alias(parfile_dict, "RAJ")

    def test_detect_astrometry_style_equatorial_aliases(self):
        from metapulsar.pint_helpers import detect_astrometry_style

        parfile_dict = {
            "RA": ["18:57:36.3937"],
            "DEC": ["+09:43:17.291"],
        }
        assert detect_astrometry_style(parfile_dict) == "equatorial"

    def test_detect_astrometry_style_ecliptic_aliases(self):
        from metapulsar.pint_helpers import detect_astrometry_style

        parfile_dict = {
            "ELONG": ["244.347"],
            "ELAT": ["-10.07"],
        }
        assert detect_astrometry_style(parfile_dict) == "ecliptic"

    def test_detect_astrometry_style_mixed_raises(self):
        from metapulsar.pint_helpers import detect_astrometry_style

        parfile_dict = {
            "RAJ": ["18:57:36.3937"],
            "DECJ": ["+09:43:17.291"],
            "LAMBDA": ["244.347"],
            "BETA": ["-10.07"],
        }
        with pytest.raises(ValueError, match="Mixed astrometry detected"):
            detect_astrometry_style(parfile_dict)
