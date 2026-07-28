"""Enterprise Selection class integration tests for staggered selections."""

import pytest

pytest.importorskip("enterprise")
pytestmark = pytest.mark.requires_enterprise

import numpy as np  # noqa: E402
from enterprise.pulsar import Tempo2Pulsar  # noqa: E402
from enterprise.signals.selections import Selection  # noqa: E402

from metapulsar.mockpulsar import MockLibstempo  # noqa: E402
from metapulsar.selection_utils import create_staggered_selection  # noqa: E402


def _make_enterprise_pulsar(flags_dict, freqs_mhz):
    """Create a real Enterprise Pulsar from mock libstempo data."""
    n_toas = len(freqs_mhz)
    toas_mjd = np.linspace(50000.0, 60000.0, n_toas)
    residuals_s = np.zeros(n_toas)
    toaerrs_us = np.ones(n_toas) * 0.1
    freqs_hz = np.asarray(freqs_mhz) * 1e6
    mock_lt = MockLibstempo(
        toas_mjd,
        residuals_s,
        toaerrs_us,
        freqs_hz,
        flags_dict,
        "mock",
        "test_pulsar",
    )
    return Tempo2Pulsar(mock_lt, planets=False)


class TestEnterpriseIntegration:
    """Test Enterprise Selection class integration."""

    def test_selection_function_directly(self):
        """Test the raw selection function without Enterprise wrapper"""
        sel_func = create_staggered_selection("test", {"group": None})

        # Test with mock data directly
        flags = {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])}
        freqs = np.array([100.0, 200.0, 300.0])

        result = sel_func(flags, freqs)
        expected = {
            "test_ASP_430": np.array([True, False, True]),
            "test_ASP_800": np.array([False, True, False]),
        }

        assert isinstance(result, dict)
        assert all(isinstance(mask, np.ndarray) for mask in result.values())
        assert all(mask.dtype == bool for mask in result.values())
        assert set(result.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_enterprise_selection_wrapper(self):
        """Test that selection function works with Enterprise Selection class"""
        sel_func = create_staggered_selection("test", {"group": None})
        selection = Selection(sel_func)

        mock_psr = _make_enterprise_pulsar(
            {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])},
            np.array([100.0, 200.0, 300.0]),
        )

        # Test selection instance creation
        selection_instance = selection(mock_psr)
        masks = selection_instance.masks

        expected = {
            "test_ASP_430": np.array([True, False, True]),
            "test_ASP_800": np.array([False, True, False]),
        }

        assert isinstance(masks, dict)
        assert all(isinstance(mask, np.ndarray) for mask in masks.values())
        assert all(mask.dtype == bool for mask in masks.values())
        assert set(masks.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_array_equal(masks[key], expected[key])

    def test_parameter_generation(self):
        """Test that selection can generate parameters correctly"""
        sel_func = create_staggered_selection("test", {"group": None})
        selection = Selection(sel_func)

        mock_psr = _make_enterprise_pulsar(
            {"group": np.array(["ASP_430", "ASP_800", "ASP_430"])},
            np.array([100.0, 200.0, 300.0]),
        )

        selection_instance = selection(mock_psr)

        # Test parameter generation
        params, masks = selection_instance("efac", lambda x: f"param_{x}")

        expected_params = {
            "test_ASP_430_efac": "param_test_pulsar_test_ASP_430_efac",
            "test_ASP_800_efac": "param_test_pulsar_test_ASP_800_efac",
        }
        expected_masks = {
            "test_ASP_430_efac": np.array([True, False, True]),
            "test_ASP_800_efac": np.array([False, True, False]),
        }

        assert isinstance(params, dict)
        assert isinstance(masks, dict)
        assert all(key.endswith("_efac") for key in params.keys())
        assert set(params.keys()) == set(expected_params.keys())
        assert set(masks.keys()) == set(expected_masks.keys())
        for key in expected_masks:
            np.testing.assert_array_equal(masks[key], expected_masks[key])
