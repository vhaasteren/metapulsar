"""Enterprise Tempo2Pulsar integration tests for MockLibstempo."""

import pytest

pytest.importorskip("enterprise")
pytestmark = pytest.mark.requires_enterprise

import numpy as np  # noqa: E402
from enterprise.pulsar import Tempo2Pulsar  # noqa: E402

from metapulsar.mockpulsar import MockLibstempo, create_mock_libstempo  # noqa: E402


class TestTempo2PulsarIntegration:
    def test_tempo2pulsar_creation(self):
        mock_lt = create_mock_libstempo(
            n_toas=20,
            name="J1857+0943",
            seed=42,
        )
        psr = Tempo2Pulsar(mock_lt, planets=True)
        assert psr.name == "J1857+0943"
        assert "Offset" in psr.fitpars
        assert "F0" in psr.fitpars
        assert len(psr._toas) == 20
        assert psr._designmatrix.shape[1] == len(psr.fitpars)

    def test_unit_conversions_correct(self):
        toas_mjd = np.array([50000.0, 50001.0, 50002.0])
        residuals_s = np.array([1e-6, 2e-6, 3e-6])
        toaerrs_us = np.array([0.1, 0.2, 0.3])
        freqs_hz = np.array([1e8, 2e8, 3e8])
        flags = {"telescope": np.array(["GBT"] * 3)}
        mock_lt = MockLibstempo(
            toas_mjd,
            residuals_s,
            toaerrs_us,
            freqs_hz,
            flags,
            "GBT",
            "J1857+0943",
        )
        psr = Tempo2Pulsar(mock_lt, planets=True)
        np.testing.assert_allclose(psr._toas, toas_mjd * 86400, rtol=1e-12)
        np.testing.assert_allclose(psr._toaerrs, toaerrs_us * 1e-6, rtol=1e-12)
        np.testing.assert_allclose(psr._ssbfreqs, freqs_hz / 1e6, rtol=1e-12)
        np.testing.assert_array_equal(psr._residuals, residuals_s)
