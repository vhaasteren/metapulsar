"""Tests for standalone (non-BasePulsar) MetaPulsar pulsar behavior."""

import numpy as np

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


def _build_unsorted_mock(name: str, telescope: str, seed: int):
    """Create a mock libstempo pulsar with intentionally unsorted TOA rows."""
    psr = create_mock_libstempo(
        n_toas=40,
        name=name,
        telescope=telescope,
        include_astrometry=True,
        include_spin=True,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1000)
    permutation = rng.permutation(len(psr._toas_mjd))
    psr._toas_mjd = psr._toas_mjd[permutation]
    psr._residuals_s = psr._residuals_s[permutation]
    psr._toaerrs_us = psr._toaerrs_us[permutation]
    psr._freqs_hz = psr._freqs_hz[permutation]
    psr._telescope = psr._telescope[permutation]
    for key, values in psr._flag_dict.items():
        psr._flag_dict[key] = values[permutation]
    psr._designmatrix = psr._designmatrix[permutation, :]
    psr._psrPos = psr._psrPos[permutation, :]
    return psr


def test_metapulsar_is_standalone_pulsar(mock_metapulsar):
    pulsars = {
        "pta_a": _build_unsorted_mock("J1857+0943", "pta_a", seed=10),
        "pta_b": _build_unsorted_mock("J1857+0943", "pta_b", seed=20),
    }
    mp = mock_metapulsar(pulsars, combination_strategy="per_pta")

    assert type(mp).__mro__ == (MetaPulsar, object)


def test_metapulsar_preserves_storage_row_order_by_default(mock_metapulsar):
    pulsars = {
        "pta_a": _build_unsorted_mock("J1857+0943", "pta_a", seed=10),
        "pta_b": _build_unsorted_mock("J1857+0943", "pta_b", seed=20),
    }
    metapulsar = mock_metapulsar(pulsars, combination_strategy="per_pta")

    assert isinstance(metapulsar.isort, slice)
    assert metapulsar.isort == slice(None, None, None)
    assert isinstance(metapulsar.iisort, slice)
    assert metapulsar.iisort == slice(None, None, None)
    np.testing.assert_array_equal(metapulsar.toas, metapulsar._toas)
    np.testing.assert_array_equal(metapulsar.residuals, metapulsar._residuals)
    np.testing.assert_array_equal(metapulsar.toaerrs, metapulsar._toaerrs)
    np.testing.assert_array_equal(metapulsar.freqs, metapulsar._ssbfreqs)
    np.testing.assert_array_equal(metapulsar.Mmat, metapulsar._designmatrix)


def test_metapulsar_surface_arrays_are_row_aligned(mock_metapulsar):
    pulsars = {
        "pta_a": _build_unsorted_mock("J1857+0943", "pta_a", seed=10),
        "pta_b": _build_unsorted_mock("J1857+0943", "pta_b", seed=20),
    }
    metapulsar = mock_metapulsar(pulsars, combination_strategy="per_pta")

    ntoas = len(metapulsar._toas)
    assert len(metapulsar.toas) == ntoas
    assert len(metapulsar.residuals) == ntoas
    assert len(metapulsar.toaerrs) == ntoas
    assert len(metapulsar.freqs) == ntoas
    assert metapulsar.Mmat.shape[0] == ntoas
    assert len(metapulsar.flags["pta_dataset"]) == ntoas
    assert len(metapulsar.backend_flags) == ntoas
    assert len(metapulsar.telescope) == ntoas
