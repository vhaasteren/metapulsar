"""Enterprise ECORR construction smoke tests on unsorted standalone hosts."""

import numpy as np
import pytest

import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


def _build_unsorted_host():
    pulsars = {}
    for idx, pta in enumerate(["pta_a", "pta_b"], start=1):
        psr = create_mock_libstempo(
            n_toas=40,
            name="J1857+0943",
            telescope=pta,
            seed=idx,
        )
        # Break monotonic TOA ordering to exercise unsorted-host ECORR setup paths.
        permutation = np.random.default_rng(idx + 100).permutation(len(psr._toas_mjd))
        psr._toas_mjd = psr._toas_mjd[permutation]
        psr._residuals_s = psr._residuals_s[permutation]
        psr._toaerrs_us = psr._toaerrs_us[permutation]
        psr._freqs_hz = psr._freqs_hz[permutation]
        psr._telescope = psr._telescope[permutation]
        for key, values in psr._flag_dict.items():
            psr._flag_dict[key] = values[permutation]
        psr._designmatrix = psr._designmatrix[permutation, :]
        psr._psrPos = psr._psrPos[permutation, :]
        pulsars[pta] = psr
    return MetaPulsar(pulsars, combination_strategy="composite")


def _build_permuted_pulsars():
    pulsars = {}
    for idx, pta in enumerate(["pta_a", "pta_b"], start=1):
        psr = create_mock_libstempo(
            n_toas=40,
            name="J1857+0943",
            telescope=pta,
            seed=idx,
        )
        permutation = np.random.default_rng(idx + 100).permutation(len(psr._toas_mjd))
        psr._toas_mjd = psr._toas_mjd[permutation]
        psr._toas_mjd[0:2] = psr._toas_mjd[0]
        psr._toas_mjd[2:4] = psr._toas_mjd[2]
        psr._residuals_s = psr._residuals_s[permutation]
        psr._toaerrs_us = psr._toaerrs_us[permutation]
        psr._freqs_hz = psr._freqs_hz[permutation]
        psr._telescope = psr._telescope[permutation]
        for key, values in psr._flag_dict.items():
            psr._flag_dict[key] = values[permutation]
        psr._designmatrix = psr._designmatrix[permutation, :]
        psr._psrPos = psr._psrPos[permutation, :]
        pulsars[pta] = psr
    return pulsars


def test_ecorr_sherman_morrison_constructs_on_unsorted_host():
    host = _build_unsorted_host()
    ecorr = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(-7.0),
        method="sherman-morrison",
    )

    signal = ecorr(host)

    assert signal is not None


def test_ecorr_fast_sherman_morrison_constructs_on_unsorted_host():
    host = _build_unsorted_host()
    ecorr = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(-7.0),
        method="fast-sherman-morrison",
    )

    signal = ecorr(host)

    assert signal is not None


@pytest.mark.parametrize("method", ["sherman-morrison", "fast-sherman-morrison"])
def test_ecorr_unsorted_solve_matches_sorted_equivalent(method):
    pulsars = _build_permuted_pulsars()
    unsorted_host = MetaPulsar(pulsars, combination_strategy="composite", sort=False)
    sorted_host = MetaPulsar(pulsars, combination_strategy="composite", sort=True)
    permutation = np.argsort(unsorted_host.toas, kind="mergesort")
    np.testing.assert_allclose(sorted_host.toas, unsorted_host.toas[permutation])

    ecorr = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(-7.0),
        method=method,
    )
    unsorted_signal = ecorr(unsorted_host)
    sorted_signal = ecorr(sorted_host)
    nvec_unsorted = np.full(len(unsorted_host.toas), 1.0e-12, dtype=float)
    nvec_sorted = nvec_unsorted[permutation]
    ndiag_unsorted = unsorted_signal.get_ndiag({}) + nvec_unsorted
    ndiag_sorted = sorted_signal.get_ndiag({}) + nvec_sorted

    probe = np.linspace(-1.0, 1.0, len(unsorted_host.toas))
    solved_unsorted, logdet_unsorted = ndiag_unsorted.solve(probe, logdet=True)
    solved_sorted, logdet_sorted = ndiag_sorted.solve(probe[permutation], logdet=True)

    np.testing.assert_allclose(solved_sorted, solved_unsorted[permutation], rtol=1e-12)
    np.testing.assert_allclose(logdet_sorted, logdet_unsorted, rtol=1e-12)
