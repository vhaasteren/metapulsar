"""Enterprise ECORR construction smoke tests on unsorted standalone hosts."""

import numpy as np

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
