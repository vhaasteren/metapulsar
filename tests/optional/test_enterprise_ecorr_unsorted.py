"""Enterprise ECORR construction smoke tests on unsorted standalone pulsars."""

import pytest

pytest.importorskip("enterprise")
pytestmark = pytest.mark.requires_enterprise

import numpy as np  # noqa: E402
import enterprise.signals.parameter as parameter  # noqa: E402
from enterprise.signals import white_signals  # noqa: E402

from metapulsar.metapulsar import MetaPulsar  # noqa: E402
from metapulsar.mockpulsar import (  # noqa: E402
    create_mock_libstempo,
    write_mock_pta_files,
)


def _build_unsorted_pulsar(directory):
    pulsars = {}
    for idx, pta in enumerate(["pta_a", "pta_b"], start=1):
        psr = create_mock_libstempo(
            n_toas=40,
            name="J1857+0943",
            telescope=pta,
            seed=idx,
        )
        # Break monotonic TOA ordering to exercise unsorted-pulsar ECORR setup paths.
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
    return MetaPulsar(
        pulsars,
        combination_strategy="per_pta",
        pta_files=write_mock_pta_files(pulsars, directory),
    )


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


def test_ecorr_sherman_morrison_constructs_on_unsorted_pulsar(tmp_path):
    pulsar = _build_unsorted_pulsar(tmp_path)
    ecorr = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(-7.0),
        method="sherman-morrison",
    )

    signal = ecorr(pulsar)

    assert signal is not None


def test_ecorr_fast_sherman_morrison_constructs_on_unsorted_pulsar(tmp_path):
    pulsar = _build_unsorted_pulsar(tmp_path)
    ecorr = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(-7.0),
        method="fast-sherman-morrison",
    )

    signal = ecorr(pulsar)

    assert signal is not None


class _PermutedView:
    """The attributes Enterprise's ECORR binds, in a chosen row order."""

    def __init__(self, pulsar, permutation):
        self.name = pulsar.name
        self.toas = np.asarray(pulsar.toas)[permutation]
        self.stoas = np.asarray(pulsar.stoas)[permutation]
        self.toaerrs = np.asarray(pulsar.toaerrs)[permutation]
        self.freqs = np.asarray(pulsar.freqs)[permutation]
        self.residuals = np.asarray(pulsar.residuals)[permutation]
        self.backend_flags = np.asarray(pulsar.backend_flags)[permutation]
        self.flags = {k: np.asarray(v)[permutation] for k, v in pulsar.flags.items()}
        self.Mmat = np.asarray(pulsar.Mmat)[permutation, :]
        self.fitpars = list(pulsar.fitpars)


def _permuted_view(pulsar, permutation):
    return _PermutedView(pulsar, permutation)


@pytest.mark.parametrize("method", ["sherman-morrison", "fast-sherman-morrison"])
def test_ecorr_unsorted_solve_matches_sorted_equivalent(method, tmp_path):
    pulsars = _build_permuted_pulsars()
    pta_files = write_mock_pta_files(pulsars, tmp_path / "pta_files")
    unsorted_pulsar = MetaPulsar(
        pulsars, combination_strategy="per_pta", pta_files=pta_files
    )
    permutation = np.argsort(unsorted_pulsar.toas, kind="mergesort")
    # MetaPulsar does not sort -- it never did by default, and since D12 it
    # cannot. The claim under test is Enterprise's, not MetaPulsar's: that
    # ECORR's solve is order-equivariant, because `create_quantization_matrix`
    # groups TOAs by value rather than adjacency. So the sorted view is built
    # here, by permuting the arrays handed to Enterprise.
    sorted_pulsar = _permuted_view(unsorted_pulsar, permutation)
    np.testing.assert_allclose(sorted_pulsar.toas, unsorted_pulsar.toas[permutation])

    ecorr = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(-7.0),
        method=method,
    )
    unsorted_signal = ecorr(unsorted_pulsar)
    sorted_signal = ecorr(sorted_pulsar)
    nvec_unsorted = np.full(len(unsorted_pulsar.toas), 1.0e-12, dtype=float)
    nvec_sorted = nvec_unsorted[permutation]
    ndiag_unsorted = unsorted_signal.get_ndiag({}) + nvec_unsorted
    ndiag_sorted = sorted_signal.get_ndiag({}) + nvec_sorted

    probe = np.linspace(-1.0, 1.0, len(unsorted_pulsar.toas))
    solved_unsorted, logdet_unsorted = ndiag_unsorted.solve(probe, logdet=True)
    solved_sorted, logdet_sorted = ndiag_sorted.solve(probe[permutation], logdet=True)

    np.testing.assert_allclose(solved_sorted, solved_unsorted[permutation], rtol=1e-12)
    np.testing.assert_allclose(logdet_sorted, logdet_unsorted, rtol=1e-12)
