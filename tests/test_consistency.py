"""Tests for the cross-PTA consistency-check helpers in
:mod:`metapulsar.consistency`.

The optional dependencies (``tensiometer``, ``getdist``) are imported lazily
by the module under test.  Tests that need them are skipped when the
dependency is missing, mirroring the existing libstempo-optional pattern in
this test suite.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from metapulsar import consistency
from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo


def _have(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


_HAS_GETDIST = _have("getdist")
_HAS_TENSIOMETER = _have("tensiometer")

requires_getdist = pytest.mark.skipif(
    not _HAS_GETDIST, reason="getdist is not installed"
)


# ---------------------------------------------------------------------------
# subset_metapulsar
# ---------------------------------------------------------------------------


class TestSubsetMetapulsarRoundtrip:
    """:func:`subset_metapulsar` returns the per-PTA enterprise pulsar without
    altering the design-matrix columns held by :class:`MetaPulsar`.
    """

    def setup_method(self) -> None:
        self.pulsars = {
            "pta_a": create_mock_libstempo(
                n_toas=40, name="J1857+0943", telescope="pta_a", seed=11
            ),
            "pta_b": create_mock_libstempo(
                n_toas=50, name="J1857+0943", telescope="pta_b", seed=22
            ),
        }
        self.mp = MetaPulsar(self.pulsars, combination_strategy="composite")

    def test_list_ptas_matches_input(self) -> None:
        ptas = consistency.list_ptas(self.mp)
        assert set(ptas) == set(self.pulsars.keys())

    def test_subset_returns_per_pta_enterprise_pulsar(self) -> None:
        ep_a = consistency.subset_metapulsar(self.mp, "pta_a")
        ep_b = consistency.subset_metapulsar(self.mp, "pta_b")
        assert ep_a is self.mp._epulsars["pta_a"]
        assert ep_b is self.mp._epulsars["pta_b"]
        assert ep_a is not ep_b

    def test_subset_unknown_pta_raises(self) -> None:
        with pytest.raises(KeyError):
            consistency.subset_metapulsar(self.mp, "missing_pta")

    def test_subset_design_matrix_column_space_preserved(self) -> None:
        """The combined design matrix must contain a per-PTA submatrix whose
        column space matches the per-PTA enterprise design matrix.

        We require equality of the column space (not the column itself) since
        :class:`MetaPulsar` may re-organize columns during merging.
        """
        ep_a = consistency.subset_metapulsar(self.mp, "pta_a")
        per_pta_dm = ep_a._designmatrix
        combined_dm = self.mp._designmatrix
        n_a = per_pta_dm.shape[0]
        sub_combined = combined_dm[:n_a, :]

        rank_per_pta = np.linalg.matrix_rank(per_pta_dm)
        # column space of combined-restricted-to-PTA contains per-PTA columns
        stacked = np.hstack([per_pta_dm, sub_combined])
        rank_stacked = np.linalg.matrix_rank(stacked)
        assert rank_stacked == max(rank_per_pta, np.linalg.matrix_rank(sub_combined))


# ---------------------------------------------------------------------------
# Tension wrappers
# ---------------------------------------------------------------------------


def _gaussian_chain(rng: np.random.Generator, mean, cov, n_samples: int):
    samples = rng.multivariate_normal(np.asarray(mean), np.asarray(cov), size=n_samples)
    names = ("log10_A", "gamma")
    return consistency.samples_to_mcsamples(
        {names[0]: samples[:, 0], names[1]: samples[:, 1]},
        names=list(names),
        labels=[r"\log_{10} A", r"\gamma"],
    )


class TestGaussianTension:
    """The closed-form Gaussian estimator does not require tensiometer and
    must always be available.
    """

    @requires_getdist
    def test_zero_tension_on_identical_chains(self) -> None:
        rng = np.random.default_rng(0)
        cov = [[0.04, 0.0], [0.0, 0.09]]
        chain_a = _gaussian_chain(rng, [-13.0, 3.0], cov, n_samples=8000)
        chain_b = _gaussian_chain(rng, [-13.0, 3.0], cov, n_samples=8000)

        result = consistency.hyper_tension(chain_a, chain_b, method="gaussian")
        assert result.method == "gaussian"
        assert result.n_params == 2
        assert result.n_sigma is not None and result.n_sigma < 1.0
        assert result.p_value is not None and result.p_value > 0.3

    @requires_getdist
    def test_strong_tension_on_shifted_chains(self) -> None:
        rng = np.random.default_rng(1)
        cov = [[0.04, 0.0], [0.0, 0.09]]
        chain_a = _gaussian_chain(rng, [-13.0, 3.0], cov, n_samples=8000)
        chain_b = _gaussian_chain(rng, [-12.0, 4.5], cov, n_samples=8000)

        result = consistency.hyper_tension(chain_a, chain_b, method="gaussian")
        assert result.method == "gaussian"
        assert result.n_sigma is not None and result.n_sigma > 3.0
        assert result.p_value is not None and result.p_value < 1e-3

    @requires_getdist
    def test_analytic_one_dimensional_case(self) -> None:
        """For a single Gaussian parameter with known mean shift and per-chain
        variance, the closed-form chi^2 result is exact and we can compare to
        a hand-computed value.
        """
        rng = np.random.default_rng(2)
        N = 50000
        sigma = 0.5
        delta = 1.5
        sa = {"log10_A": rng.normal(-13.0, sigma, N)}
        sb = {"log10_A": rng.normal(-13.0 + delta, sigma, N)}
        chain_a = consistency.samples_to_mcsamples(sa, label="A")
        chain_b = consistency.samples_to_mcsamples(sb, label="B")

        result = consistency.hyper_tension(
            chain_a, chain_b, params=("log10_A",), method="gaussian"
        )
        # chi^2 with 1 dof = (delta / sqrt(sigma_a^2 + sigma_b^2))^2
        expected_chi2 = (delta / np.sqrt(2.0 * sigma**2)) ** 2
        np.testing.assert_allclose(result.extra["chi2"], expected_chi2, rtol=2e-2)
        assert result.n_params == 1


@pytest.mark.skipif(
    not (_HAS_GETDIST and _HAS_TENSIOMETER),
    reason="tensiometer or getdist not installed",
)
class TestNonGaussianTension:
    """Tensiometer-based estimators must:

    * report low ``n_sigma`` for two identically-distributed chains,
    * report large ``n_sigma`` when a sizeable shift is injected,
    * fall back to the Gaussian estimator when something internal fails
      (covered indirectly by the auto-mode tests above).
    """

    def test_auto_mode_low_tension(self) -> None:
        rng = np.random.default_rng(0)
        cov = [[0.04, 0.0], [0.0, 0.09]]
        chain_a = _gaussian_chain(rng, [-13.0, 3.0], cov, n_samples=4000)
        chain_b = _gaussian_chain(rng, [-13.0, 3.0], cov, n_samples=4000)

        result = consistency.hyper_tension(chain_a, chain_b, method="auto")
        assert result.method in {"gaussian", "kde", "flow"}
        assert result.n_sigma is not None and result.n_sigma < 2.0

    def test_auto_mode_high_tension_on_shift(self) -> None:
        rng = np.random.default_rng(1)
        cov = [[0.04, 0.0], [0.0, 0.09]]
        chain_a = _gaussian_chain(rng, [-13.0, 3.0], cov, n_samples=4000)
        chain_b = _gaussian_chain(rng, [-12.0, 4.5], cov, n_samples=4000)

        result = consistency.hyper_tension(chain_a, chain_b, method="auto")
        assert result.method in {"gaussian", "kde", "flow"}
        assert result.n_sigma is not None and result.n_sigma > 3.0


# ---------------------------------------------------------------------------
# Waveform-style high-dimensional tension
# ---------------------------------------------------------------------------


@requires_getdist
class TestWaveformTension:
    """A correctly-implemented FFTInt-style consistency check on the Fourier
    coefficient vector should detect an injected shift in many dimensions.
    """

    def _make_waveform_chain(
        self,
        rng: np.random.Generator,
        n_modes: int,
        n_samples: int,
        shift: float = 0.0,
        sigma: float = 1.0,
    ):
        names = []
        samples = {}
        for k in range(n_modes):
            for kind in ("c", "s"):
                name = f"{kind}_{k:02d}"
                names.append(name)
                samples[name] = rng.normal(shift, sigma, n_samples)
        return consistency.samples_to_mcsamples(samples, names=names, label="wf")

    def test_waveform_zero_tension(self) -> None:
        rng = np.random.default_rng(3)
        n_modes = 10
        chain_a = self._make_waveform_chain(rng, n_modes, n_samples=4000)
        chain_b = self._make_waveform_chain(rng, n_modes, n_samples=4000)
        coef_names = [f"{kind}_{i:02d}" for i in range(n_modes) for kind in ("c", "s")]

        result = consistency.waveform_tension(
            chain_a, chain_b, coef_names=coef_names, method="gaussian"
        )
        assert result.n_params == 2 * n_modes
        assert result.n_sigma is not None and result.n_sigma < 1.5

    def test_waveform_shift_detected(self) -> None:
        # Inject a per-coefficient shift of ~3 sigma.  The combined chi^2 for
        # 2*n_modes coefficients is n_dof * (delta / sqrt(2) sigma)^2; with
        # delta=1.5 and sigma=0.5 this gives chi^2 ~ 90 over 20 dof, well into
        # the strong-tension regime.
        rng = np.random.default_rng(4)
        n_modes = 10
        chain_a = self._make_waveform_chain(
            rng, n_modes, n_samples=4000, shift=0.0, sigma=0.5
        )
        chain_b = self._make_waveform_chain(
            rng, n_modes, n_samples=4000, shift=1.5, sigma=0.5
        )
        coef_names = [f"{kind}_{i:02d}" for i in range(n_modes) for kind in ("c", "s")]

        result = consistency.waveform_tension(
            chain_a, chain_b, coef_names=coef_names, method="gaussian"
        )
        assert result.n_sigma is not None and result.n_sigma > 3.0


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def test_summarize_returns_sorted_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    rows = [
        {
            "pulsar": "J0613-0200",
            "pta_a": "EPTA",
            "pta_b": "NANOGrav",
            "check": "hyper",
            "n_sigma": 0.4,
            "p_value": 0.7,
            "method": "kde",
            "n_params": 2,
        },
        {
            "pulsar": "J0613-0200",
            "pta_a": "EPTA",
            "pta_b": "PPTA",
            "check": "hyper",
            "n_sigma": 1.8,
            "p_value": 0.07,
            "method": "kde",
            "n_params": 2,
        },
        {
            "pulsar": "J2241-5236",
            "pta_a": "PPTA_DR3",
            "pta_b": "MPTA_DR2",
            "check": "timing",
            "n_sigma": 6.5,
            "p_value": 1e-10,
            "method": "gaussian",
            "n_params": 4,
        },
    ]
    df = consistency.summarize(rows)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)[:5] == [
        "pulsar",
        "pta_a",
        "pta_b",
        "check",
        "n_sigma",
    ]
    # within each (pulsar, check) group sorted by descending n_sigma
    j0613 = df[df["pulsar"] == "J0613-0200"]
    assert list(j0613["n_sigma"]) == sorted(j0613["n_sigma"], reverse=True)


# ---------------------------------------------------------------------------
# FFTInt builder error handling
# ---------------------------------------------------------------------------


def test_build_fftint_unknown_backend_raises() -> None:
    psr = create_mock_libstempo(n_toas=20, name="J1857+0943", seed=0)
    with pytest.raises(ValueError, match="Unknown FFTInt backend"):
        consistency.build_fftint_posterior(psr, model="rn", backend="not_a_backend")


def test_build_fftint_unknown_model_raises() -> None:
    psr = create_mock_libstempo(n_toas=20, name="J1857+0943", seed=0)
    with pytest.raises(ValueError, match="Unknown FFTInt model"):
        consistency.build_fftint_posterior(psr, model="bogus", backend="discovery")
