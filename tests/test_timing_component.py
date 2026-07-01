"""Slice-5 tests for NonLinearTimingModel component behavior."""

import numpy as np
import pytest

import jax.random as jr
from numpyro import handlers

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugTimingBackend
from metapulsar.timing.backends.pint import LinearizedPintTimingBackend
from metapulsar.timing.component import NonLinearTimingModel
from metapulsar.timing.partition import resolve_partition
from metapulsar.timing.whitening import schur_delta_wls


class _Host:
    def __init__(self):
        self.name = "J0000+0000"
        self.fitpars = ("F0", "F1", "DM")
        self._toas = np.linspace(0.0, 1.0, 8)
        self._residuals = np.linspace(-2e-6, 2e-6, 8)
        self._toaerrs = np.full(8, 1.0e-6, dtype=float)
        self._freqs = np.full(8, 1400.0, dtype=float)
        self._flags = {"pta": np.array(["demo"] * 8, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 8, dtype="U8")
        self._cache_token = "token-v1"
        self.backend_calls = []
        self.default_analytically_marginalize = ["F0", "DM"]

        design = np.array(
            [
                [1.0, 0.0, 0.2],
                [1.0, 0.1, 0.3],
                [1.0, 0.2, -0.1],
                [1.0, -0.3, 0.4],
                [1.0, 0.4, -0.2],
                [1.0, -0.5, 0.1],
                [1.0, 0.6, -0.2],
                [1.0, -0.7, 0.3],
            ],
            dtype=float,
        )
        self._design = design
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=design,
            theta_exact={"F0": "10.0", "F1": "1.0", "DM": "5.0"},
        )
        self._jug_backend = LinearizedJugTimingBackend.from_linear_model(model)
        self._pint_backend = LinearizedPintTimingBackend.from_linear_model(model)

    @property
    def toas(self):
        return self._toas

    @property
    def residuals(self):
        return self._residuals

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def freqs(self):
        return self._freqs

    @property
    def Mmat(self):
        return self._design

    @property
    def flags(self):
        return self._flags

    @property
    def backend_flags(self):
        return self._backend_flags

    def cache_token(self):
        return self._cache_token

    def pint_model(self):
        return object()

    def timing_backend(self, name: str, **kwargs):
        self.backend_calls.append((name, dict(kwargs)))
        if name == "jug":
            return self._jug_backend
        if name == "pint":
            return self._pint_backend
        raise ValueError(f"Unsupported backend for test host: {name}")


@pytest.fixture
def host():
    return _Host()


def _monkeypatch_numpyro(monkeypatch, sample_value):
    calls = {"sample": [], "factor": [], "deterministic": []}

    def _sample(name, dist):
        calls["sample"].append((name, dist))
        return sample_value

    def _factor(name, value):
        calls["factor"].append((name, value))

    def _deterministic(name, value):
        calls["deterministic"].append((name, value))

    monkeypatch.setattr("numpyro.sample", _sample)
    monkeypatch.setattr("numpyro.factor", _factor)
    monkeypatch.setattr("numpyro.deterministic", _deterministic)
    return calls


def _schur_fisher(host, *, analytically_marginalize, variance):
    part = resolve_partition(host, analytically_marginalize=analytically_marginalize)
    return schur_delta_wls(host=host, partition=part, variance=variance).fisher


def _covariance_from_fisher(fisher: np.ndarray) -> np.ndarray:
    import scipy.linalg as sl

    cf = sl.cho_factor(fisher)
    return sl.cho_solve(cf, np.eye(fisher.shape[0], dtype=float))


def _z_space_fisher(space, delta_fisher):
    jac = np.asarray(
        space.prior_bijector.jacobian_diag_delta_from_z(space.linear.z0, np),
        dtype=float,
    )
    J = np.diag(jac)
    return J @ delta_fisher @ J


def test_component_config_only_build_and_with_backend():
    ntm = NonLinearTimingModel(
        backend="jug",
        jug_compatibility="tempo2",
        transform="whitening",
        analytically_marginalize=["F0"],
        prior_policy="fallback",
        whitening_config={"name": "diagonal_white"},
        name="timing",
    )
    swapped = ntm.with_backend("pint")

    assert ntm.backend == "jug"
    assert ntm.jug_compatibility == "tempo2"
    assert swapped.backend == "pint"
    assert swapped.transform == ntm.transform
    assert swapped.analytically_marginalize == ntm.analytically_marginalize
    assert swapped.prior_policy == ntm.prior_policy
    assert swapped.whitening_config == ntm.whitening_config


def test_space_cached_and_invalidated_by_cache_token(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    first = ntm.space(host)
    second = ntm.space(host)
    assert first is second

    host._cache_token = "token-v2"
    third = ntm.space(host)
    assert third is not first


def test_whitening_named_builders_bind_from_serializable_configs(host):
    ntm_default = NonLinearTimingModel(
        backend="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
    )
    ntm_fixed = NonLinearTimingModel(
        backend="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        whitening_config={
            "name": "fixed_hyperparameters",
            "hyperparameters": {"efac": {"demo": 1.2}, "equad": {"demo": 1.0e-7}},
        },
    )
    sp_default = ntm_default.space(host)
    sp_fixed = ntm_fixed.space(host)

    assert sp_default.linear.C.shape == (1, 1)
    assert sp_fixed.linear.C.shape == (1, 1)
    assert float(sp_default.linear.C[0, 0]) != float(sp_fixed.linear.C[0, 0])


def test_whitening_builders_condition_fisher_to_unit_scale(host):
    analytically_marginalize_cfg = None
    ntm_default = NonLinearTimingModel(
        backend="jug",
        transform="whitening",
        analytically_marginalize=analytically_marginalize_cfg,
    )
    ntm_fixed = NonLinearTimingModel(
        backend="jug",
        transform="whitening",
        analytically_marginalize=analytically_marginalize_cfg,
        whitening_config={
            "name": "fixed_hyperparameters",
            "hyperparameters": {"efac": {"demo": 1.2}, "equad": {"demo": 1.0e-7}},
        },
    )

    default_fisher = _schur_fisher(
        host,
        analytically_marginalize=analytically_marginalize_cfg,
        variance=np.asarray(host.toaerrs, dtype=float) ** 2,
    )
    labels = np.asarray(host.backend_flags)
    efac = np.asarray([1.2 if label == "demo" else 1.0 for label in labels])
    fixed_fisher = _schur_fisher(
        host,
        analytically_marginalize=analytically_marginalize_cfg,
        variance=(efac * np.asarray(host.toaerrs, dtype=float)) ** 2 + 1.0e-14,
    )

    for space, fisher in (
        (ntm_default.space(host), default_fisher),
        (ntm_fixed.space(host), fixed_fisher),
    ):
        fisher_z = _z_space_fisher(space, fisher)
        conditioned = space.linear.C.T @ fisher_z @ space.linear.C
        np.testing.assert_allclose(
            conditioned,
            np.eye(len(space.names)),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_standardized_builder_uses_z_space_marginal_scales(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )
    space = ntm.space(host)
    fisher = _schur_fisher(
        host,
        analytically_marginalize=None,
        variance=np.asarray(host.toaerrs, dtype=float) ** 2,
    )
    fisher_z = _z_space_fisher(space, fisher)
    covariance_z = _covariance_from_fisher(fisher_z)

    assert np.allclose(space.linear.C, np.diag(np.diag(space.linear.C)))
    np.testing.assert_allclose(
        np.diag(space.linear.C),
        np.sqrt(np.diag(covariance_z)),
    )


def test_cheat_wls_prior_is_wide_uniform_box(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )
    block = ntm.priors(host)
    fisher = _schur_fisher(
        host,
        analytically_marginalize=None,
        variance=np.asarray(host.toaerrs, dtype=float) ** 2,
    )
    expected_stds = np.sqrt(np.diag(_covariance_from_fisher(fisher)))

    assert set(block.sources.values()) == {"cheat_wls"}
    assert all(prior.family == "uniform" for prior in block.priors)
    # Flat box of half-width scale * sigma, centered on the par-file value
    # (F0/F1/DM are physically unbounded here, so no clipping occurs).
    half_widths = np.array(
        [(prior.upper - prior.lower) / 2.0 for prior in block.priors], dtype=float
    )
    centers = np.array(
        [(prior.upper + prior.lower) / 2.0 for prior in block.priors], dtype=float
    )
    np.testing.assert_allclose(
        half_widths, ntm.cheat_prior_scale * expected_stds, rtol=1e-6
    )
    np.testing.assert_allclose(centers, 0.0, atol=1e-12)


def test_cheat_prior_box_clipped_to_physical_bounds():
    class _BoundedHost(_Host):
        def __init__(self):
            super().__init__()
            self.fitpars = ("ECC", "F1", "DM")
            self._ecc_ref = 1.0e-5
            model = LinearModel.from_host(
                fitpars=self.fitpars,
                design=self._design,
                theta_exact={"ECC": repr(self._ecc_ref), "F1": "1.0", "DM": "5.0"},
            )
            self._jug_backend = LinearizedJugTimingBackend.from_linear_model(model)

    bounded = _BoundedHost()
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=["F1", "DM"],
        name="timing",
    )
    block = ntm.priors(bounded)
    fisher = _schur_fisher(
        bounded,
        analytically_marginalize=["F1", "DM"],
        variance=np.asarray(bounded.toaerrs, dtype=float) ** 2,
    )
    sigma_ecc = float(np.sqrt(_covariance_from_fisher(fisher)[0, 0]))
    half = ntm.cheat_prior_scale * sigma_ecc

    ecc_prior = block.priors[block.names.index("ECC")]
    assert ecc_prior.family == "uniform"
    # 50 sigma extends below 0, so the lower edge is clipped to ECC >= 0.
    assert half > bounded._ecc_ref
    np.testing.assert_allclose(ecc_prior.lower, -bounded._ecc_ref, rtol=1e-9)
    np.testing.assert_allclose(ecc_prior.upper, half, rtol=1e-6)


def test_discovery_signals_delta_only_and_jax_gate(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    signals = ntm.discovery_signals(host)
    delay = signals[-1]
    output = np.asarray(delay({f"{host.name}_timing_F1": 0.25}), dtype=float)
    expected = -host._jug_backend.residual_delta(np.array([0.0, 0.25, 0.0]))
    np.testing.assert_allclose(output, expected)

    ntm_nonjax = NonLinearTimingModel(
        backend="pint",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    with pytest.raises(ValueError, match="JAX-capable backend"):
        ntm_nonjax.discovery_signals(host)


def test_all_analytically_marginalized_paths(host):
    ntm = NonLinearTimingModel(
        backend="pint",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    assert len(ntm.discovery_signals(host)) == 1
    ent = ntm.enterprise_signal()
    bound = ent(host)
    assert hasattr(bound, "get_basis")
    assert ntm.timing_param_keys(host) == ()


def test_non_timing_params_and_timing_param_keys_are_plain_set_subtraction(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    keys = ntm.timing_param_keys(host)
    assert keys[0] == f"{host.name}_timing_x"
    assert keys[1:] == (f"{host.name}_timing_F1",)
    params = ("efac", keys[0], "gamma", keys[1], "log10_A")
    assert ntm.non_timing_params(host, params) == ("efac", "gamma", "log10_A")

    ntm_all_marg = NonLinearTimingModel(
        backend="jug",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    plain = ("efac", "gamma", "log10_A")
    assert ntm_all_marg.timing_param_keys(host) == ()
    assert ntm_all_marg.non_timing_params(host, plain) == plain


def test_contribute_timing_samples_joint_site_factors_prior_and_injects_delta(
    host, monkeypatch
):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([0.2]))

    out = ntm.contribute_timing(host, {"efac": 1.0})

    assert f"{host.name}_timing_F1" in out
    assert out["efac"] == 1.0
    assert calls["sample"][0][0] == f"{host.name}_timing_x"
    assert calls["factor"][0][0] == f"{host.name}_timing_x_logprior"


def test_contribute_timing_noop_when_no_sampled(host, monkeypatch):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([]))
    params = {"efac": 1.0}
    out = ntm.contribute_timing(host, params)
    assert out is params
    assert calls["sample"] == []
    assert calls["factor"] == []


def test_set_prior_validated_against_sampled_partition(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    ntm.set_prior("F0", "uniform", lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="non-sampled"):
        ntm.space(host)


def test_set_prior_unknown_name_raises(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    ntm.set_prior("F11", "uniform", lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="unknown fit parameters"):
        ntm.space(host)


def test_enterprise_signal_forwards_jug_compatibility(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        jug_compatibility="tempo2",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    ent = ntm.enterprise_signal()
    _ = ent(host)
    assert ("jug", {"jug_compatibility": "tempo2"}) in host.backend_calls


def test_contribute_timing_improper_uniform_site_has_vector_event_shape(host):
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    def model():
        ntm.contribute_timing(host, {})

    substituted = handlers.substitute(
        model,
        data={f"{host.name}_timing_x": np.array([0.0])},
    )
    trace = handlers.trace(handlers.seed(substituted, jr.PRNGKey(0))).get_trace()
    site = trace[f"{host.name}_timing_x"]
    assert tuple(site["fn"].batch_shape) == ()
    assert tuple(site["fn"].event_shape) == (1,)
