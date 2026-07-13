"""Slice-5 tests for NonLinearTimingModel component behavior."""

import numpy as np
import pytest

import jax.random as jr
from numpyro import handlers

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.backends.pint import LinearizedPintEngine
from metapulsar.timing.nonlinear_timing_model import NonLinearTimingModel
from metapulsar.timing.whitening import normalized_basis
from metapulsar.timing.sampling.numpyro import contribute_timing
from metapulsar.timing.partition import resolve_partition
from metapulsar.timing.whitening import schur_delta_wls


class _Pulsar:
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
        self._jug_backend = LinearizedJugEngine.from_linear_model(model)
        self._pint_backend = LinearizedPintEngine.from_linear_model(model)

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

    def timing_backend(self, engines="jug", **kwargs):
        self.backend_calls.append((engines, dict(kwargs)))
        if isinstance(engines, dict) and engines.get("pint") == "pint":
            return self._pint_backend
        return self._jug_backend


@pytest.fixture
def host():
    return _Pulsar()


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
    return schur_delta_wls(pulsar=host, partition=part, variance=variance).fisher


def _autodiff_test_matrix(host):
    design = np.asarray(host.Mmat, dtype=float).copy()
    design[:, 0] *= 2.0
    design[:, 1] *= 3.0
    design[:, 2] *= 4.0
    return design


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


def test_component_config_only_build_and_with_engines():
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0"],
        prior_policy="fallback",
        whitening_config={"name": "diagonal_white"},
        name="timing",
    )
    swapped = ntm.with_engines({"tempo2": "jug", "pint": "pint"})

    assert ntm.engines == {"tempo2": "jug", "pint": "jug"}
    assert swapped.engines == {"tempo2": "jug", "pint": "pint"}
    assert swapped.transform == ntm.transform
    assert swapped.analytically_marginalize == ntm.analytically_marginalize
    assert swapped.prior_policy == ntm.prior_policy
    assert swapped.whitening_config == ntm.whitening_config


def test_space_cached_and_invalidated_by_cache_token(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    first = ntm.bind(host).space
    second = ntm.bind(host).space
    assert first is second

    host._cache_token = "token-v2"
    third = ntm.bind(host).space
    assert third is not first


def test_whitening_named_builders_bind_from_serializable_configs(host):
    ntm_default = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
    )
    ntm_fixed = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        whitening_config={
            "name": "fixed_hyperparameters",
            "hyperparameters": {"efac": {"demo": 1.2}, "equad": {"demo": 1.0e-7}},
        },
    )
    sp_default = ntm_default.bind(host).space
    sp_fixed = ntm_fixed.bind(host).space

    assert sp_default.linear.C.shape == (1, 1)
    assert sp_fixed.linear.C.shape == (1, 1)
    assert float(sp_default.linear.C[0, 0]) != float(sp_fixed.linear.C[0, 0])


def test_whitening_builders_condition_fisher_to_unit_scale(host):
    analytically_marginalize_cfg = None
    ntm_default = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=analytically_marginalize_cfg,
    )
    ntm_fixed = NonLinearTimingModel(
        engines="jug",
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
        (ntm_default.bind(host).space, default_fisher),
        (ntm_fixed.bind(host).space, fixed_fisher),
    ):
        fisher_z = _z_space_fisher(space, fisher)
        conditioned = space.linear.C.T @ fisher_z @ space.linear.C
        np.testing.assert_allclose(
            conditioned,
            np.eye(len(space.names)),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_autodiff_design_matrix_method_feeds_whitening(host):
    autodiff_matrix = _autodiff_test_matrix(host)
    host._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    resolved = ntm.bind(host)
    np.testing.assert_allclose(resolved.design_matrix, autodiff_matrix)

    part = resolve_partition(host, analytically_marginalize=["F0", "DM"])
    expected_fisher = schur_delta_wls(
        pulsar=host,
        partition=part,
        variance=np.asarray(host.toaerrs, dtype=float) ** 2,
        design_matrix=autodiff_matrix,
    ).fisher
    fisher_z = _z_space_fisher(resolved.space, expected_fisher)
    conditioned = resolved.space.linear.C.T @ fisher_z @ resolved.space.linear.C
    np.testing.assert_allclose(
        conditioned,
        np.eye(len(resolved.space.names)),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_standardized_builder_uses_z_space_marginal_scales(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )
    space = ntm.bind(host).space
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
        engines="jug",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )
    block = ntm.bind(host).priors
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


def test_autodiff_design_matrix_method_feeds_cheat_prior_widths(host):
    autodiff_matrix = _autodiff_test_matrix(host)
    host._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="standardized",
        analytically_marginalize=None,
        name="timing",
    )

    block = ntm.bind(host).priors
    part = resolve_partition(host, analytically_marginalize=None)
    expected_fisher = schur_delta_wls(
        pulsar=host,
        partition=part,
        variance=np.asarray(host.toaerrs, dtype=float) ** 2,
        design_matrix=autodiff_matrix,
    ).fisher
    expected_stds = np.sqrt(np.diag(_covariance_from_fisher(expected_fisher)))
    half_widths = np.array(
        [(prior.upper - prior.lower) / 2.0 for prior in block.priors], dtype=float
    )
    np.testing.assert_allclose(
        half_widths, ntm.cheat_prior_scale * expected_stds, rtol=1e-6
    )


def test_cheat_prior_box_clipped_to_physical_bounds():
    class _BoundedHost(_Pulsar):
        def __init__(self):
            super().__init__()
            self.fitpars = ("ECC", "F1", "DM")
            self._ecc_ref = 1.0e-5
            model = LinearModel.from_host(
                fitpars=self.fitpars,
                design=self._design,
                theta_exact={"ECC": repr(self._ecc_ref), "F1": "1.0", "DM": "5.0"},
            )
            self._jug_backend = LinearizedJugEngine.from_linear_model(model)

    bounded = _BoundedHost()
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F1", "DM"],
        name="timing",
    )
    block = ntm.bind(bounded).priors
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
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    signals = ntm.bind(host).discovery_signals()
    delay = signals[-1]
    output = np.asarray(delay({f"{host.name}_timing_F1": 0.25}), dtype=float)
    expected = -host._jug_backend.residual_delta(np.array([0.0, 0.25, 0.0]))
    np.testing.assert_allclose(output, expected)

    ntm_nonjax = NonLinearTimingModel(
        engines={"tempo2": "jug", "pint": "pint"},
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    with pytest.raises(ValueError, match="JAX-capable backend"):
        ntm_nonjax.bind(host).discovery_signals()


def test_autodiff_design_matrix_method_feeds_discovery_gp_basis(host, monkeypatch):
    autodiff_matrix = _autodiff_test_matrix(host)
    host._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    captured = {}

    def fake_makegp_improper(psr, basis, *, constant, name):
        captured["basis"] = np.asarray(basis, dtype=float)
        captured["constant"] = constant
        captured["name"] = name
        return "gp"

    monkeypatch.setattr("discovery.signals.makegp_improper", fake_makegp_improper)
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    signals = ntm.bind(host).discovery_signals()

    assert signals[0] == "gp"
    np.testing.assert_allclose(
        captured["basis"], normalized_basis(autodiff_matrix[:, [0, 2]])
    )
    assert not hasattr(host, "iisort")


def test_all_analytically_marginalized_paths(host):
    ntm = NonLinearTimingModel(
        engines={"tempo2": "jug", "pint": "pint"},
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    assert len(ntm.bind(host).discovery_signals()) == 1
    ent = ntm.enterprise_signal()
    bound = ent(host)
    assert hasattr(bound, "get_basis")
    assert ntm.bind(host).timing_param_keys() == ()


def test_non_timing_params_and_timing_param_keys_are_plain_set_subtraction(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    keys = ntm.bind(host).timing_param_keys()
    assert keys[0] == f"{host.name}_timing_x"
    assert keys[1:] == (f"{host.name}_timing_F1",)
    params = ("efac", keys[0], "gamma", keys[1], "log10_A")
    assert ntm.bind(host).non_timing_params(params) == ("efac", "gamma", "log10_A")

    ntm_all_marg = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    plain = ("efac", "gamma", "log10_A")
    assert ntm_all_marg.bind(host).timing_param_keys() == ()
    assert ntm_all_marg.bind(host).non_timing_params(plain) == plain


def test_contribute_timing_samples_joint_site_factors_prior_and_injects_delta(
    host, monkeypatch
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([0.2]))

    out = contribute_timing(ntm.bind(host), {"efac": 1.0})

    assert f"{host.name}_timing_F1" in out
    assert out["efac"] == 1.0
    assert f"{host.name}_timing_x" not in out
    assert calls["sample"][0][0] == f"{host.name}_timing_x"
    assert calls["factor"][0][0] == f"{host.name}_timing_x_logprior"


def test_contribute_timing_noop_when_no_sampled(host, monkeypatch):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "F1", "DM"],
        name="timing",
    )
    calls = _monkeypatch_numpyro(monkeypatch, sample_value=np.array([]))
    params = {"efac": 1.0}
    out = contribute_timing(ntm.bind(host), params)
    assert out is params
    assert calls["sample"] == []
    assert calls["factor"] == []


def test_set_prior_validated_against_sampled_partition(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        prior_override_policy="strict",
        name="timing",
    )
    ntm.set_prior("F0", "uniform", lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="non-sampled"):
        ntm.bind(host).space


def test_set_prior_unknown_name_raises(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        prior_override_policy="strict",
        name="timing",
    )
    ntm.set_prior("F11", "uniform", lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="unknown fit parameters"):
        ntm.bind(host).space


def test_enterprise_signal_forwards_engines(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )
    ent = ntm.enterprise_signal()
    _ = ent(host)
    # The Enterprise frontend consumes the binding, so its backend is built
    # with the model's full timing_backend kwargs (not a bare re-query).
    engines_seen, kwargs_seen = host.backend_calls[-1]
    assert engines_seen == {"tempo2": "jug", "pint": "jug"}
    assert kwargs_seen["design_matrix_method"] == "analytic"
    assert kwargs_seen["subtract_tzr"] is False


def test_autodiff_design_matrix_method_feeds_enterprise_gp_basis(host):
    autodiff_matrix = _autodiff_test_matrix(host)
    host._jug_backend.linearized_design_matrix = lambda params=None: autodiff_matrix
    ntm = NonLinearTimingModel(
        engines="jug",
        design_matrix_method="autodiff",
        transform="none",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    ent = ntm.enterprise_signal()
    bound = ent(host)

    np.testing.assert_allclose(
        bound.get_basis(), normalized_basis(autodiff_matrix[:, [0, 2]])
    )


def test_contribute_timing_improper_uniform_site_has_vector_event_shape(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0", "DM"],
        name="timing",
    )

    def model():
        contribute_timing(ntm.bind(host), {})

    substituted = handlers.substitute(
        model,
        data={f"{host.name}_timing_x": np.array([0.0])},
    )
    trace = handlers.trace(handlers.seed(substituted, jr.PRNGKey(0))).get_trace()
    site = trace[f"{host.name}_timing_x"]
    assert tuple(site["fn"].batch_shape) == ()
    assert tuple(site["fn"].event_shape) == (1,)
