"""Tests for the sampler glue in metapulsar.timing.sampling."""

import numpy as np
import pytest
from numpyro import handlers

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.nonlinear_timing_model import NonLinearTimingModel
from metapulsar.timing.sampling import numpyro as nlt_numpyro
from metapulsar.timing.sampling import ptmcmc as nlt_ptmcmc


class _Host:
    def __init__(self):
        self.name = "J1111+1111"
        self.fitpars = ("F0", "F1")
        self._toas = np.linspace(0.0, 1.0, 5)
        self._residuals = np.zeros(5)
        self._toaerrs = np.full(5, 1.0e-6)
        self._freqs = np.full(5, 1400.0)
        self._flags = {"pta": np.array(["demo"] * 5, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 5, dtype="U8")
        self._cache_token = "sampling-token"
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=np.column_stack([np.ones(5), np.linspace(-0.5, 0.5, 5)]),
            theta_exact={"F0": "100.0", "F1": "1.0"},
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

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
        return self._backend.design_matrix()

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
        return self._backend


@pytest.fixture
def host():
    return _Host()


def _binding(transform="whitening", **kwargs):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform=transform,
        analytically_marginalize=["F0"],
        name="timing",
        **kwargs,
    )
    return ntm


class _FakeLogL:
    def __init__(self, params):
        self.params = list(params)

    def __call__(self, params):
        return -0.5 * float(sum(float(v) ** 2 for v in params.values()))


class _FakeLikelihood:
    def __init__(self, params):
        self.logL = _FakeLogL(params)


# ---------------------------------------------------------------------------
# numpyro glue


def _trace_model(model_fn, binding):
    """Trace a model with the improper timing site pinned at the reference."""
    import jax.numpy as jnp
    import jax.random as jr

    init = {binding.coord_site_name(): jnp.zeros(len(binding.sampled))}
    with handlers.seed(rng_seed=jr.PRNGKey(0)):
        with handlers.substitute(data=init):
            return handlers.trace(model_fn).get_trace()


def test_model_traces_timing_site_and_ll_factor(host):
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood(
        [*binding.timing_param_keys(), "efac"],
    )
    model_fn = nlt_numpyro.model(likelihood, binding, fixed={"efac": 1.0})

    trace = _trace_model(model_fn, binding)

    assert binding.coord_site_name() in trace
    assert f"{binding.coord_site_name()}_logprior" in trace
    assert "ll" in trace
    # efac is fixed, not sampled
    assert "efac" not in trace


def test_model_free_params_use_priordict_bounds(host):
    pytest.importorskip("discovery")
    binding = _binding().bind(host)
    likelihood = _FakeLikelihood(
        [*binding.timing_param_keys(), "J1111+1111_efac"],
    )
    model_fn = nlt_numpyro.model(
        likelihood,
        binding,
        priors={"J1111+1111_efac": [0.5, 1.5]},
    )

    trace = _trace_model(model_fn, binding)

    assert "J1111+1111_efac" in trace
    value = float(trace["J1111+1111_efac"]["value"])
    assert 0.5 <= value <= 1.5


def test_timing_init_values_zero_at_reference(host):
    binding = _binding().bind(host)
    init = nlt_numpyro.timing_init_values(binding)
    assert set(init) == {binding.coord_site_name()}
    np.testing.assert_array_equal(
        np.asarray(init[binding.coord_site_name()]),
        np.zeros(len(binding.sampled)),
    )


def test_timing_draws_flattens_chains(host):
    binding = _binding().bind(host)
    site = binding.coord_site_name()
    ndim = len(binding.sampled)
    flat = nlt_numpyro.timing_draws({site: np.zeros((7, ndim))}, binding)
    assert flat.shape == (7, ndim)
    stacked = nlt_numpyro.timing_draws({site: np.zeros((2, 7, ndim))}, binding)
    assert stacked.shape == (14, ndim)


def test_ensure_x64_enables_float64():
    nlt_numpyro.ensure_x64()
    import jax.numpy as jnp

    assert jnp.zeros(1).dtype == jnp.float64


# ---------------------------------------------------------------------------
# ptmcmc glue


def test_eval_params_whitening_uses_joint_site(host):
    binding = _binding(transform="whitening").bind(host)
    vec = np.array([0.25])
    params = nlt_ptmcmc.eval_params(binding, vec, fixed={"efac": 1.0})
    assert params["efac"] == 1.0
    np.testing.assert_array_equal(params[binding.coord_site_name()], vec)


def test_eval_params_standardized_uses_scalar_delay_keys(host):
    binding = _binding(transform="standardized").bind(host)
    vec = np.array([0.25])
    params = nlt_ptmcmc.eval_params(binding, vec)
    assert params == {binding.delay_keys[0]: 0.25}


def test_eval_params_rejects_wrong_length(host):
    binding = _binding().bind(host)
    with pytest.raises(ValueError, match="expected vector of length 1"):
        nlt_ptmcmc.eval_params(binding, np.zeros(3))


def test_initial_point_is_zero_reference(host):
    binding = _binding().bind(host)
    np.testing.assert_array_equal(
        nlt_ptmcmc.initial_point(binding), np.zeros(len(binding.sampled))
    )


def test_initial_cov_matches_wls_in_sampling_coords(host):
    binding = _binding(transform="whitening").bind(host)
    cov = nlt_ptmcmc.initial_cov(binding, nsamples=4000, seed=1)
    assert cov.shape == (1, 1)
    # positive definite
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    # whitening scales the WLS posterior to roughly unit coordinates, so the
    # sampled-coordinate variance must be O(1), not the raw delta variance
    assert 0.1 < float(cov[0, 0]) < 10.0


def test_timing_param_names_layouts(host):
    whitening = _binding(transform="whitening").bind(host)
    site = whitening.coord_site_name()
    assert nlt_ptmcmc.timing_param_names(whitening) == (f"{site}_0",)

    standardized = _binding(transform="standardized").bind(host)
    assert nlt_ptmcmc.timing_param_names(standardized) == standardized.delay_keys


def test_chain_layout_locates_timing_columns(host):
    binding = _binding(transform="standardized").bind(host)
    names = ["noise_param", *nlt_ptmcmc.timing_param_names(binding)]
    layout = nlt_ptmcmc.chain_layout(binding, names)
    assert layout == {"kind": "ptmcmc", "file": "chain_1.txt", "columns": [1]}


def test_chain_layout_missing_key_raises(host):
    binding = _binding().bind(host)
    with pytest.raises(ValueError, match="not found in sampler param names"):
        nlt_ptmcmc.chain_layout(binding, ["something_else"])
