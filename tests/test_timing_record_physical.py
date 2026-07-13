"""Slice-5 tests for record_physical timing deterministic output."""

import numpy as np
import pytest

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.nonlinear_timing_model import NonLinearTimingModel
from metapulsar.timing.sampling.numpyro import contribute_timing, record_physical


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
        self._cache_token = "record-token"
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


def _patch_numpyro(monkeypatch, sample_value):
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


def test_record_physical_timing_scope_emits_prefixed_theta_sites(host, monkeypatch):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    calls = _patch_numpyro(monkeypatch, sample_value=np.array([0.25]))
    params = contribute_timing(ntm.bind(host), {})

    record_physical(ntm.bind(host), params, scope="timing")

    det_names = [name for name, _ in calls["deterministic"]]
    assert f"{host.name}_timing_F1_theta_native" in det_names
    assert f"{host.name}_timing_F1_theta_display" in det_names
    expected_theta = ntm.bind(host).space.theta_from_delta(
        np.array([params[f"{host.name}_timing_F1"]])
    )
    native_call = next(
        v for n, v in calls["deterministic"] if n.endswith("_theta_native")
    )
    np.testing.assert_allclose(native_call, expected_theta[0])


def test_record_physical_scope_all_raises(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    with pytest.raises(NotImplementedError, match="scope='all'"):
        record_physical(ntm.bind(host), {}, scope="all")


def test_record_physical_does_not_change_density_calls(host, monkeypatch):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    calls = _patch_numpyro(monkeypatch, sample_value=np.array([0.1]))
    params = contribute_timing(ntm.bind(host), {})
    n_sample = len(calls["sample"])
    n_factor = len(calls["factor"])

    record_physical(ntm.bind(host), params, scope="timing")

    assert len(calls["sample"]) == n_sample
    assert len(calls["factor"]) == n_factor
    assert len(calls["deterministic"]) == 2


def test_record_physical_explicit_coord_handles_standardized_scalar_params(
    host, monkeypatch
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0"],
        name="timing",
    )
    calls = _patch_numpyro(monkeypatch, sample_value=np.array([0.0]))
    x_value = 0.25
    params = {f"{host.name}_timing_F1": x_value}

    record_physical(ntm.bind(host), params, scope="timing", coord="x")

    space = ntm.bind(host).space
    delta = space.delta_from_coord(np.array([x_value], dtype=float), np, coord="x")
    expected_theta = space.theta_from_delta(delta)
    det_names = [name for name, _ in calls["deterministic"]]
    assert f"{host.name}_timing_F1_theta_native" in det_names
    assert f"{host.name}_timing_F1_theta_display" in det_names
    native_call = next(
        v for n, v in calls["deterministic"] if n.endswith("_theta_native")
    )
    np.testing.assert_allclose(native_call, expected_theta[0])


def test_record_physical_implicit_coord_handles_standardized_contribute_output(
    host, monkeypatch
):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0"],
        name="timing",
    )
    calls = _patch_numpyro(monkeypatch, sample_value=np.array([0.2]))
    params = contribute_timing(ntm.bind(host), {})

    record_physical(ntm.bind(host), params, scope="timing")

    # contribute_timing injects backend-facing delta keys; implicit standardized
    # record_physical should interpret those as delta, not as standardized x.
    expected_theta = ntm.bind(host).space.theta_from_delta(
        np.array([params[f"{host.name}_timing_F1"]], dtype=float)
    )
    det_names = [name for name, _ in calls["deterministic"]]
    assert f"{host.name}_timing_F1_theta_native" in det_names
    assert f"{host.name}_timing_F1_theta_display" in det_names
    native_call = next(
        v for n, v in calls["deterministic"] if n.endswith("_theta_native")
    )
    np.testing.assert_allclose(native_call, expected_theta[0])


def test_record_physical_invalid_coord_raises(host):
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )
    with pytest.raises(ValueError, match="coord must be one of"):
        record_physical(ntm.bind(host), {}, scope="timing", coord="invalid")
