"""Slice-5 tests for multi-host timing component binding."""

import numpy as np

from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugTimingBackend
from metapulsar.timing.component import NonLinearTimingModel


class _Host:
    def __init__(self, name: str, token: str):
        self.name = name
        self.fitpars = ("F0", "F1")
        self._toas = np.linspace(0.0, 1.0, 4)
        self._residuals = np.zeros(4)
        self._toaerrs = np.full(4, 1.0e-6)
        self._freqs = np.full(4, 1400.0)
        self._flags = {"pta": np.array(["demo"] * 4, dtype="U8")}
        self._backend_flags = np.array(["demo"] * 4, dtype="U8")
        self._cache_token = token
        model = LinearModel.from_host(
            fitpars=self.fitpars,
            design=np.column_stack([np.ones(4), np.linspace(-0.5, 0.5, 4)]),
            theta_exact={"F0": "100.0", "F1": "1.0"},
        )
        self._backend = LinearizedJugTimingBackend.from_linear_model(model)

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

    def timing_backend(self, name: str, **kwargs):
        if name != "jug":
            raise ValueError(name)
        return self._backend


def test_multi_host_prefixes_and_cache_independence(monkeypatch):
    host_a = _Host("J0001+0001", "tok-a")
    host_b = _Host("J0002+0002", "tok-b")
    ntm = NonLinearTimingModel(
        backend="jug",
        transform="none",
        analytically_marginalize=["F0"],
        name="timing",
    )

    sample_values = {
        f"{host_a.name}_timing_delta": np.array([0.1]),
        f"{host_b.name}_timing_delta": np.array([-0.2]),
    }
    deterministic_calls = []

    monkeypatch.setattr("numpyro.factor", lambda *args, **kwargs: None)
    monkeypatch.setattr("numpyro.sample", lambda name, dist: sample_values[name])
    monkeypatch.setattr(
        "numpyro.deterministic",
        lambda name, value: deterministic_calls.append((name, value)),
    )

    params_a = ntm.contribute_timing(host_a, {"efac": 1.0})
    params_b = ntm.contribute_timing(host_b, {"efac": 1.0})

    assert f"{host_a.name}_timing_F1" in params_a
    assert f"{host_b.name}_timing_F1" not in params_a
    assert f"{host_b.name}_timing_F1" in params_b
    assert f"{host_a.name}_timing_F1" not in params_b

    space_a_1 = ntm.space(host_a)
    space_b_1 = ntm.space(host_b)
    assert space_a_1 is not space_b_1
    assert ntm.space(host_a) is space_a_1
    assert ntm.space(host_b) is space_b_1

    host_a._cache_token = "tok-a-updated"
    assert ntm.space(host_a) is not space_a_1
    assert ntm.space(host_b) is space_b_1

    ntm.record_physical(host_a, params_a, scope="timing")
    ntm.record_physical(host_b, params_b, scope="timing")
    deterministic_names = {name for name, _ in deterministic_calls}
    assert f"{host_a.name}_timing_F1_theta" in deterministic_names
    assert f"{host_b.name}_timing_F1_theta" in deterministic_names
