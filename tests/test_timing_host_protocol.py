"""Slice-3 tests for host protocol and backend conformance helpers."""

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from metapulsar.timing.backends.base import (
    validate_backend_against_host,
    validate_enterprise_host,
)
from metapulsar.timing.protocols import EnterprisePulsarLike, TimingHost


def test_fake_host_satisfies_protocol_and_shape_validators(fake_timing_host):
    assert isinstance(fake_timing_host, EnterprisePulsarLike)
    assert isinstance(fake_timing_host, TimingHost)
    validate_enterprise_host(fake_timing_host)

    backend = fake_timing_host.timing_backend("pint")
    assert tuple(fake_timing_host.fitpars) == backend.fitpars
    validate_backend_against_host(backend, fake_timing_host)


def test_reference_theta_exact_roundtrip(fake_timing_host):
    backend = fake_timing_host.timing_backend("tempo2")
    exact = backend.reference_theta_exact()
    floats = backend.reference_theta()
    for i, name in enumerate(backend.fitpars):
        assert float(exact[name]) == floats[i]


def test_non_metapulsar_host_can_conform():
    class LocalHost:
        name = "LOCAL"
        fitpars = ("F0", "F1")
        _toas = np.array([5.0, 1.0], dtype=float)
        _residuals = np.zeros(2, dtype=float)
        _toaerrs = np.ones(2, dtype=float)
        _freqs = np.ones(2, dtype=float) * 1400.0
        _Mmat = np.zeros((2, 2), dtype=float)
        _flags = {"pta": np.array(["x", "x"])}
        _backend_flags = np.array(["a", "a"])

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
            return self._Mmat

        @property
        def flags(self):
            return self._flags

        @property
        def backend_flags(self):
            return self._backend_flags

        def pint_model(self):
            return None

        def timing_backend(self, name: str = "jug"):
            return fake_timing_host_backend

        def has_timing_backend(self, name: str):
            return True

        def cache_token(self):
            return "local"

    class LocalBackend:
        fitpars = ("F0", "F1")
        native_units = {"F0": "native", "F1": "native"}

        def reference_theta(self):
            return np.zeros(2, dtype=float)

        def reference_theta_exact(self):
            return {"F0": "0.0", "F1": "0.0"}

        def residual_delta(self, delta_theta):
            return np.zeros(2, dtype=float)

        def design_matrix(self, params=None):
            return np.zeros((2, 2), dtype=float)

    fake_timing_host_backend = LocalBackend()
    host = LocalHost()
    assert isinstance(host, TimingHost)
    validate_enterprise_host(host)
    validate_backend_against_host(host.timing_backend(), host)


def test_backend_validator_rejects_design_row_mismatch(fake_timing_host):
    class BadBackend:
        fitpars = tuple(fake_timing_host.fitpars)
        native_units = {name: "native" for name in fitpars}

        def reference_theta(self):
            return np.zeros(len(self.fitpars), dtype=float)

        def reference_theta_exact(self):
            return {name: "0.0" for name in self.fitpars}

        def residual_delta(self, delta_theta):
            return np.zeros(len(fake_timing_host.toas), dtype=float)

        def design_matrix(self, params=None):
            design = np.asarray(fake_timing_host.Mmat, dtype=float).copy()
            design[0, 0] = 1.0
            return design

    with pytest.raises(ValueError, match="canonical row order"):
        validate_backend_against_host(BadBackend(), fake_timing_host)


def _build_real_host():
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_a", seed=11
        ),
        "pta_b": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_b", seed=22
        ),
    }
    return MetaPulsar(pulsars, combination_strategy="composite")


def test_metapulsar_timing_host_surface_and_backend_roundtrip():
    host = _build_real_host()
    assert isinstance(host, TimingHost)
    validate_enterprise_host(host)

    # Native in-memory tempo2 adapters are available for tempo2-origin hosts.
    assert host.has_timing_backend("tempo2")
    assert not host.has_timing_backend("jug")
    assert not host.has_timing_backend("pint")
    assert host.has_timing_backend("tempo2", linearized=True)
    assert host.has_timing_backend("jug", linearized=True)
    assert not host.has_timing_backend("pint", linearized=True)

    native_backend = host.timing_backend("tempo2")
    assert tuple(host.fitpars) == native_backend.fitpars
    validate_backend_against_host(native_backend, host)

    backend = host.timing_backend("tempo2", linearized=True)
    assert tuple(host.fitpars) == backend.fitpars
    validate_backend_against_host(backend, host)

    # cache_token should be stable across repeated reads in unchanged state.
    assert host.cache_token() == host.cache_token()


def test_metapulsar_pint_model_and_backend_error_paths():
    host = _build_real_host()
    model = host.pint_model()
    assert model is not None

    with pytest.raises(ValueError, match="cannot be honored"):
        host.timing_backend("pint", linearized=True)


def test_metapulsar_reference_theta_missing_values_raise():
    host = _build_real_host()
    pta = next(iter(host._epulsars))
    host._parfile_dicts[pta] = {}
    host._invalidate_timing_caches()

    with pytest.raises(ValueError, match="Missing reference theta"):
        host.timing_backend("tempo2", linearized=True)


def test_metapulsar_timing_backend_cache_tracks_host_state():
    host = _build_real_host()
    backend = host.timing_backend("tempo2", linearized=True)
    token = host.cache_token()

    host._designmatrix = host._designmatrix.copy()
    host._designmatrix[0, 0] += 1.0
    changed = host.timing_backend("tempo2", linearized=True)

    assert host.cache_token() != token
    assert changed is not backend
