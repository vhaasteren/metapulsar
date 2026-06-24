"""Slice-3 tests for host protocol and backend conformance helpers."""

import numpy as np

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

    import pytest

    with pytest.raises(ValueError, match="canonical row order"):
        validate_backend_against_host(BadBackend(), fake_timing_host)
