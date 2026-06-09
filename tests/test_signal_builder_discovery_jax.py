"""Tests for Discovery builder JAX path selection."""

import sys
import types

import numpy as np
import pytest

from metapulsar.nonlinear_timing_model import (
    JugDeltaEngine,
    build_discovery_nonlinear_timing_components,
)


class _DiscoveryDummyPulsar:
    def __init__(self):
        self.name = "J0000+0000"
        self.residuals = np.zeros(4, dtype=float)
        self.Mmat = np.eye(4)


class _JaxDummyEngine:
    fitpars = ["F0", "F1"]
    sampled_params = ("F0",)
    output_shape = (4,)
    output_dtype = float
    _reference_residuals = np.zeros(4, dtype=float)

    def timing_delay_jax(self, z_flat):
        import jax.numpy as jnp

        z_flat = jnp.asarray(z_flat, dtype=jnp.float64).reshape(-1)
        return jnp.zeros((4,), dtype=jnp.float64) * z_flat[0]

    def timing_delay_np(self, z_flat):
        return np.zeros(4, dtype=float)


def _install_fake_discovery(monkeypatch):
    fake_discovery = types.ModuleType("discovery")
    fake_signals = types.ModuleType("discovery.signals")
    fake_likelihood = types.ModuleType("discovery.likelihood")

    class _FakeTimingGP:
        def __init__(self, psr, fmat, constant, name, variable):
            del psr, fmat, constant, name, variable

    def _makegp_improper(psr, fmat, constant=1.0e40, name="improperGP", variable=False):
        return _FakeTimingGP(psr, fmat, constant, name, variable)

    fake_signals.makegp_improper = _makegp_improper
    fake_likelihood.PulsarLikelihood = lambda signals: signals
    fake_discovery.signals = fake_signals
    fake_discovery.likelihood = fake_likelihood
    monkeypatch.setitem(sys.modules, "discovery", fake_discovery)
    monkeypatch.setitem(sys.modules, "discovery.signals", fake_signals)
    monkeypatch.setitem(sys.modules, "discovery.likelihood", fake_likelihood)


def test_discovery_builder_selects_native_jax_path(monkeypatch):
    _install_fake_discovery(monkeypatch)
    psr = _DiscoveryDummyPulsar()
    engine = _JaxDummyEngine()
    components = build_discovery_nonlinear_timing_components(
        psr=psr,
        engine=engine,
        sampled_params=["F0"],
        mode="nmat",
    )
    params = {components.sampled_parameter_names["F0"]: 0.0}
    np.testing.assert_allclose(components.delay(params), np.zeros(4))


def test_discovery_builder_rejects_host_jug_delta_engine(monkeypatch):
    _install_fake_discovery(monkeypatch)

    class _Session:
        params = {"F0": 1.0, "F1": 0.0}

        def compute_residuals(self, params=None, subtract_tzr=True):
            return {"residuals_sec": np.zeros(4)}

    psr = _DiscoveryDummyPulsar()
    engine = JugDeltaEngine(_Session(), fitpars=["F0", "F1"])
    with pytest.raises(TypeError, match="JugDeltaEngine"):
        build_discovery_nonlinear_timing_components(
            psr=psr,
            engine=engine,
            sampled_params=["F0"],
            mode="nmat",
        )
