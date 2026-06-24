"""Slice-3 tests for per-session timing backends and validators."""

import numpy as np
import pytest

from metapulsar.timing.backends.base import (
    LinearModel,
    validate_backend_shapes,
    validate_backend_zero_delta,
)
from metapulsar.timing.backends.jug import JugTimingBackend, LinearizedJugTimingBackend
from metapulsar.timing.backends.pint import (
    PintTimingBackend,
    LinearizedPintTimingBackend,
)
from metapulsar.timing.backends.tempo2 import (
    Tempo2TimingBackend,
    LinearizedTempo2TimingBackend,
)


def _linear_model():
    fitpars = ("F0", "F1")
    design = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )
    theta_exact = {"F0": "1234.567890123456789", "F1": "-1.0e-15"}
    return LinearModel.from_host(
        fitpars=fitpars, design=design, theta_exact=theta_exact
    )


def _assert_linear_tangent(backend):
    delta = np.array([0.2, -0.5], dtype=float)
    np.testing.assert_allclose(
        backend.residual_delta(delta),
        backend.design_matrix() @ delta,
        atol=1e-12,
    )
    validate_backend_zero_delta(backend)
    validate_backend_shapes(backend)


def test_pint_backend_linear_contract():
    backend = LinearizedPintTimingBackend.from_linear_model(_linear_model())
    _assert_linear_tangent(backend)
    assert backend.fitpars == ("F0", "F1")
    assert set(backend.reference_theta_exact()) == {"F0", "F1"}


def test_tempo2_backend_linear_contract():
    backend = LinearizedTempo2TimingBackend.from_linear_model(_linear_model())
    _assert_linear_tangent(backend)
    assert backend.fitpars == ("F0", "F1")


def test_jug_backend_jax_surface_and_precision_metadata():
    backend = LinearizedJugTimingBackend.from_linear_model(
        _linear_model(),
        compatibility="tempo2",
        precision_critical=frozenset({"F0"}),
    )
    _assert_linear_tangent(backend)
    assert backend.compatibility == "tempo2"
    assert backend.precision_critical_fitpars() == frozenset({"F0"})

    jnp = __import__("jax.numpy", fromlist=["*"])
    delta = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(backend.residual_delta_jax(delta)),
        backend.design_matrix() @ np.asarray(delta),
        atol=1e-12,
    )


def test_native_engine_placeholders_fail_clearly_until_host_wiring():
    for cls in (PintTimingBackend, Tempo2TimingBackend, JugTimingBackend):
        with pytest.raises(NotImplementedError, match="Slice 3b"):
            cls()
