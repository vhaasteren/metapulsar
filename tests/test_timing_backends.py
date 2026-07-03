"""Slice-3 tests for per-session timing backends and validators."""

import numpy as np
import pytest

from metapulsar.timing.backends.base import (
    LinearModel,
    validate_backend_shapes,
    validate_backend_zero_delta,
    zero_delta_tolerance,
)
from metapulsar.timing.backends.jug import (
    JugEngine,
    LinearizedJugEngine,
    _is_exact_linear_param,
)
from metapulsar.timing.backends.pint import (
    PintEngine,
    LinearizedPintEngine,
)
from metapulsar.timing.backends.tempo2 import (
    LibstempoEngine,
    LinearizedLibstempoEngine,
)


class _FakeDeltaEngine:
    def delta_residuals(self, delta_params):
        delta = np.array([delta_params["F0"], delta_params["F1"]], dtype=float)
        return _linear_model().design @ delta


class _FakeJaxState:
    def residual_delta_np(self, delta):
        return _linear_model().design @ np.asarray(delta, dtype=float)

    def residual_delta_jax(self, delta):
        import jax.numpy as jnp

        return jnp.asarray(_linear_model().design) @ jnp.asarray(delta)


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
    backend = LinearizedPintEngine.from_linear_model(_linear_model())
    _assert_linear_tangent(backend)
    assert backend.fitpars == ("F0", "F1")
    assert set(backend.reference_theta_exact()) == {"F0", "F1"}


def test_tempo2_backend_linear_contract():
    backend = LinearizedLibstempoEngine.from_linear_model(_linear_model())
    _assert_linear_tangent(backend)
    assert backend.fitpars == ("F0", "F1")


def test_jug_backend_jax_surface_and_precision_metadata():
    backend = LinearizedJugEngine.from_linear_model(
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


def test_native_backends_wrap_engines_with_host_metadata():
    model = _linear_model()
    backends = [
        PintEngine(engine=_FakeDeltaEngine(), linear_model=model),
        LibstempoEngine(engine=_FakeDeltaEngine(), linear_model=model),
        JugEngine(state=_FakeJaxState(), linear_model=model),
    ]
    for backend in backends:
        _assert_linear_tangent(backend)
        assert backend.reference_theta_exact()["F0"] == "1234.567890123456789"


def test_jug_backend_adds_exact_linear_to_numpy_and_jax_paths():
    model = LinearModel.from_host(
        fitpars=("PB", "Offset"),
        design=np.array(
            [
                [2.0, 1.0],
                [3.0, 1.0],
                [5.0, 1.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "Offset": "0.0"},
    )
    state = _FakeJaxState()
    state.residual_delta_np = lambda delta: model.design[:, :1] @ np.asarray(
        delta, dtype=float
    )

    def residual_delta_jax(delta):
        import jax.numpy as jnp

        return jnp.asarray(model.design[:, :1]) @ jnp.asarray(delta)

    state.residual_delta_jax = residual_delta_jax
    backend = JugEngine(state=state, linear_model=model)
    backend._jug_indices = (0,)
    backend._jug_fitpars = ("PB",)
    backend._exact_linear_indices = (1,)
    backend._exact_linear_fitpars = frozenset({"Offset"})

    delta = np.array([0.5, -0.25], dtype=float)
    expected = model.design @ delta
    np.testing.assert_allclose(backend.residual_delta(delta), expected)

    jnp = __import__("jax.numpy", fromlist=["*"])
    np.testing.assert_allclose(
        np.asarray(backend.residual_delta_jax(jnp.asarray(delta))),
        expected,
    )


def test_jug_exact_linear_does_not_capture_spin_frequency_params():
    assert not _is_exact_linear_param("F0")
    assert not _is_exact_linear_param("F1")
    assert not _is_exact_linear_param("F12")
    assert _is_exact_linear_param("Offset")
    assert _is_exact_linear_param("DMX_0001")
    assert _is_exact_linear_param("JUMP1")


class _OffsetZeroDeltaBackend:
    backend_name = "jug"
    fitpars = ("F0",)

    def __init__(self, *, compatibility: str, offset_sec: float):
        self.compatibility = compatibility
        self._offset_sec = float(offset_sec)

    def residual_delta(self, delta_theta):
        return np.full(3, self._offset_sec, dtype=float)


def test_zero_delta_tolerance_relaxed_for_jug_tempo2_only():
    backend = _OffsetZeroDeltaBackend(compatibility="tempo2", offset_sec=2.7e-8)
    assert zero_delta_tolerance(backend, 1e-9) == 1e-7
    with pytest.warns(UserWarning, match="tempo2"):
        validate_backend_zero_delta(backend, tol=1e-9)

    strict = _OffsetZeroDeltaBackend(compatibility="pint", offset_sec=2.7e-8)
    assert zero_delta_tolerance(strict, 1e-9) == 1e-9
    with pytest.raises(ValueError, match="residual_delta\\(0\\)"):
        validate_backend_zero_delta(strict, tol=1e-9)


def test_zero_delta_tolerance_still_fails_large_jug_tempo2_offset():
    backend = _OffsetZeroDeltaBackend(compatibility="tempo2", offset_sec=1e-3)
    with pytest.raises(ValueError, match="residual_delta\\(0\\)"):
        validate_backend_zero_delta(backend, tol=1e-9)
