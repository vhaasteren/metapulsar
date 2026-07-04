"""Slice-3 tests for JUG/JAX backend behavior."""

import numpy as np
import pytest

from jug.io.par_reader import get_longdouble
from metapulsar.timing.backends.base import LinearModel
from metapulsar.timing.backends.jug import LinearizedJugEngine
from metapulsar.timing.backends import jug_jax_state
from metapulsar.timing.backends.jug_jax_state import JaxTimingState

pytestmark = [pytest.mark.requires_jug]


def _jug_backend(compatibility: str) -> LinearizedJugEngine:
    return LinearizedJugEngine.from_linear_model(
        LinearModel.from_host(
            fitpars=("F0", "F1"),
            design=np.array(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                ],
                dtype=float,
            ),
            theta_exact={"F0": "1000.0", "F1": "1e-15"},
        ),
        compatibility=compatibility,
        precision_critical=frozenset({"F0"}),
    )


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
def test_jug_residual_delta_jax_matches_numpy_path(compatibility):
    jnp = pytest.importorskip("jax.numpy")
    backend = _jug_backend(compatibility)
    delta = np.array([0.2, -0.1], dtype=float)
    np.testing.assert_allclose(
        np.asarray(backend.residual_delta_jax(jnp.asarray(delta))),
        backend.residual_delta(delta),
        atol=1e-12,
    )


def test_jug_tangent_near_zero_matches_design_matrix():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    backend = _jug_backend("pint")

    def fn(x):
        return backend.residual_delta_jax(x)

    jac = jax.jacfwd(fn)(jnp.zeros((2,), dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(jac), backend.design_matrix(), atol=1e-12)


def test_jug_precision_critical_fitpars_exposed_with_canonical_names():
    backend = _jug_backend("tempo2")
    assert backend.precision_critical_fitpars() == frozenset({"F0"})


def test_jax_timing_state_residual_delta_np_preserves_high_precision_f0(monkeypatch):
    base_f0 = np.longdouble("326.60056708749672367")
    ref_params = {
        "F0": float(base_f0),
        "_high_precision": {"F0": "326.60056708749672367"},
    }

    def compute_residuals(params, setup):
        f0 = get_longdouble(params, "F0")
        return (
            np.array([float((f0 - base_f0) * np.longdouble("1e6"))]),
            None,
            None,
            None,
        )

    monkeypatch.setattr(
        jug_jax_state, "_compute_full_model_residuals", compute_residuals
    )
    state = JaxTimingState(
        fit_params=("F0",),
        param_mapping=(),
        ref_params=ref_params,
        ref_theta=np.array([float(base_f0)]),
        reference_residuals_sec=np.array([0.0]),
        subtract_tzr=True,
        compatibility="pint",
        phase_mean_mode="weighted",
        isort=None,
        design_matrix=np.empty((1, 1)),
        column_units=("Hz",),
        setup=object(),
        _residual_delta_jax_fn=lambda delta: delta,
    )

    with pytest.deprecated_call():
        np.testing.assert_allclose(
            state.residual_delta_np(np.zeros(1)), [0.0], atol=1e-18
        )
