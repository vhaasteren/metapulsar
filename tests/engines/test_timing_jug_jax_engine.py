"""Slice-3 tests for JUG/JAX engine behavior."""

import numpy as np
import pytest

from nltiming.engine_support import LinearModel
from metapulsar.engines.jug import LinearizedJugEngine


def _jug_backend(compatibility: str) -> LinearizedJugEngine:
    return LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
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
    engine = _jug_backend(compatibility)
    delta = np.array([0.2, -0.1], dtype=float)
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(jnp.asarray(delta))),
        engine.residual_delta(delta),
        atol=1e-12,
    )


def test_jug_tangent_near_zero_matches_design_matrix():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    engine = _jug_backend("pint")

    def fn(x):
        return engine.residual_delta_jax(x)

    jac = jax.jacfwd(fn)(jnp.zeros((2,), dtype=jnp.float64))
    # residual_delta ≈ -M δ, so autodiff Jacobian equals -design_matrix.
    np.testing.assert_allclose(np.asarray(jac), -engine.design_matrix(), atol=1e-12)


def test_jug_precision_critical_fitpars_exposed_with_canonical_names():
    engine = _jug_backend("tempo2")
    assert engine.precision_critical_fitpars() == frozenset({"F0"})
