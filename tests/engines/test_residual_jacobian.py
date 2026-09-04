"""residual_jacobian contract and route agreement."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engine_support import LinearModel
from metapulsar.engines.jug import JugEngine, LinearizedJugEngine
from nltiming.protocols import GaugeProvenance


def _gf():
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


def test_linearized_jug_residual_jacobian_is_minus_m():
    model = LinearModel.from_design(
        fitpars=("F0", "Offset"),
        design=np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=float),
        theta_exact={"F0": "1.0", "Offset": "0.0"},
    )
    eng = LinearizedJugEngine.from_linear_model(model, gauge_provenance=_gf())
    np.testing.assert_allclose(eng.residual_jacobian(), -model.design)


@pytest.mark.requires_jug
def test_jug_engine_residual_jacobian_matches_jacfwd():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    model = LinearModel.from_design(
        fitpars=("PB", "Offset"),
        design=np.array([[2.0, 1.0], [3.0, 1.0], [5.0, 1.0]], dtype=float),
        theta_exact={"PB": "1.0", "Offset": "0.0"},
    )
    native_J = -model.design[:, :1]

    class _State:
        design_matrix = native_J
        param_mapping = ()

        def residual_delta_jax(self, delta):
            return jnp.asarray(native_J) @ jnp.asarray(delta)

        def residual_jacobian_native(self):
            return np.asarray(native_J, dtype=float)

    engine = JugEngine(state=_State(), linear_model=model)
    engine._jug_indices = (0,)
    engine._jug_fitpars = ("PB",)
    engine._exact_linear_indices = (1,)
    engine._exact_linear_fitpars = frozenset({"Offset"})

    J = engine.residual_jacobian()
    zeros = jnp.zeros(len(engine.fitpars))
    J_fwd = np.asarray(jax.jacfwd(engine.residual_delta_jax)(zeros), dtype=float)
    np.testing.assert_allclose(J, J_fwd, atol=1e-12)
    # Route agreement: -J (autodiff) equals analytic design_matrix (= M).
    np.testing.assert_allclose(-J, engine.design_matrix(), atol=1e-12)


@pytest.mark.requires_jug
def test_ecliptic_zero_column_guard():
    pytest.importorskip("jax")
    model = LinearModel.from_design(
        fitpars=("ELONG", "F0"),
        design=np.array([[1.0, 0.5], [1.0, 1.0]], dtype=float),
        theta_exact={"ELONG": "0.0", "F0": "1.0"},
    )

    class _State:
        design_matrix = np.zeros((2, 2), dtype=float)
        param_mapping = ()

        def residual_delta_jax(self, delta):
            import jax.numpy as jnp

            return jnp.zeros((2,), dtype=jnp.asarray(delta).dtype)

        def residual_jacobian_native(self):
            return np.zeros((2, 2), dtype=float)

    engine = JugEngine(state=_State(), linear_model=model)
    with pytest.raises(ValueError, match="ecliptic|ELONG|derivative_method"):
        engine.residual_jacobian()
