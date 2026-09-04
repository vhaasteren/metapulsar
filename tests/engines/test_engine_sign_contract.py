"""Fitter sign contract: residual_delta(δ) ≈ -M δ."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engine_support import LinearModel
from metapulsar.engines.composite import (
    PtaContribution,
    build_composite_engine,
)
from metapulsar.engines.jug import JugEngine, LinearizedJugEngine
from metapulsar.engines.pint import LinearizedPintEngine, PintEngine
from metapulsar.engines.tempo2 import LibstempoEngine, LinearizedLibstempoEngine
from nltiming.protocols import GaugeProvenance


def _gauge_free() -> GaugeProvenance:
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


def _applied_unknown() -> GaugeProvenance:
    return GaugeProvenance(
        export="applied-unknown",
        reference_mode="unknown",
        reporting_mode="mean",
        reporting_weighted=False,
    )


def _model(fitpars=("F0", "Offset"), n=4):
    design = np.ones((n, len(fitpars)), dtype=float)
    for j in range(len(fitpars)):
        design[:, j] = np.linspace(1.0 + j, 2.0 + j, n)
    return LinearModel.from_design(
        fitpars=fitpars,
        design=design,
        theta_exact={name: "0.0" for name in fitpars},
    )


def _assert_sign_contract(
    engine, *, modulo_constant: bool = False, rtol=1e-10, atol=1e-12
):
    """Δr(δ) ≈ -M δ; on pre-gauged blocks, compare modulo an additive constant."""
    M = np.asarray(engine.design_matrix(), dtype=float)
    delta = np.linspace(0.01, -0.02, len(engine.fitpars))
    got = np.asarray(engine.residual_delta(delta), dtype=float)
    expected = -(M @ delta)
    if modulo_constant:
        # Fit a single additive constant; report it rather than hiding in tol.
        offset = float(np.mean(got - expected))
        np.testing.assert_allclose(got - offset, expected, rtol=rtol, atol=atol)
        assert np.isfinite(offset)
        return offset
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
    return 0.0


def test_linearized_leaf_engines_sign_contract():
    model = _model()
    for eng in (
        LinearizedPintEngine.from_linear_model(model, gauge_provenance=_gauge_free()),
        LinearizedJugEngine.from_linear_model(model, gauge_provenance=_gauge_free()),
        LinearizedLibstempoEngine.from_linear_model(
            model, gauge_provenance=_gauge_free()
        ),
    ):
        _assert_sign_contract(eng)


@pytest.mark.requires_jug
def test_jug_engine_exact_linear_sign_contract():
    pytest.importorskip("jax")
    model = _model(("PB", "Offset"), n=3)
    native = -model.design[:, :1]  # J = -M for the JUG column

    class _State:
        design_matrix = native
        param_mapping = ()

        def residual_delta_jax(self, delta):
            import jax.numpy as jnp

            return jnp.asarray(native) @ jnp.asarray(delta)

        def residual_jacobian_native(self):
            return np.asarray(native, dtype=float)

    engine = JugEngine(state=_State(), linear_model=model)
    engine._jug_indices = (0,)
    engine._jug_fitpars = ("PB",)
    engine._exact_linear_indices = (1,)
    engine._exact_linear_fitpars = frozenset({"Offset"})
    _assert_sign_contract(engine, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(engine.residual_jacobian(), -model.design, atol=1e-12)


def test_composite_sign_contract_per_block():
    a = LinearizedPintEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_epta"),
            design=np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float),
            theta_exact={"F0": "1.0", "Offset_epta": "0.0"},
        ),
        gauge_provenance=_gauge_free(),
    )
    b = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_ppta"),
            design=np.array([[3.0, 1.0], [4.0, 1.0]], dtype=float),
            theta_exact={"F0": "1.0", "Offset_ppta": "0.0"},
        ),
        gauge_provenance=_gauge_free(),
    )
    engine = build_composite_engine(
        fitpars=("F0", "Offset_epta", "Offset_ppta"),
        nrows=4,
        contributions=[
            PtaContribution(name="epta", row_indices=np.array([0, 1]), engine=a),
            PtaContribution(name="ppta", row_indices=np.array([2, 3]), engine=b),
        ],
    )
    M = engine.design_matrix()
    delta = np.array([0.1, -0.2, 0.3], dtype=float)
    got = engine.residual_delta(delta)
    expected = -(M @ delta)
    np.testing.assert_allclose(got[0:2], expected[0:2], atol=1e-12)
    np.testing.assert_allclose(got[2:4], expected[2:4], atol=1e-12)


def test_mixed_gauge_block_compares_modulo_constant():
    """Pre-gauged libstempo block compared modulo an additive constant."""
    model = _model(("F0", "Offset"), n=3)

    class _CenteredDelta:
        """Pretend libstempo: returns -Mδ then subtracts the mean."""

        def delta_residuals(self, delta_params):
            delta = np.array(
                [delta_params.get("F0", 0.0), delta_params.get("Offset", 0.0)],
                dtype=float,
            )
            raw = -(model.design @ delta)
            return raw - np.mean(raw)

    leaf = LibstempoEngine(
        engine=_CenteredDelta(),
        linear_model=model,
        native_fitpars=("F0", "Offset"),
        exact_linear_fitpars=frozenset(),
    )
    assert leaf.gauge_applied is True
    offset = _assert_sign_contract(leaf, modulo_constant=True)
    assert abs(offset) > 0.0


def test_pint_engine_wrapper_sign_with_fake_delta():
    model = _model(("F0", "F1"), n=4)

    class _Fake:
        def delta_residuals(self, delta_params):
            delta = np.array([delta_params["F0"], delta_params["F1"]], dtype=float)
            return -(model.design @ delta)

    eng = PintEngine(engine=_Fake(), linear_model=model)
    _assert_sign_contract(eng)
