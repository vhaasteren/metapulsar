"""Composite ownership and exact-linear sign (fitter contract)."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engine_support import LinearModel
from metapulsar.engines.composite import (
    PtaContribution,
    PulsarJaxTimingEngine,
    build_composite_engine,
)
from metapulsar.engines.jug import LinearizedJugEngine
from metapulsar.engines.pint import LinearizedPintEngine
from nltiming.protocols import GaugeProvenance


def _gf(**kwargs) -> GaugeProvenance:
    base = dict(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )
    base.update(kwargs)
    return GaugeProvenance(**base)


def test_host_only_exact_linear_uses_negative_design_column():
    """Host-only fallback contributes -M_rows δ, once."""
    contribution = PtaContribution(
        name="pta_a",
        row_indices=np.array([0, 1]),
        engine=LinearizedPintEngine.from_linear_model(
            LinearModel.from_design(
                fitpars=("F0",),
                design=np.array([[1.0], [1.0]], dtype=float),
                theta_exact={"F0": "1.0"},
            ),
            gauge_provenance=_gf(),
        ),
        exact_linear_fitpars=frozenset({"A1"}),
        fallback_reference_exact={"A1": "2.0"},
    )
    design_matrix = np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float)
    delta = np.array([0.0, 3.0], dtype=float)
    engine = build_composite_engine(
        fitpars=("F0", "A1"),
        nrows=2,
        contributions=[contribution],
        design_matrix=design_matrix,
    )
    np.testing.assert_allclose(engine.residual_delta(delta), [-30.0, -33.0])
    np.testing.assert_allclose(engine.design_matrix()[:, 1], [10.0, 11.0])


def test_leaf_owned_exact_linear_not_double_counted():
    """Contribution exact-linear that the leaf already owns is not re-added."""
    model = LinearModel.from_design(
        fitpars=("PB", "JUMP"),
        design=np.array([[2.0, 10.0], [3.0, 11.0]], dtype=float),
        theta_exact={"PB": "1.0", "JUMP": "0.0"},
    )
    leaf = LinearizedJugEngine.from_linear_model(model, gauge_provenance=_gf())
    engine = build_composite_engine(
        fitpars=model.fitpars,
        nrows=2,
        contributions=[
            PtaContribution(
                name="pta",
                row_indices=np.array([0, 1]),
                engine=leaf,
                exact_linear_fitpars=frozenset({"JUMP"}),
            )
        ],
        design_matrix=model.design,
    )
    delta = np.array([0.5, -0.25], dtype=float)
    expected = -(model.design @ delta)
    np.testing.assert_allclose(engine.residual_delta(delta), expected)


def test_numpy_and_jax_host_only_fallback_errors_match():
    contribution = PtaContribution(
        name="pta",
        row_indices=np.array([0, 1]),
        engine=LinearizedJugEngine.from_linear_model(
            LinearModel.from_design(
                fitpars=("F0",),
                design=np.array([[1.0], [1.0]], dtype=float),
                theta_exact={"F0": "1.0"},
            ),
            gauge_provenance=_gf(),
        ),
        exact_linear_fitpars=frozenset({"A1"}),
        fallback_reference_exact={"A1": "0.0"},
    )
    engine = build_composite_engine(
        fitpars=("F0", "A1"),
        nrows=2,
        contributions=[contribution],
        design_matrix=None,
    )
    assert isinstance(engine, PulsarJaxTimingEngine)
    delta = np.array([0.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="no pulsar design matrix"):
        engine.residual_delta(delta)
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="no pulsar design matrix"):
        engine.residual_delta_jax(jnp.asarray(delta))


def test_pulsar_jax_engine_exposes_residual_jacobian():
    pytest.importorskip("jax")
    leaf = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset"),
            design=np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float),
            theta_exact={"F0": "1.0", "Offset": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    engine = build_composite_engine(
        fitpars=("F0", "Offset"),
        nrows=2,
        contributions=[
            PtaContribution(name="pta", row_indices=np.array([0, 1]), engine=leaf)
        ],
    )
    assert isinstance(engine, PulsarJaxTimingEngine)
    J = engine.residual_jacobian()
    np.testing.assert_allclose(J, -engine.design_matrix())
