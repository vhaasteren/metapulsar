"""Slice-3 tests for composite timing engine behavior."""

import numpy as np
import pytest

from nltiming.engine_support import LinearModel
from metapulsar.engines.composite import (
    PtaContribution,
    PulsarJaxTimingEngine,
    PulsarTimingEngine,
    build_composite_engine,
)
from metapulsar.engines import build_engine
from metapulsar.engines.jug import JugEngine, LinearizedJugEngine
from metapulsar.engines.pint import LinearizedPintEngine


def _session_backends():
    # Host canonical rows are unsorted by time on purpose.
    # session_a rows -> [2, 0], session_b rows -> [3, 1]
    a = LinearizedPintEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "A1"),
            design=np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float),
            theta_exact={"F0": "10.0", "A1": "1.0"},
        )
    )
    b = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "PB"),
            design=np.array([[1.0, 20.0], [1.0, 21.0]], dtype=float),
            theta_exact={"F0": "10.0", "PB": "5.0"},
        ),
        precision_critical=frozenset({"F0"}),
    )
    return [
        PtaContribution(name="pta_a", row_indices=np.array([2, 0]), engine=a),
        PtaContribution(name="pta_b", row_indices=np.array([3, 1]), engine=b),
    ]


def test_composite_residual_and_design_scatter_in_canonical_rows():
    contributions = _session_backends()
    engine = build_composite_engine(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        contributions=contributions,
    )
    assert isinstance(engine, PulsarTimingEngine)
    delta = np.array([0.5, 2.0, -1.0], dtype=float)

    residual = engine.residual_delta(delta)
    expected = np.zeros(4, dtype=float)
    # Fitter sign: Δr ≈ -M δ
    expected[[2, 0]] = -(np.array([[1.0, 10.0], [1.0, 11.0]]) @ np.array([0.5, 2.0]))
    expected[[3, 1]] = -(np.array([[1.0, 20.0], [1.0, 21.0]]) @ np.array([0.5, -1.0]))
    np.testing.assert_allclose(residual, expected)

    design = engine.design_matrix()
    assert design.shape == (4, 3)
    np.testing.assert_allclose(
        design[[2, 0], :2],
        np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float),
    )
    np.testing.assert_allclose(
        design[[3, 1], ::2],
        np.array([[1.0, 20.0], [1.0, 21.0]], dtype=float),
    )


def test_composite_reference_theta_exact_validates_shared_values():
    contributions = _session_backends()
    engine = build_composite_engine(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        contributions=contributions,
    )
    ref = engine.reference_theta_exact()
    assert ref["F0"] == "10.0"
    assert ref["A1"] == "1.0"
    assert ref["PB"] == "5.0"

    bad_contributions = _session_backends()
    bad_backend = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0",),
            design=np.array([[1.0], [1.0]], dtype=float),
            theta_exact={"F0": "11.0"},
        )
    )
    bad_contributions[1] = PtaContribution(
        name="pta_b",
        row_indices=np.array([3, 1]),
        engine=bad_backend,
    )
    with pytest.raises(ValueError, match="disagrees"):
        build_composite_engine(
            fitpars=("F0", "A1", "PB"),
            nrows=4,
            contributions=bad_contributions,
        )


def test_composite_jax_capability_requires_all_contributions_and_unions_precision():
    contributions = _session_backends()
    engine = build_composite_engine(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        contributions=contributions,
    )
    assert not isinstance(engine, PulsarJaxTimingEngine)

    only_jax_contributions = [
        PtaContribution(
            name="s1",
            row_indices=np.array([0, 1]),
            engine=LinearizedJugEngine.from_linear_model(
                LinearModel.from_design(
                    fitpars=("F0",),
                    design=np.array([[1.0], [1.0]], dtype=float),
                    theta_exact={"F0": "1.0"},
                ),
                precision_critical=frozenset({"F0"}),
            ),
        ),
        PtaContribution(
            name="s2",
            row_indices=np.array([2, 3]),
            engine=LinearizedJugEngine.from_linear_model(
                LinearModel.from_design(
                    fitpars=("PB",),
                    design=np.array([[2.0], [3.0]], dtype=float),
                    theta_exact={"PB": "2.0"},
                ),
                precision_critical=frozenset({"PB"}),
            ),
        ),
    ]
    jax_backend = build_composite_engine(
        fitpars=("F0", "PB"),
        nrows=4,
        contributions=only_jax_contributions,
    )
    assert isinstance(jax_backend, PulsarJaxTimingEngine)
    assert jax_backend.precision_critical_fitpars() == frozenset({"F0", "PB"})


def test_absent_session_params_contribute_zero():
    contributions = _session_backends()
    delta = np.array([0.0, 3.0, 0.0], dtype=float)  # A1 absent from session_b

    engine = build_composite_engine(
        fitpars=("F0", "A1", "PB"),
        nrows=4,
        contributions=contributions,
    )
    out = engine.residual_delta(delta)
    # session_b does not own A1, so A1 contributes exactly zero on its rows.
    np.testing.assert_allclose(out[[3, 1]], np.array([0.0, 0.0]))


def test_mapped_but_unevaluable_param_uses_exact_linear_design_column():
    contribution = PtaContribution(
        name="pta_a",
        row_indices=np.array([0, 1]),
        engine=LinearizedPintEngine.from_linear_model(
            LinearModel.from_design(
                fitpars=("F0",),
                design=np.array([[1.0], [1.0]], dtype=float),
                theta_exact={"F0": "1.0"},
            )
        ),
        exact_linear_fitpars=frozenset({"A1"}),
        fallback_reference_exact={"A1": "2.0"},
    )
    design_matrix = np.array([[1.0, 10.0], [1.0, 11.0]], dtype=float)
    delta = np.array([0.0, 3.0], dtype=float)

    exact_backend = build_composite_engine(
        fitpars=("F0", "A1"),
        nrows=2,
        contributions=[contribution],
        design_matrix=design_matrix,
    )
    np.testing.assert_allclose(exact_backend.residual_delta(delta), [-30.0, -33.0])
    np.testing.assert_allclose(exact_backend.design_matrix()[:, 1], [10.0, 11.0])


@pytest.mark.requires_jug
def test_composite_keeps_jug_effective_exact_linear_column():
    pytest.importorskip("jax")
    model = LinearModel.from_design(
        fitpars=("PB", "JUMP"),
        design=np.array([[2.0, 10.0], [3.0, 11.0]], dtype=float),
        theta_exact={"PB": "1.0", "JUMP": "0.0"},
    )

    class _State:
        design_matrix = model.design[:, :1]
        param_mapping = ()

        def residual_delta_jax(self, jug_delta):
            import jax.numpy as jnp

            delta = jnp.asarray(jug_delta)
            return -(jnp.asarray(self.design_matrix) @ delta)

    jug = JugEngine(state=_State(), linear_model=model)
    jug._jug_indices = (0,)
    jug._jug_fitpars = ("PB",)
    jug._exact_linear_indices = (1,)
    jug._exact_linear_fitpars = frozenset({"JUMP"})
    engine = build_composite_engine(
        fitpars=model.fitpars,
        nrows=2,
        contributions=[
            PtaContribution(
                name="pta",
                row_indices=np.array([0, 1]),
                engine=jug,
                exact_linear_fitpars=frozenset({"JUMP"}),
            )
        ],
        design_matrix=model.design,
    )

    np.testing.assert_allclose(engine.design_matrix(), model.design)
    # JUMP is exact-linear via the host design matrix (fitter sign -M δ).
    delta = np.array([0.0, 2.0], dtype=float)
    np.testing.assert_allclose(
        engine.residual_delta(delta), -(model.design[:, 1] * 2.0)
    )


def test_build_engine_accepts_mixed_engine_families():
    contributions = _session_backends()
    engine = build_engine(
        fitpars=("F0", "A1", "PB"), nrows=4, contributions=contributions
    )
    assert isinstance(engine, PulsarTimingEngine)
