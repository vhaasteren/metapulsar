"""Residual-centering facts on leaf and composite engines."""

from __future__ import annotations

import numpy as np
from psrdata import ResidualCentering, SINGLE_KEY

from nltiming.engine_support import LinearModel
from metapulsar.engines.centering import PINT_FAMILY, TEMPO2_FAMILY
from metapulsar.engines.composite import PtaContribution, build_composite_engine
from metapulsar.engines.jug import JugEngine, LinearizedJugEngine
from metapulsar.engines.pint import LinearizedPintEngine, PintEngine
from metapulsar.engines.tempo2 import LibstempoEngine, LinearizedLibstempoEngine


def _model():
    return LinearModel.from_design(
        fitpars=("F0", "Offset"),
        design=np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float),
        theta_exact={"F0": "1.0", "Offset": "0.0"},
    )


def test_leaf_centering_table():
    model = _model()
    pint = LinearizedPintEngine.from_linear_model(model)
    assert pint.residual_centering[SINGLE_KEY] == PINT_FAMILY

    jug = LinearizedJugEngine.from_linear_model(model)
    assert jug.residual_centering[SINGLE_KEY].stored_residuals == "unknown"
    assert jug.residual_centering[SINGLE_KEY].standard_weighted is True

    lib_lin = LinearizedLibstempoEngine.from_linear_model(model)
    assert lib_lin.residual_centering[SINGLE_KEY].stored_residuals == "unknown"

    lib = LibstempoEngine(
        engine=type("E", (), {"delta_residuals": lambda self, d: np.zeros(2)})(),
        linear_model=model,
    )
    assert lib.residual_centering[SINGLE_KEY] == TEMPO2_FAMILY


def test_pint_engine_wrapper_centering():
    model = _model()

    class _Fake:
        def delta_residuals(self, delta_params):
            return np.zeros(2)

    eng = PintEngine(engine=_Fake(), linear_model=model)
    assert eng.residual_centering[SINGLE_KEY] == PINT_FAMILY


def test_jug_engine_centering_is_psrdata_type():
    model = _model()

    class _RefGauge:
        mode = "mean"
        weights = np.array([0.5, 0.5])

    class _State:
        compatibility = "pint"
        reference_gauge = _RefGauge()
        param_mapping = ()

        def residual_delta_jax(self, delta):
            import jax.numpy as jnp

            return jnp.zeros((2,))

    eng = JugEngine(state=_State(), linear_model=model)
    centering = eng.residual_centering[SINGLE_KEY]
    assert isinstance(centering, ResidualCentering)
    assert centering.stored_residuals == "none"
    assert centering.standard_output == "mean_removed"
    assert centering.standard_weighted is True
    assert "jug" not in type(centering).__module__


def test_composite_rekeys_centering_by_contribution():
    jug = LinearizedJugEngine.from_linear_model(_model())
    lib = LibstempoEngine(
        engine=type("E", (), {"delta_residuals": lambda self, d: np.zeros(2)})(),
        linear_model=_model(),
    )
    engine = build_composite_engine(
        fitpars=("F0", "Offset"),
        nrows=4,
        contributions=[
            PtaContribution(name="epta", row_indices=np.array([0, 1]), engine=jug),
            PtaContribution(name="ppta", row_indices=np.array([2, 3]), engine=lib),
        ],
    )
    assert set(engine.residual_centering) == {"epta", "ppta"}
    assert engine.residual_centering["ppta"] == TEMPO2_FAMILY
