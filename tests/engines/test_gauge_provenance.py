"""Gauge provenance population at context construction."""

from __future__ import annotations

import numpy as np

from nltiming.engine_support import LinearModel
from metapulsar.engines.composite import PtaContribution, build_composite_engine
from metapulsar.engines.jug import JugEngine, LinearizedJugEngine
from metapulsar.engines.pint import LinearizedPintEngine, PintEngine
from metapulsar.engines.tempo2 import LibstempoEngine, LinearizedLibstempoEngine
from nltiming.protocols import GaugeProvenance


def _gf(**kwargs):
    base = dict(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )
    base.update(kwargs)
    return GaugeProvenance(**base)


def _model():
    return LinearModel.from_design(
        fitpars=("F0", "Offset"),
        design=np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float),
        theta_exact={"F0": "1.0", "Offset": "0.0"},
    )


def test_leaf_provenance_table():
    model = _model()
    cases = [
        (
            LinearizedPintEngine.from_linear_model(model),
            "none",
            "unknown",
            False,
        ),
        (
            LinearizedJugEngine.from_linear_model(model),
            "none",
            "unknown",
            False,
        ),
        (
            LinearizedLibstempoEngine.from_linear_model(model),
            "none",
            "unknown",
            False,
        ),
        (
            LibstempoEngine(
                engine=type(
                    "E",
                    (),
                    {"delta_residuals": lambda self, d: np.zeros(2)},
                )(),
                linear_model=model,
            ),
            "applied-unknown",
            "unknown",
            True,
        ),
    ]
    for eng, export, ref_mode, applied in cases:
        prov = eng.gauge_provenance()
        assert prov.export == export
        assert prov.reference_mode == ref_mode
        assert eng.gauge_applied is applied
        assert eng.gauge_applied == (prov.export != "none")


def test_pint_engine_wrapper_provenance():
    model = _model()

    class _Fake:
        def delta_residuals(self, delta_params):
            return np.zeros(2)

    eng = PintEngine(engine=_Fake(), linear_model=model)
    assert eng.gauge_provenance().export == "none"
    assert eng.gauge_applied is False


def test_jug_engine_translates_reference_gauge_without_exporting_jug_type():
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
    prov = eng.gauge_provenance()
    assert prov.export == "none"
    assert prov.reference_mode == "mean"
    assert prov.reference_weighted is True
    # Provenance is backend-neutral — not a JUG ReferenceGauge.
    assert type(prov).__name__ == "GaugeProvenance"
    assert "jug" not in type(prov).__module__


def test_composite_or_gauge_applied():
    jug = LinearizedJugEngine.from_linear_model(_model(), gauge_provenance=_gf())
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
    assert engine.gauge_applied is True
