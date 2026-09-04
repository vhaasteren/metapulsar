"""engine.design_matrix() agrees with pulsar.Mmat columns."""

from __future__ import annotations

import numpy as np

from nltiming.engine_support import LinearModel
from metapulsar.engines.composite import PtaContribution, build_composite_engine
from metapulsar.engines.jug import LinearizedJugEngine
from metapulsar.engines.pint import LinearizedPintEngine
from nltiming.protocols import GaugeProvenance


def _gf():
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


def test_leaf_design_matrix_matches_mmat_block():
    M = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=float)
    fitpars = ("F0", "Offset")
    model = LinearModel.from_design(
        fitpars=fitpars, design=M, theta_exact={"F0": "1.0", "Offset": "0.0"}
    )
    for eng in (
        LinearizedJugEngine.from_linear_model(model, gauge_provenance=_gf()),
        LinearizedPintEngine.from_linear_model(model, gauge_provenance=_gf()),
    ):
        np.testing.assert_allclose(eng.design_matrix(), M, rtol=1e-10, atol=0.0)


def test_composite_design_matrix_matches_mmat_per_contribution():
    n = 4
    mid = 2
    M = np.zeros((n, 3), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:mid, 1] = 1.0
    M[mid:, 2] = 1.0
    fitpars = ("F0", "Offset_epta", "Offset_ppta")
    a = LinearizedPintEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_epta"),
            design=M[:mid, :2],
            theta_exact={"F0": "1.0", "Offset_epta": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    b = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_ppta"),
            design=np.column_stack([M[mid:, 0], M[mid:, 2]]),
            theta_exact={"F0": "1.0", "Offset_ppta": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    engine = build_composite_engine(
        fitpars=fitpars,
        nrows=n,
        contributions=[
            PtaContribution(name="epta", row_indices=np.arange(mid), engine=a),
            PtaContribution(name="ppta", row_indices=np.arange(mid, n), engine=b),
        ],
    )
    got = engine.design_matrix()
    np.testing.assert_allclose(got[:mid, :2], M[:mid, :2], rtol=1e-10)
    np.testing.assert_allclose(
        got[mid:][:, [0, 2]],
        np.column_stack([M[mid:, 0], M[mid:, 2]]),
        rtol=1e-10,
    )
