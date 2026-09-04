"""Mixed-gauge composite behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engine_support import LinearModel
from metapulsar.engines.composite import PtaContribution, build_composite_engine
from metapulsar.engines.jug import LinearizedJugEngine
from metapulsar.engines.tempo2 import LibstempoEngine
from nltiming.nonlinear_timing_model import (
    _timing_design_matrix,
    assert_gauge_column_present,
)
from nltiming.protocols import GaugeProvenance


def _gf():
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


class _Pulsar:
    def __init__(self, fitpars, Mmat):
        self.name = "J0000+0000"
        self.fitpars = list(fitpars)
        self.Mmat = np.asarray(Mmat, dtype=float)


def _mixed_engine():
    n = 6
    mid = 3
    M = np.zeros((n, 3), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:mid, 1] = 1.0
    M[mid:, 2] = 1.0
    fitpars = ("F0", "Offset_epta", "Offset_ppta")

    jug = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_epta"),
            design=M[:mid, :2],
            theta_exact={"F0": "1.0", "Offset_epta": "0.0"},
        ),
        gauge_provenance=_gf(),
    )

    class _Centered:
        def delta_residuals(self, delta_params):
            delta = np.array(
                [
                    delta_params.get("F0", 0.0),
                    delta_params.get("Offset_ppta", 0.0),
                ],
                dtype=float,
            )
            design = np.column_stack([M[mid:, 0], M[mid:, 2]])
            raw = -(design @ delta)
            return raw - np.mean(raw)

    lib = LibstempoEngine(
        engine=_Centered(),
        linear_model=LinearModel.from_design(
            fitpars=("F0", "Offset_ppta"),
            design=np.column_stack([M[mid:, 0], M[mid:, 2]]),
            theta_exact={"F0": "1.0", "Offset_ppta": "0.0"},
        ),
        native_fitpars=("F0", "Offset_ppta"),
        exact_linear_fitpars=frozenset(),
    )
    engine = build_composite_engine(
        fitpars=fitpars,
        nrows=n,
        contributions=[
            PtaContribution(name="epta", row_indices=np.arange(mid), engine=jug),
            PtaContribution(name="ppta", row_indices=np.arange(mid, n), engine=lib),
        ],
    )
    return fitpars, M, engine


def test_mixed_composite_reports_gauge_applied():
    _, _, engine = _mixed_engine()
    assert engine.gauge_applied is True


def test_mixed_composite_passes_g6():
    fitpars, M, engine = _mixed_engine()
    assert_gauge_column_present(_Pulsar(fitpars, M), engine, M)


def test_autodiff_on_mixed_composite_raises_naming_libstempo():
    fitpars, M, engine = _mixed_engine()
    pulsar = _Pulsar(fitpars, M)
    with pytest.raises(ValueError, match="LibstempoEngine|residual_jacobian"):
        _timing_design_matrix(pulsar, engine, method="autodiff")
