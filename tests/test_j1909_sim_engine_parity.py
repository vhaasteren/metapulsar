"""PINT vs Tempo2 residual parity for the notebook-05 ELL1 simulation.

The Vela vs nltiming overlay notebook must not carry this check. It asserts
that the committed dual-engine J1909-3744-sim files agree at the < 1 ns RMS
residual-difference floor used for barycentric, DM-0 comparisons.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

libstempo = pytest.importorskip("libstempo")

pytestmark = [
    pytest.mark.requires_libstempo,
    pytest.mark.skipif(
        shutil.which("tempo2") is None, reason="tempo2 binary not available"
    ),
]

SIM_DIR = (
    Path(__file__).resolve().parents[1]
    / "ref-packages"
    / "nltiming"
    / "examples"
    / "data"
    / "J1909-3744-sim"
)
PAR = SIM_DIR / "J1909-3744.par"
TIM = SIM_DIR / "J1909-3744.tim"


@pytest.mark.skipif(not PAR.is_file() or not TIM.is_file(), reason="sim files missing")
def test_j1909_sim_pint_tempo2_residual_difference_under_1ns():
    import astropy.units as u
    from pint.models import get_model
    from pint.residuals import Residuals
    from pint.toa import get_TOAs

    model = get_model(str(PAR))
    toas = get_TOAs(str(TIM), model=model, include_pn=True)
    r_pint = np.asarray(Residuals(toas, model).time_resids.to_value(u.s))

    psr = libstempo.tempopulsar(parfile=str(PAR), timfile=str(TIM), dofit=False)
    r_t2 = np.asarray(
        psr.residuals(updatebats=True, formresiduals=True, removemean=True),
        dtype=float,
    )
    assert len(r_pint) == len(r_t2) == 100

    r_pint = r_pint - r_pint.mean()
    r_t2 = r_t2 - r_t2.mean()
    rms_ns = float(np.sqrt(np.mean((r_pint - r_t2) ** 2))) * 1e9
    assert rms_ns < 1.0, f"PINT vs tempo2 residual-difference RMS {rms_ns:.4f} ns"
