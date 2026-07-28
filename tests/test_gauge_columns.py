"""MetaPulsar combined Mmat carries one named gauge column per PTA (G6 / §10.3)."""

from __future__ import annotations

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from nltiming.nonlinear_timing_model import GaugeColumnMissingError


def _build_mp():
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_a", seed=11
        ),
        "pta_b": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_b", seed=22
        ),
    }
    return MetaPulsar(pulsars, combination_strategy="per_pta")


def test_combined_mmat_has_one_constant_column_per_pta():
    mp = _build_mp()
    gauge_names = [p for p in mp.fitpars if p == "Offset" or p.startswith("Offset_")]
    assert len(gauge_names) >= len(mp._pta_data)
    # Per-PTA layout uses suffixed offsets.
    for pta in mp._pta_data:
        assert f"Offset_{pta}" in mp.fitpars

    slices = mp._get_pta_slices()
    for pta, slc in slices.items():
        col = mp.fitpars.index(f"Offset_{pta}")
        block = np.asarray(mp._designmatrix[slc.start : slc.stop, col], dtype=float)
        assert np.linalg.norm(block) > 0.0
        # Constant on its own PTA rows (up to scale).
        assert np.std(block) / max(np.mean(np.abs(block)), 1e-300) < 1e-8


def test_remove_nonidentifiable_never_silently_drops_gauge_column():
    mp = _build_mp()
    gauge_cols = [i for i, name in enumerate(mp.fitpars) if name.startswith("Offset_")]
    assert gauge_cols
    # Zero one gauge column and re-run removal + assertion: must raise, not drop.
    mp._designmatrix = np.array(mp._designmatrix, copy=True)
    mp._designmatrix[:, gauge_cols[0]] = 0.0
    with pytest.raises(GaugeColumnMissingError):
        mp._remove_nonidentifiable_parameters()
        mp._assert_gauge_columns()
