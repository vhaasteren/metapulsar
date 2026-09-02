"""MetaPulsar combined Mmat carries one named gauge column per PTA (G6).

Two halves: the mock-backed gate that every leg family shares, and the
vela-jax leg, whose gauge column is not a constant vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo, write_mock_pta_files
from nltiming.nonlinear_timing_model import GaugeColumnMissingError


def _build_mp(directory):
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_a", seed=11
        ),
        "pta_b": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_b", seed=22
        ),
    }
    return MetaPulsar(
        pulsars,
        combination_strategy="per_pta",
        pta_files=write_mock_pta_files(pulsars, directory),
    )


def test_combined_mmat_has_one_constant_column_per_pta(tmp_path):
    mp = _build_mp(tmp_path)
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


def test_remove_nonidentifiable_never_silently_drops_gauge_column(tmp_path):
    mp = _build_mp(tmp_path)
    gauge_cols = [i for i, name in enumerate(mp.fitpars) if name.startswith("Offset_")]
    assert gauge_cols
    # Zero one gauge column and re-run removal + assertion: must raise, not drop.
    mp._designmatrix = np.array(mp._designmatrix, copy=True)
    mp._designmatrix[:, gauge_cols[0]] = 0.0
    with pytest.raises(GaugeColumnMissingError):
        mp._remove_nonidentifiable_parameters()
        mp._assert_gauge_columns()


# --- a vela-jax leg's gauge column (PR-4, D10) ------------------------------
#
# The check above is worth having only if it can fail, and a vela-jax leg
# makes it non-trivial: its gauge column is 1/F_i over the *instantaneous*
# doppler-shifted spin frequency, not the constant vector a constant-F0
# engine gives, so the leg declares its own direction
# (vela_jax.engine.Engine.gauge_direction). The negative test below is the
# proof that the declaration is checked rather than assumed.

AEI = Path(__file__).resolve().parents[1] / "data" / "aei-dr2"
PSR = "J1853+1303"

#: The leg tests read a real par/tim through vela-jax; the mock gate above
#: needs neither, so the marks go on the tests rather than on the module.
real_leg = pytest.mark.skipif(
    not (AEI / "nanograv_9y" / "par" / f"{PSR}.par").exists(),
    reason="AEI-DR2 tree not present",
)


def _leg_pulsar(release, package):
    from metapulsar import create_metapulsar

    base = AEI / release
    return create_metapulsar(
        {
            release: [
                {
                    "par": base / "par" / f"{PSR}.par",
                    "tim": base / "tim" / f"{PSR}.tim",
                    "timing_package": package,
                }
            ]
        },
        combination_strategy="per_pta",
        use_pulse_numbers="reuse",
        engines="vela_jax",
    )


@pytest.mark.requires_vela_jax
@pytest.mark.real_data
@real_leg
def test_a_real_build_carries_its_gauge_column():
    """Construction runs the check; reaching here is the positive case."""
    pytest.importorskip("vela_jax")
    pulsar = _leg_pulsar("nanograv_9y", "pint")
    gauge = [name for name in pulsar.fitpars if name.startswith(("PHOFF", "Offset"))]
    assert gauge, f"no gauge column among {pulsar.fitpars}"


@pytest.mark.requires_vela_jax
@pytest.mark.real_data
@real_leg
def test_faked_gauge_columns_are_caught():
    """Replace the leg's ``1/F_i`` column with ones and the check must raise.

    A constant column is what a constant-``F0`` engine would give, and it is
    what the check accepted before the direction was declared. If this passes
    silently the assertion has gone tautological.
    """
    pytest.importorskip("vela_jax")
    from nltiming.nonlinear_timing_model import assert_gauge_column_present

    pulsar = _leg_pulsar("nanograv_9y", "pint")
    real = np.asarray(pulsar._designmatrix, dtype=float)
    design = np.array(real, dtype=float)
    columns = [
        i
        for i, name in enumerate(pulsar.fitpars)
        if name.startswith(("PHOFF", "Offset"))
    ]
    assert columns
    design[:, columns] = 1.0

    class _Leaf:
        gauge_applied = False

        def __init__(self, record):
            self.gauge_direction = record._leg.engine.gauge_direction

        def gauge_provenance(self):
            from nltiming.protocols import GaugeProvenance

            return GaugeProvenance(export="none", reference_mode="none")

    class _Contribution:
        def __init__(self, name, rows, record):
            self.name = name
            self.row_indices = np.asarray(rows, dtype=int)
            self.engine = _Leaf(record)

    class _StubEngine:
        def __init__(self, contributions):
            self.contributions = contributions

    slices = pulsar._get_pta_slices()
    stub = _StubEngine(
        [
            _Contribution(
                pta, np.arange(slc.start, slc.stop, dtype=int), pulsar._pta_data[pta]
            )
            for pta, slc in slices.items()
        ]
    )

    # The same call on the real matrix must pass, or the raise below would be
    # about the stub rather than about the faked columns.
    assert_gauge_column_present(pulsar, stub, real)

    with pytest.raises(Exception, match="(?i)gauge"):
        assert_gauge_column_present(pulsar, stub, design)
