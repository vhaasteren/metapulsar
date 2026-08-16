"""Par preparation for pyvela ingestion: empty frozen mask parameters.

``pyvela.model.read_mask`` asserts that every mask parameter selects at least
one TOA. Release pars (PPTA-style backend flag JUMPs carried into IPTA/AEI
combined products) routinely violate that with frozen leftovers that tempo2
happily ignores, so the Vela adapter strips them before ``SPNTA`` sees the par.
"""

from pathlib import Path

import pytest
from pint.models import get_model_and_toas

from metapulsar.engines.vela import (
    EmptyMaskParameterError,
    _prepare_par_for_spnta,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample_parfiles"
TIM_FILE = SAMPLE_DIR / "simple.tim"

# Every TOA in simple.tim carries `-sys TEST -group TEST` and sits near MJD 54500.
LIVE_JUMP = "JUMP -sys TEST 1.0e-6 0"
# A leftover PPTA backend key, and a flag named for another pulsar entirely.
DEAD_FLAG_JUMP = "JUMP -sys caspsr 2.0e-6 0"
DEAD_GROUP_JUMP = "JUMP -group dfb3_J0437_56160_60000 3.0e-6 0"
# MJD-window selector: exercises the numeric key-value comparison.
DEAD_MJD_JUMP = "JUMP MJD 40000 41000 4.0e-6 0"


def _write_par(tmp_path, jump_lines, *, fit_f1=False):
    text = (SAMPLE_DIR / "simple.par").read_text()
    if fit_f1:
        text = text.replace("F1          -1.61e-15", "F1          -1.61e-15 1")
    lines = text.rstrip("\n").splitlines() + list(jump_lines)
    par = tmp_path / "with_jumps.par"
    par.write_text("\n".join(lines) + "\n")
    return par


def _jump_selectors(par, tim):
    """Every JUMP's TOA count, the way pyvela's read_mask would see it."""
    model, toas = get_model_and_toas(str(par), str(tim), planets=False)
    component = model.components.get("PhaseJump")
    if component is None:
        return {}
    return {
        param.name: len(param.select_toa_mask(toas))
        for param in component.get_jump_param_objects()
    }


def test_empty_frozen_jumps_are_stripped(tmp_path):
    par = _write_par(
        tmp_path, [LIVE_JUMP, DEAD_FLAG_JUMP, DEAD_GROUP_JUMP, DEAD_MJD_JUMP]
    )
    assert sorted(_jump_selectors(par, TIM_FILE).values()) == [0, 0, 0, 5]

    prepared = _prepare_par_for_spnta(par, TIM_FILE)
    assert prepared != par
    text = prepared.read_text()
    assert "-sys TEST" in text
    assert "caspsr" not in text
    assert "dfb3_J0437_56160_60000" not in text
    assert "40000" not in text
    # What pyvela now loads has no empty mask left to assert on.
    assert list(_jump_selectors(prepared, TIM_FILE).values()) == [5]


def test_empty_fitted_jump_raises(tmp_path):
    par = _write_par(tmp_path, [LIVE_JUMP, "JUMP -sys caspsr 2.0e-6 1"])
    with pytest.raises(EmptyMaskParameterError, match="no TOAs"):
        _prepare_par_for_spnta(par, TIM_FILE)


def test_populated_jumps_leave_the_par_untouched(tmp_path):
    par = _write_par(tmp_path, [LIVE_JUMP])
    assert _prepare_par_for_spnta(par, TIM_FILE) == par


def test_par_without_jumps_never_reads_the_tim(tmp_path):
    par = _write_par(tmp_path, [])
    assert _prepare_par_for_spnta(par, tmp_path / "does-not-exist.tim") == par


def test_preloaded_mask_reference_avoids_a_second_toa_read(tmp_path):
    par = _write_par(tmp_path, [LIVE_JUMP, DEAD_FLAG_JUMP])
    reference = get_model_and_toas(str(par), str(TIM_FILE), planets=False)
    prepared = _prepare_par_for_spnta(
        par, tmp_path / "does-not-exist.tim", mask_reference=reference
    )
    assert "caspsr" not in prepared.read_text()


def test_uncertainty_shim_still_applies_after_stripping(tmp_path):
    par = _write_par(tmp_path, [LIVE_JUMP, DEAD_FLAG_JUMP], fit_f1=True)
    prepared = _prepare_par_for_spnta(par, TIM_FILE)
    lines = prepared.read_text().splitlines()
    (f1_line,) = [ln for ln in lines if ln.split()[:1] == ["F1"]]
    assert f1_line.split() == ["F1", "-1.61e-15", "1", "1.0"]
    assert "caspsr" not in prepared.read_text()
