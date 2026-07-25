"""Regression: PINT-normalized FB0 vs tempo2 Enterprise PB fitpars.

ParameterManager always builds a PINT model (synthesizing FB0 for PB+FBn
hybrids). Tempo2/libstempo Enterprise pulsars keep PB. The bridge must map
and scale design-matrix columns so MetaPulsar construction succeeds.
"""

from pathlib import Path

import numpy as np
import pytest

from metapulsar.parameter_manager import ParameterManager
from metapulsar.pint_helpers import (
    designmatrix_scale_fb0_from_pb,
    par_text_has_ordinary_pb_without_fb0,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_parfiles"
DATA_CHECK = Path(__file__).resolve().parents[1] / "data-check"


def test_parameter_manager_maps_fb0_to_pb_for_tempo2_hybrid():
    text = (FIXTURE_DIR / "j2241_pb_fbn_no_fb0.par").read_text()
    assert par_text_has_ordinary_pb_without_fb0(text)

    pm = ParameterManager(
        file_data={
            "PPTA": {
                "par": None,
                "par_content": text,
                "timing_package": "tempo2",
            }
        },
        combine_components=["binary"],
        add_dm_derivatives=False,
    )
    mapping = pm.build_parameter_mappings()
    assert "FB0" in mapping.fitparameters
    assert mapping.fitparameters["FB0"]["PPTA"] == "PB"


@pytest.mark.slow
@pytest.mark.requires_libstempo
@pytest.mark.requires_ipta_data
def test_j2241_metapulsar_tempo2_builds_with_fb0_pb_bridge():
    par = DATA_CHECK / "PPTA_DR3" / "J2241-5236.par"
    tim = DATA_CHECK / "PPTA_DR3" / "J2241-5236.tim"
    if not par.exists() or not tim.exists():
        pytest.skip("PPTA_DR3 J2241 data not present under data-check/")

    from tests.helpers import make_tim_metadata
    from metapulsar import create_metapulsar

    meta = make_tim_metadata(timespan_days=4409.1, toa_count=6238, pn_status="complete")
    mp = create_metapulsar(
        {
            "PPTA": [
                {
                    "par": par,
                    "tim": tim,
                    "par_content": par.read_text(),
                    "timing_package": "tempo2",
                    "tim_metadata": meta,
                }
            ]
        },
        combination_strategy="consistent",
        combine_components=["astrometry", "spindown", "binary", "dispersion"],
        use_pulse_numbers="no",
        add_dm_derivatives=True,
    )
    assert mp.name.startswith("J2241")
    # Merged binary period is FB0; backend column comes from tempo2 PB
    assert "FB0" in mp.fitpars
    assert "PB" not in mp.fitpars
    assert mp._fitparameters["FB0"]["PPTA"] == "PB"
    pb = mp._backend_param_values["PPTA"]["PB"]
    col = mp._designmatrix[:, mp.fitpars.index("FB0")]
    assert np.sum(np.abs(col)) > 0.0
    # Scale factor must match the analytic Jacobian used by the bridge
    assert designmatrix_scale_fb0_from_pb(pb) != 0.0


@pytest.mark.slow
@pytest.mark.requires_ipta_data
def test_j2241_metapulsar_pint_builds_after_normalize():
    par = DATA_CHECK / "PPTA_DR3" / "J2241-5236.par"
    tim = DATA_CHECK / "PPTA_DR3" / "J2241-5236.tim"
    if not par.exists() or not tim.exists():
        pytest.skip("PPTA_DR3 J2241 data not present under data-check/")

    from tests.helpers import make_tim_metadata
    from metapulsar import create_metapulsar

    meta = make_tim_metadata(timespan_days=4409.1, toa_count=6238, pn_status="complete")
    mp = create_metapulsar(
        {
            "PPTA": [
                {
                    "par": par,
                    "tim": tim,
                    "par_content": par.read_text(),
                    "timing_package": "pint",
                    "tim_metadata": meta,
                }
            ]
        },
        combination_strategy="consistent",
        combine_components=["astrometry", "spindown", "binary", "dispersion"],
        use_pulse_numbers="no",
        add_dm_derivatives=True,
    )
    assert "FB0" in mp.fitpars
    assert mp._fitparameters["FB0"]["PPTA"] == "FB0"
    assert np.sum(np.abs(mp._designmatrix[:, mp.fitpars.index("FB0")])) > 0.0
