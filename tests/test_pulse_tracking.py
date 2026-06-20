"""Regression tests for pulse-number tracking during consistent merging."""

from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from pint.models import get_model_and_toas
from pint.residuals import Residuals

from metapulsar.metapulsar_factory import MetaPulsarFactory


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pulse_tracking"
EPTA_PAR = FIXTURE_DIR / "epta_like.par"
EPTA_TIM = FIXTURE_DIR / "epta_like.tim"
NANOGRAV_PAR = FIXTURE_DIR / "nanograv_like.par"
NANOGRAV_TIM = FIXTURE_DIR / "nanograv_like.tim"
EXPECTED_TOAS = 600
COHERENT_RMS_THRESHOLD_S = 1.0e-6
COHERENT_POSTFIT_RMS_THRESHOLD_S = 1.0e-6
INCOHERENT_POSTFIT_RMS_THRESHOLD_S = 1.0e-5


def _load_prefit_rms(par_path: Path, tim_path: Path) -> float:
    model, toas = get_model_and_toas(
        str(par_path),
        str(tim_path),
        allow_T2=True,
        planets=True,
    )
    residuals = Residuals(toas, model).time_resids.to(u.s).value
    return float(np.sqrt(np.mean(residuals**2)))


def _build_file_data(timing_package: str = "pint"):
    return {
        "epta_like": [
            {
                "par": EPTA_PAR,
                "tim": EPTA_TIM,
                "timing_package": timing_package,
            }
        ],
        "nanograv_like": [
            {
                "par": NANOGRAV_PAR,
                "tim": NANOGRAV_TIM,
                "timing_package": timing_package,
            }
        ],
    }


def _weighted_postfit_rms(metapulsar) -> float:
    residuals = np.asarray(metapulsar._residuals, dtype=float)
    design_matrix = np.asarray(metapulsar._designmatrix, dtype=float)
    toa_errors = np.asarray(metapulsar._toaerrs, dtype=float)

    weighted_matrix = design_matrix / toa_errors[:, None]
    weighted_residuals = residuals / toa_errors
    solution, *_ = np.linalg.lstsq(weighted_matrix, weighted_residuals, rcond=None)
    postfit_residuals = residuals - design_matrix @ solution
    return float(np.sqrt(np.mean(postfit_residuals**2)))


@pytest.mark.unit
@pytest.mark.slow
def test_pulse_tracking_fixtures_are_individually_coherent():
    epta_rms = _load_prefit_rms(EPTA_PAR, EPTA_TIM)
    nanograv_rms = _load_prefit_rms(NANOGRAV_PAR, NANOGRAV_TIM)

    assert epta_rms < COHERENT_RMS_THRESHOLD_S
    assert nanograv_rms < COHERENT_RMS_THRESHOLD_S


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.parametrize(
    "timing_package",
    [
        "pint",
        pytest.param("tempo2", marks=pytest.mark.requires_libstempo),
    ],
)
def test_pulse_tracking_recovers_coherent_postfit_solution(timing_package):
    file_data = _build_file_data(timing_package)
    factory = MetaPulsarFactory()

    metapulsar_without_pn = factory.create_metapulsar(
        file_data=file_data,
        use_pulse_numbers="no",
    )
    metapulsar_with_pn = factory.create_metapulsar(
        file_data=file_data,
        use_pulse_numbers="yes",
    )

    assert len(metapulsar_without_pn._toas) == EXPECTED_TOAS
    assert len(metapulsar_with_pn._toas) == EXPECTED_TOAS

    postfit_without_pn = _weighted_postfit_rms(metapulsar_without_pn)
    postfit_with_pn = _weighted_postfit_rms(metapulsar_with_pn)

    assert postfit_with_pn <= COHERENT_POSTFIT_RMS_THRESHOLD_S, (
        f"[{timing_package}] Pulse-number-tracked fit did not recover a coherent "
        "post-fit solution. "
        f"threshold={COHERENT_POSTFIT_RMS_THRESHOLD_S:.3e}s "
        f"postfit_with_pn={postfit_with_pn:.3e}s"
    )
    assert postfit_without_pn >= INCOHERENT_POSTFIT_RMS_THRESHOLD_S, (
        f"[{timing_package}] Disabling pulse-number tracking did not produce the "
        "expected incoherent post-fit solution. "
        f"threshold={INCOHERENT_POSTFIT_RMS_THRESHOLD_S:.3e}s "
        f"postfit_without_pn={postfit_without_pn:.3e}s"
    )
