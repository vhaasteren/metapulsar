"""Tests for the Vela.jl (pyvela) engine adapter."""

from pathlib import Path

import numpy as np
import pytest
from nltiming.engine_support import LinearModel

from metapulsar.engines.vela import VelaDeltaEngine, VelaEngine

pytestmark = pytest.mark.requires_vela


class _MockSPNTA:
    """Linear stand-in for pyvela.SPNTA with Vela's unit/offset conventions."""

    def __init__(self):
        self.param_names = np.array(["F0", "F1", "PB"])
        # PINT units -> raw units, componentwise.
        self.scale_factors = np.array([2.0, 0.5, 4.0])
        self.default_params = np.array([100.0, -1.0e-15, 40.0])
        # d(residual)/d(raw param): 4 TOAs x 3 params.
        self.response = np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
            ]
        )
        self.calls = 0

    def time_residuals(self, raw):
        self.calls += 1
        raw = np.asarray(raw, dtype=float)
        return self.response @ (raw - self.default_params)

    def scaled_toa_unceritainties(self, raw):  # pyvela spelling (sic)
        return np.full(self.response.shape[0], 2.0)


def _linear_model(fitpars=("F0", "F1", "PB"), n=4):
    rng = np.random.default_rng(7)
    return LinearModel.from_design(
        fitpars=tuple(fitpars),
        design=rng.normal(size=(n, len(fitpars))),
        theta_exact={name: "1.0" for name in fitpars},
    )


def test_delta_engine_scales_native_deltas_to_raw_units():
    spnta = _MockSPNTA()
    engine = VelaDeltaEngine(spnta, phase_mean_mode=None)
    delta = engine.delta_residuals({"F0": 0.25, "PB": -1.0})
    expected = spnta.response @ (np.array([0.25 * 2.0, 0.0, -1.0 * 4.0]))
    np.testing.assert_allclose(delta, expected)


def test_delta_engine_zero_delta_short_circuits():
    spnta = _MockSPNTA()
    engine = VelaDeltaEngine(spnta, phase_mean_mode=None)
    calls_after_init = spnta.calls
    np.testing.assert_array_equal(engine.delta_residuals({"F0": 0.0}), np.zeros(4))
    assert spnta.calls == calls_after_init


def test_delta_engine_unknown_param_raises():
    engine = VelaDeltaEngine(_MockSPNTA(), phase_mean_mode=None)
    with pytest.raises(KeyError, match="no free parameter 'DMX_0001'"):
        engine.delta_residuals({"DMX_0001": 1.0})


def test_delta_engine_applies_isort():
    spnta = _MockSPNTA()
    isort = np.array([3, 2, 1, 0])
    engine = VelaDeltaEngine(spnta, isort=isort, phase_mean_mode=None)
    delta = engine.delta_residuals({"F1": 1.0})
    expected = (spnta.response @ np.array([0.0, 0.5, 0.0]))[isort]
    np.testing.assert_allclose(delta, expected)


def test_vela_engine_routes_unsupported_params_to_exact_linear():
    model = _linear_model(fitpars=("F0", "F1", "PB", "DMX_0001", "JUMPX"))
    engine = VelaEngine.from_contribution(
        _MockSPNTA(), linear_model=model, phase_mean_mode=None
    )
    assert engine.exact_linear_fitpars() == {"DMX_0001", "JUMPX"}

    delta = np.array([0.1, -0.2, 0.3, 1.0, -1.0])
    out = engine.residual_delta(delta)
    spnta = _MockSPNTA()
    nonlinear = spnta.response @ (delta[:3] * spnta.scale_factors)
    # Exact-linear path follows the fitter sign contract residual ≈ -M δ.
    exact = -(model.design[:, [3, 4]] @ delta[[3, 4]])
    np.testing.assert_allclose(out, nonlinear + exact)


def test_vela_engine_serves_pulsar_design_and_reference():
    model = _linear_model()
    engine = VelaEngine.from_contribution(_MockSPNTA(), linear_model=model)
    np.testing.assert_allclose(engine.design_matrix(), model.design)
    assert engine.reference_theta_exact() == dict(model.theta_exact)


def test_vela_engine_param_mapping_translates_names():
    model = _linear_model(fitpars=("A1DOT",))
    spnta = _MockSPNTA()
    spnta.param_names = np.array(["XDOT"])
    spnta.scale_factors = np.array([3.0])
    spnta.default_params = np.array([0.5])
    spnta.response = np.array([[1.0], [2.0], [0.0], [-1.0]])
    engine = VelaEngine.from_contribution(
        spnta,
        linear_model=model,
        param_mapping={"A1DOT": "XDOT"},
        phase_mean_mode=None,
    )
    out = engine.residual_delta(np.array([2.0]))
    np.testing.assert_allclose(out, spnta.response @ np.array([2.0 * 3.0]))


def test_vela_engine_requires_some_native_params():
    model = _linear_model(fitpars=("DMX_0001",))
    with pytest.raises(ValueError, match="No Vela-evaluable"):
        VelaEngine.from_contribution(_MockSPNTA(), linear_model=model)


def test_delta_engine_is_gauge_free_by_default():
    """Default phase_mean_mode=None exports residuals with no mean removal."""
    spnta = _MockSPNTA()
    engine = VelaDeltaEngine(spnta)
    delta = engine.delta_residuals({"F0": 1.0})
    raw_delta = spnta.response @ np.array([1.0 * 2.0, 0.0, 0.0])
    np.testing.assert_allclose(delta, raw_delta)
    assert abs(delta.mean()) > 1e-15


def test_delta_engine_subtracts_weighted_phase_mean_when_enabled():
    spnta = _MockSPNTA()
    engine = VelaDeltaEngine(spnta, phase_mean_mode="weighted")
    delta = engine.delta_residuals({"F0": 1.0})
    raw_delta = spnta.response @ np.array([1.0 * 2.0, 0.0, 0.0])
    # Mock uncertainties are uniform, so the weighted mean is the plain mean.
    np.testing.assert_allclose(delta, raw_delta - raw_delta.mean())
    assert abs(delta.mean()) < 1e-15


def test_delta_engine_unweighted_and_invalid_phase_mean_modes():
    spnta = _MockSPNTA()
    engine = VelaDeltaEngine(spnta, phase_mean_mode="unweighted")
    delta = engine.delta_residuals({"F1": 2.0})
    raw_delta = spnta.response @ np.array([0.0, 2.0 * 0.5, 0.0])
    np.testing.assert_allclose(delta, raw_delta - raw_delta.mean())
    with pytest.raises(ValueError, match="phase_mean_mode"):
        VelaDeltaEngine(spnta, phase_mean_mode="bogus")


def test_spnta_ingest_has_no_ecorr_and_keeps_toa_order(tmp_path):
    """Two interleaved ECORR groups would permute Julia TOAs if left on the par.

    Compare ``spnta.mjds`` (Julia / ``form_residuals`` order), not
    ``spnta.toas_pint``, which pyvela stores before ``ecorr_sort``.
    """
    from pint.models import get_model_and_toas
    from pyvela import SPNTA
    from pyvela.ecorr import ecorr_sort

    from metapulsar.engines.vela import _prepare_par_for_spnta, _refuse_ecorr_kernel

    sample_dir = Path(__file__).resolve().parents[1] / "fixtures" / "sample_parfiles"
    par = tmp_path / "ecorr.par"
    par.write_text(
        (sample_dir / "simple.par").read_text().rstrip("\n")
        + "\nECORR -f A 0.01\nECORR -f B 0.01\n"
    )
    tim = tmp_path / "interleaved.tim"
    # Same ~1s ECORR epoch, interleaved backends: ecorr_sort bunches A then B.
    tim.write_text(
        "FORMAT 1\n"
        "toa0 1400.0 54500.123456000 1.5 g -f A\n"
        "toa1 1400.0 54500.123456100 1.5 g -f B\n"
        "toa2 1400.0 54500.123456200 1.5 g -f A\n"
        "toa3 1400.0 54500.123456300 1.5 g -f B\n"
        "toa4 1400.0 54500.123456400 1.5 g -f A\n"
    )

    noisy_model, noisy_toas = get_model_and_toas(str(par), str(tim), planets=True)
    assert "EcorrNoise" in noisy_model.components
    sorted_toas, _, _ = ecorr_sort(noisy_model, noisy_toas)
    assert not np.allclose(
        noisy_toas.get_mjds().value,
        sorted_toas.get_mjds().value,
        rtol=0.0,
        atol=1e-12,
    )

    prepared = _prepare_par_for_spnta(par, tim)
    assert prepared != par
    assert "ECORR" in par.read_text()
    assert "ECORR" not in prepared.read_text()

    spnta = SPNTA(str(prepared), str(tim), center_epochs=False, check=False)
    _refuse_ecorr_kernel(spnta)
    assert not spnta.has_ecorr_noise
    assert "EcorrNoise" not in spnta.model_pint.components

    _, host = get_model_and_toas(str(prepared), str(tim), planets=True)
    # ``SPNTA.mjds`` is days from PEPOCH, not absolute MJD; compare offsets.
    # Clock/TDB conversion noise is ~1e-12 day; an ecorr_sort permutation is
    # a 1e-7 day jump.
    np.testing.assert_allclose(
        host.get_mjds().value - host.get_mjds().value[0],
        spnta.mjds - spnta.mjds[0],
        rtol=0.0,
        atol=1e-10,
    )
