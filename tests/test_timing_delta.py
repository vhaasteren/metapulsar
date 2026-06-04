"""Tests for non-linear timing residual-deviation providers."""

from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
from pint.models import get_model_and_toas
from pint.residuals import Residuals

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from metapulsar.nonlinear_timing_model import (
    JugDeltaEngine,
    PintDeltaEngine,
    Tempo2DeltaEngine,
    infer_jug_param_mapping,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pulse_tracking"


def _mock_metapulsar():
    return MetaPulsar(
        {
            "pta_a": create_mock_libstempo(
                n_toas=12, name="J1857+0943", telescope="pta_a", seed=10
            ),
            "pta_b": create_mock_libstempo(
                n_toas=9, name="J1857+0943", telescope="pta_b", seed=20
            ),
        },
        combination_strategy="consistent",
    )


def test_timing_delta_zero_is_exact_for_metapulsar():
    mp = _mock_metapulsar()

    np.testing.assert_array_equal(mp.timing_delta({}), np.zeros(len(mp.residuals)))
    np.testing.assert_array_equal(
        mp.timing_delta({mp.fitpars[0]: 0.0}), np.zeros(len(mp.residuals))
    )
    np.testing.assert_array_equal(mp.residuals_at({}), mp.residuals)


def test_timing_delta_matches_design_matrix_for_mock_tempo2_path():
    mp = _mock_metapulsar()

    for param in mp.fitpars[:4]:
        delta = 1.0e-9
        got = mp.timing_delta({param: delta})
        column = mp._designmatrix[:, mp.fitpars.index(param)][mp._isort]
        expected = column * delta
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-18)


def test_timing_delta_output_uses_residuals_sort_order():
    mp = _mock_metapulsar()
    param = mp.fitpars[0]
    delta = 1.0e-9

    internal_expected = mp._designmatrix[:, mp.fitpars.index(param)] * delta

    np.testing.assert_allclose(
        mp.timing_delta({param: delta}),
        internal_expected[mp._isort],
        rtol=0.0,
        atol=1.0e-18,
    )


def test_tempo2_delta_engine_zero_and_mock_linear_fallback():
    mock_lt = create_mock_libstempo(n_toas=8, seed=1)
    engine = Tempo2DeltaEngine(mock_lt)

    np.testing.assert_array_equal(
        engine.delta_residuals({}), np.zeros(len(mock_lt.residuals()))
    )

    delta = 1.0e-9
    expected = mock_lt.designmatrix()[:, 1] * delta
    np.testing.assert_allclose(
        engine.delta_residuals({"F0": delta}), expected, rtol=0.0, atol=1.0e-18
    )


def test_pint_delta_engine_matches_pint_residual_recompute():
    parfile = FIXTURE_DIR / "epta_like.par"
    timfile = FIXTURE_DIR / "epta_like.tim"
    model, toas = get_model_and_toas(
        str(parfile),
        str(timfile),
        allow_T2=True,
        planets=True,
    )
    engine = PintDeltaEngine(model, toas)

    np.testing.assert_array_equal(
        engine.delta_residuals({}), np.zeros(len(toas.get_mjds()))
    )

    delta_params = {"DM": 1.0e-8}
    perturbed_model = deepcopy(model)
    perturbed_model.DM.value = model.DM.value + delta_params["DM"]

    expected = (
        (
            Residuals(toas, perturbed_model).time_resids
            - Residuals(toas, model).time_resids
        )
        .to(u.s)
        .value
    )
    got = engine.delta_residuals(delta_params)

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-16)


class _MockJugSession:
    def __init__(self, matrix, reference_params, param_order=None):
        self._matrix = np.asarray(matrix, dtype=float)
        self.params = dict(reference_params)
        self._param_order = (
            list(param_order)
            if param_order is not None
            else list(reference_params.keys())
        )

    def _residuals_for(self, params, *, subtract_tzr=True):
        assert subtract_tzr is True
        current = dict(self.params)
        if params:
            current.update(params)
        residuals_sec = np.zeros(self._matrix.shape[0], dtype=float)
        for col_idx, param_name in enumerate(self._param_order):
            residuals_sec += self._matrix[:, col_idx] * float(current[param_name])
        return {"residuals_us": residuals_sec * 1.0e6}

    def compute_residuals(self, params=None, subtract_tzr=True):
        return self._residuals_for(params, subtract_tzr=subtract_tzr)

    def residuals_at_params(self, params, subtract_tzr=True):
        return self._residuals_for(params, subtract_tzr=subtract_tzr)


def test_jug_delta_engine_zero_and_linear_delta():
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 2.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    session = _MockJugSession(matrix, {"F0": 2.0, "DM": -3.0})
    engine = JugDeltaEngine(session, fitpars=["F0", "DM"])

    np.testing.assert_array_equal(engine.delta_residuals({}), np.zeros(matrix.shape[0]))

    delta_params = {"F0": 1.0e-9, "DM": -2.0e-9}
    got = engine.delta_residuals(delta_params)
    expected = matrix[:, 0] * delta_params["F0"] + matrix[:, 1] * delta_params["DM"]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1.0e-15)


def test_infer_jug_param_mapping_maps_ecliptic_aliases():
    backend = {"LAMBDA", "BETA", "PMLAMBDA", "PMBETA", "E", "PB", "F0"}
    mapping = infer_jug_param_mapping(
        ["ELONG", "ELAT", "PMELONG", "PMELAT", "ECC", "PB"],
        backend,
    )
    assert mapping == {
        "ELONG": "LAMBDA",
        "ELAT": "BETA",
        "PMELONG": "PMLAMBDA",
        "PMELAT": "PMBETA",
        "ECC": "E",
    }


def test_jug_delta_engine_supports_canonical_mapping():
    matrix = np.array([[1.0], [2.0]], dtype=float)
    session = _MockJugSession(matrix, {"F0": 10.0}, param_order=["F0"])
    engine = JugDeltaEngine(
        session,
        fitpars=["F0_CANON"],
        param_names=["F0"],
        param_mapping={"F0_CANON": "F0"},
    )

    delta = 2.0e-9
    got = engine.delta_residuals({"F0_CANON": delta})
    np.testing.assert_allclose(got, matrix[:, 0] * delta, rtol=0.0, atol=1.0e-15)


def test_jug_delta_engine_applies_isort():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    session = _MockJugSession(matrix, {"F0": 1.0, "DM": 2.0}, param_order=["F0", "DM"])
    isort = np.array([2, 0, 1])
    engine = JugDeltaEngine(session, fitpars=["F0", "DM"], isort=isort)

    delta = {"F0": 1.0e-9}
    expected = (matrix[:, 0] * delta["F0"])[isort]
    np.testing.assert_allclose(
        engine.delta_residuals(delta), expected, rtol=0.0, atol=1.0e-14
    )


def test_jug_delta_engine_ignores_explicit_zero_deltas():
    matrix = np.array([[1.0, 0.0], [0.5, 2.0]], dtype=float)
    session = _MockJugSession(matrix, {"F0": 2.0, "DM": -3.0})
    engine = JugDeltaEngine(session, fitpars=["F0", "DM"])

    delta = 1.0e-9
    single = engine.delta_residuals({"F0": delta})
    with_zeros = engine.delta_residuals({"F0": delta, "DM": 0.0})
    np.testing.assert_allclose(with_zeros, single, rtol=0.0, atol=0.0)


def test_jug_delta_engine_uses_residuals_at_params_for_perturbations(monkeypatch):
    matrix = np.array([[1.0], [2.0]], dtype=float)
    session = _MockJugSession(matrix, {"F0": 10.0}, param_order=["F0"])
    engine = JugDeltaEngine(session, fitpars=["F0"])

    def _fail_slow(*_args, **kwargs):
        if kwargs.get("params"):
            raise AssertionError("slow temp-par path should not run for deltas")
        return session.compute_residuals(params=None, subtract_tzr=True)

    monkeypatch.setattr(session, "compute_residuals", _fail_slow)
    got = engine.delta_residuals({"F0": 2.0e-9})
    np.testing.assert_allclose(got, matrix[:, 0] * 2.0e-9, rtol=0.0, atol=1.0e-15)
