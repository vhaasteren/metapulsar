"""Slice-3 tests for pulsar protocol and engine conformance helpers."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from nltiming.engine_support import (
    validate_engine_against_pulsar,
    validate_pulsar_surface,
)
from nltiming.protocols import EnterprisePulsarLike, TimingPulsar


def test_fake_pulsar_satisfies_protocol_and_shape_validators(fake_timing_pulsar):
    assert isinstance(fake_timing_pulsar, EnterprisePulsarLike)
    assert isinstance(fake_timing_pulsar, TimingPulsar)
    validate_pulsar_surface(fake_timing_pulsar)

    engine = fake_timing_pulsar.timing_engine({"tempo2": "jug", "pint": "pint"})
    assert tuple(fake_timing_pulsar.fitpars) == engine.fitpars
    validate_engine_against_pulsar(engine, fake_timing_pulsar)


def test_reference_theta_exact_roundtrip(fake_timing_pulsar):
    engine = fake_timing_pulsar.timing_engine({"tempo2": "libstempo", "pint": "jug"})
    exact = engine.reference_theta_exact()
    floats = engine.reference_theta()
    for i, name in enumerate(engine.fitpars):
        assert float(exact[name]) == floats[i]


def test_non_metapulsar_pulsar_can_conform():
    class LocalPulsar:
        name = "LOCAL"
        fitpars = ("F0", "F1")
        _toas = np.array([5.0, 1.0], dtype=float)
        _residuals = np.zeros(2, dtype=float)
        _toaerrs = np.ones(2, dtype=float)
        _freqs = np.ones(2, dtype=float) * 1400.0
        _Mmat = np.zeros((2, 2), dtype=float)
        _flags = {"pta": np.array(["x", "x"])}
        _backend_flags = np.array(["a", "a"])

        @property
        def toas(self):
            return self._toas

        @property
        def residuals(self):
            return self._residuals

        @property
        def toaerrs(self):
            return self._toaerrs

        @property
        def freqs(self):
            return self._freqs

        @property
        def Mmat(self):
            return self._Mmat

        @property
        def flags(self):
            return self._flags

        @property
        def backend_flags(self):
            return self._backend_flags

        def pint_model(self):
            return None

        def timing_engine(self, engines="jug"):
            return fake_timing_pulsar_backend

        def can_use_engines(self, engines="jug"):
            return True

        def state_id(self):
            return "local"

    class LocalBackend:
        fitpars = ("F0", "F1")
        native_units = {"F0": "native", "F1": "native"}

        def reference_theta(self):
            return np.zeros(2, dtype=float)

        def reference_theta_exact(self):
            return {"F0": "0.0", "F1": "0.0"}

        def residual_delta(self, delta_theta):
            return np.zeros(2, dtype=float)

        def design_matrix(self, params=None):
            return np.zeros((2, 2), dtype=float)

    fake_timing_pulsar_backend = LocalBackend()
    pulsar = LocalPulsar()
    assert isinstance(pulsar, TimingPulsar)
    validate_pulsar_surface(pulsar)
    validate_engine_against_pulsar(pulsar.timing_engine(), pulsar)


def test_engine_validator_rejects_design_row_mismatch(fake_timing_pulsar):
    class BadBackend:
        fitpars = tuple(fake_timing_pulsar.fitpars)
        native_units = {name: "native" for name in fitpars}

        def reference_theta(self):
            return np.zeros(len(self.fitpars), dtype=float)

        def reference_theta_exact(self):
            return {name: "0.0" for name in self.fitpars}

        def residual_delta(self, delta_theta):
            return np.zeros(len(fake_timing_pulsar.toas), dtype=float)

        def design_matrix(self, params=None):
            design = np.asarray(fake_timing_pulsar.Mmat, dtype=float).copy()
            design[0, 0] = 1.0
            return design

    with pytest.raises(ValueError, match="canonical row order"):
        validate_engine_against_pulsar(BadBackend(), fake_timing_pulsar)


def _build_real_pulsar():
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_a", seed=11
        ),
        "pta_b": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_b", seed=22
        ),
    }
    return MetaPulsar(pulsars, combination_strategy="per_pta")


def test_metapulsar_timing_pulsar_surface_and_engine_roundtrip():
    pulsar = _build_real_pulsar()
    assert isinstance(pulsar, TimingPulsar)
    validate_pulsar_surface(pulsar)

    # Native in-memory tempo2 adapters are available for tempo2-origin hosts.
    native_engines = {"tempo2": "libstempo", "pint": "jug"}
    assert pulsar.can_use_engines(native_engines)
    assert not pulsar.can_use_engines("jug")
    assert not pulsar.can_use_engines({"tempo2": "jug", "pint": "pint"})
    assert pulsar.can_use_engines(native_engines, linearized=True)
    assert pulsar.can_use_engines("jug", linearized=True)
    assert pulsar.can_use_engines({"tempo2": "jug", "pint": "pint"}, linearized=True)

    native_backend = pulsar.timing_engine(native_engines)
    assert tuple(pulsar.fitpars) == native_backend.fitpars
    validate_engine_against_pulsar(native_backend, pulsar)

    engine = pulsar.timing_engine(native_engines, linearized=True)
    assert tuple(pulsar.fitpars) == engine.fitpars
    validate_engine_against_pulsar(engine, pulsar)

    # state_id should be stable across repeated reads in unchanged state.
    assert pulsar.state_id() == pulsar.state_id()


def test_metapulsar_pint_model_and_engine_error_paths():
    pulsar = _build_real_pulsar()
    model = pulsar.pint_model()
    assert model is not None

    engine = pulsar.timing_engine({"tempo2": "jug", "pint": "pint"}, linearized=True)
    assert getattr(engine._contributions[0].engine, "engine_name") == "jug"


def test_metapulsar_reference_theta_missing_values_raise():
    pulsar = _build_real_pulsar()
    pta = next(iter(pulsar._pta_data))
    pulsar._parfile_dicts[pta] = {}
    pulsar._invalidate_timing_caches()

    with pytest.raises(ValueError, match="Missing reference theta"):
        pulsar.timing_engine({"tempo2": "libstempo", "pint": "jug"}, linearized=True)


def test_metapulsar_timing_engine_cache_tracks_pulsar_state():
    pulsar = _build_real_pulsar()
    native_engines = {"tempo2": "libstempo", "pint": "jug"}
    engine = pulsar.timing_engine(native_engines, linearized=True)
    token = pulsar.state_id()

    pulsar._designmatrix = pulsar._designmatrix.copy()
    pulsar._designmatrix[0, 0] += 1.0
    changed = pulsar.timing_engine(native_engines, linearized=True)

    assert pulsar.state_id() != token
    assert changed is not engine


def test_timing_engine_accepts_nonlinear_params_kwarg():
    assert "nonlinear_params" in inspect.signature(MetaPulsar.timing_engine).parameters
    assert (
        "nonlinear_params"
        in inspect.signature(MetaPulsar._build_jug_session).parameters
    )


def test_build_jug_session_forwards_nonlinear_params(monkeypatch, tmp_path):
    captured: dict = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("jug.engine.session.TimingSession", FakeSession)

    par = tmp_path / "a.par"
    tim = tmp_path / "a.tim"
    par.write_text("PSRJ J0000+0000\n")
    tim.write_text("FORMAT 1\n")

    mp = MetaPulsar.__new__(MetaPulsar)
    mp._clock_dir = None
    mp._pta_files = {"epta": SimpleNamespace(par_path=par, tim_path=tim)}
    mp._ensure_clock_aliases = lambda: None  # type: ignore[method-assign]
    mp._pta_files_available = lambda _name: True  # type: ignore[method-assign]

    MetaPulsar._build_jug_session(mp, "epta", "pint", nonlinear_params="binary+")
    assert captured["nonlinear_params"] == "binary+"


def test_metapulsar_timing_opens_nltiming_evaluator():
    from nltiming import TimingEvaluator

    pulsar = _build_real_pulsar()
    timing = pulsar.timing(
        {"tempo2": "libstempo", "pint": "jug"},
        linearized=True,
    )

    assert isinstance(timing, TimingEvaluator)
    assert timing.pulsar is pulsar
    assert timing.parameters.names == tuple(pulsar.fitpars)
    assert timing.reference_exact == timing.engine.reference_theta_exact()
    assert timing.pulsar.timing_parameter_mapping() == {
        name: dict(pulsar._fitparameters[name]) for name in pulsar.fitpars
    }


def test_filtered_pulsar_rejects_unaligned_live_timing_contributions():
    pulsar = _build_real_pulsar()
    pulsar.filter_data(mask=np.arange(len(pulsar.toas)) % 2 == 0)

    assert not pulsar.can_use_engines(
        {"tempo2": "libstempo", "pint": "jug"}, linearized=True
    )
    with pytest.raises(ValueError, match="after filter_data"):
        pulsar.timing(
            {"tempo2": "libstempo", "pint": "jug"},
            linearized=True,
        )
    assert pulsar.Mmat.shape[1] == len(pulsar.fitpars)


def test_libstempo_engine_uses_xdot_param_mapping_from_parameter_manager():
    """LibstempoEngine should accept A1DOT->XDOT mapping built by ParameterManager."""
    from metapulsar.mockpulsar import MockParameter
    from metapulsar.parameter_manager import ParameterManager
    from metapulsar.engines import LibstempoEngine
    from metapulsar.engines.delta import Tempo2DeltaEngine
    from nltiming.engine_support import LinearModel

    file_data = {
        "epta": {
            "timing_package": "tempo2",
            "par_content": (
                "PSRJ J1640+2224\n"
                "F0 316.12397933185408713 1 0\n"
                "PEPOCH 55000\n"
                "DM 18.417 1 0\n"
                "BINARY T2\n"
                "PB 175.46066459623014253 1 0\n"
                "T0 51626.179967495799449 1 0\n"
                "A1 55.329722354525327725 1 0\n"
                "OM 50.733505043065199373 1 0\n"
                "ECC 0.0007972975541058369088 1 0\n"
                "XDOT 8.1279761448223669144e-15 1 0\n"
                "EPHEM DE421\n"
                "CLK TT(BIPM2011)\n"
            ),
        }
    }
    pm = ParameterManager(file_data=file_data, combine_components=["binary"])
    mapping = pm.build_parameter_mappings()
    param_mapping = {"A1DOT": mapping.fitparameters["A1DOT"]["epta"]}
    assert param_mapping == {"A1DOT": "XDOT"}

    class XdotPulsar:
        def __init__(self):
            self._fitpars = ["XDOT", "F0"]
            self._setpars = ["XDOT", "F0", "PB"]
            self._params = {
                "XDOT": MockParameter(8.1279761448223669144e-15),
                "F0": MockParameter(316.12397933185408713),
            }
            self._residuals = np.zeros(4)
            self._design = np.array(
                [[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [1.0, 0.3]], dtype=float
            )

        def pars(self, which="fit"):
            return tuple(self._fitpars if which == "fit" else self._setpars)

        def __getitem__(self, name):
            return self._params[name]

        def formbats(self):
            pass

        def residuals(self):
            return self._residuals.copy()

        def designmatrix(self):
            return self._design.copy()

    psr = XdotPulsar()
    model = LinearModel.from_design(
        fitpars=("A1DOT", "F0"),
        design=np.array([[0.5, 0.0], [0.5, 0.1], [0.5, 0.2], [0.5, 0.3]], dtype=float),
        theta_exact={
            "A1DOT": "8.1279761448223669144e-15",
            "F0": "316.12397933185408713",
        },
    )
    engine = LibstempoEngine.from_contribution(
        psr, linear_model=model, param_mapping=param_mapping
    )
    np.testing.assert_allclose(
        engine.residual_delta(np.zeros(2, dtype=float)), np.zeros(4)
    )

    engine = Tempo2DeltaEngine(psr)
    with pytest.raises(KeyError, match="A1DOT"):
        engine.delta_residuals({"A1DOT": 1e-18})
