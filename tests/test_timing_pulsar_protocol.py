"""Slice-3 tests for host protocol and backend conformance helpers."""

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar
from metapulsar.mockpulsar import create_mock_libstempo
from nltiming.backends.base import (
    validate_backend_against_pulsar,
    validate_enterprise_pulsar,
)
from nltiming.protocols import EnterprisePulsarLike, PulsarInterface


def test_fake_host_satisfies_protocol_and_shape_validators(fake_pulsar_interface):
    assert isinstance(fake_pulsar_interface, EnterprisePulsarLike)
    assert isinstance(fake_pulsar_interface, PulsarInterface)
    validate_enterprise_pulsar(fake_pulsar_interface)

    backend = fake_pulsar_interface.timing_backend({"tempo2": "jug", "pint": "pint"})
    assert tuple(fake_pulsar_interface.fitpars) == backend.fitpars
    validate_backend_against_pulsar(backend, fake_pulsar_interface)


def test_reference_theta_exact_roundtrip(fake_pulsar_interface):
    backend = fake_pulsar_interface.timing_backend(
        {"tempo2": "libstempo", "pint": "jug"}
    )
    exact = backend.reference_theta_exact()
    floats = backend.reference_theta()
    for i, name in enumerate(backend.fitpars):
        assert float(exact[name]) == floats[i]


def test_non_metapulsar_host_can_conform():
    class LocalHost:
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

        def timing_backend(self, engines="jug"):
            return fake_pulsar_interface_backend

        def can_use_engines(self, engines="jug"):
            return True

        def cache_token(self):
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

    fake_pulsar_interface_backend = LocalBackend()
    host = LocalHost()
    assert isinstance(host, PulsarInterface)
    validate_enterprise_pulsar(host)
    validate_backend_against_pulsar(host.timing_backend(), host)


def test_backend_validator_rejects_design_row_mismatch(fake_pulsar_interface):
    class BadBackend:
        fitpars = tuple(fake_pulsar_interface.fitpars)
        native_units = {name: "native" for name in fitpars}

        def reference_theta(self):
            return np.zeros(len(self.fitpars), dtype=float)

        def reference_theta_exact(self):
            return {name: "0.0" for name in self.fitpars}

        def residual_delta(self, delta_theta):
            return np.zeros(len(fake_pulsar_interface.toas), dtype=float)

        def design_matrix(self, params=None):
            design = np.asarray(fake_pulsar_interface.Mmat, dtype=float).copy()
            design[0, 0] = 1.0
            return design

    with pytest.raises(ValueError, match="canonical row order"):
        validate_backend_against_pulsar(BadBackend(), fake_pulsar_interface)


def _build_real_host():
    pulsars = {
        "pta_a": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_a", seed=11
        ),
        "pta_b": create_mock_libstempo(
            n_toas=20, name="J1857+0943", telescope="pta_b", seed=22
        ),
    }
    return MetaPulsar(pulsars, combination_strategy="composite")


def test_metapulsar_timing_host_surface_and_backend_roundtrip():
    host = _build_real_host()
    assert isinstance(host, PulsarInterface)
    validate_enterprise_pulsar(host)

    # Native in-memory tempo2 adapters are available for tempo2-origin hosts.
    native_engines = {"tempo2": "libstempo", "pint": "jug"}
    assert host.can_use_engines(native_engines)
    assert not host.can_use_engines("jug")
    assert not host.can_use_engines({"tempo2": "jug", "pint": "pint"})
    assert host.can_use_engines(native_engines, linearized=True)
    assert host.can_use_engines("jug", linearized=True)
    assert host.can_use_engines({"tempo2": "jug", "pint": "pint"}, linearized=True)

    native_backend = host.timing_backend(native_engines)
    assert tuple(host.fitpars) == native_backend.fitpars
    validate_backend_against_pulsar(native_backend, host)

    backend = host.timing_backend(native_engines, linearized=True)
    assert tuple(host.fitpars) == backend.fitpars
    validate_backend_against_pulsar(backend, host)

    # cache_token should be stable across repeated reads in unchanged state.
    assert host.cache_token() == host.cache_token()


def test_metapulsar_pint_model_and_backend_error_paths():
    host = _build_real_host()
    model = host.pint_model()
    assert model is not None

    backend = host.timing_backend({"tempo2": "jug", "pint": "pint"}, linearized=True)
    assert getattr(backend._sessions[0].backend, "backend_name") == "jug"


def test_metapulsar_reference_theta_missing_values_raise():
    host = _build_real_host()
    pta = next(iter(host._epulsars))
    host._parfile_dicts[pta] = {}
    host._invalidate_timing_caches()

    with pytest.raises(ValueError, match="Missing reference theta"):
        host.timing_backend({"tempo2": "libstempo", "pint": "jug"}, linearized=True)


def test_metapulsar_timing_backend_cache_tracks_host_state():
    host = _build_real_host()
    native_engines = {"tempo2": "libstempo", "pint": "jug"}
    backend = host.timing_backend(native_engines, linearized=True)
    token = host.cache_token()

    host._designmatrix = host._designmatrix.copy()
    host._designmatrix[0, 0] += 1.0
    changed = host.timing_backend(native_engines, linearized=True)

    assert host.cache_token() != token
    assert changed is not backend


def test_metapulsar_timing_opens_nltiming_evaluator():
    from nltiming import TimingEvaluator

    host = _build_real_host()
    timing = host.timing(
        {"tempo2": "libstempo", "pint": "jug"},
        linearized=True,
    )

    assert isinstance(timing, TimingEvaluator)
    assert timing.pulsar is host
    assert timing.parameters.names == tuple(host.fitpars)
    assert timing.reference_exact == timing.backend.reference_theta_exact()
    assert timing.pulsar.timing_parameter_mapping() == {
        name: dict(host._fitparameters[name]) for name in host.fitpars
    }


def test_filtered_host_rejects_unaligned_live_timing_sessions():
    host = _build_real_host()
    host.filter_data(mask=np.arange(len(host.toas)) % 2 == 0)

    assert not host.can_use_engines(
        {"tempo2": "libstempo", "pint": "jug"}, linearized=True
    )
    with pytest.raises(ValueError, match="after filter_data"):
        host.timing(
            {"tempo2": "libstempo", "pint": "jug"},
            linearized=True,
        )
    assert host.Mmat.shape[1] == len(host.fitpars)


def test_libstempo_engine_uses_xdot_param_mapping_from_parameter_manager():
    """LibstempoEngine should accept A1DOT->XDOT mapping built by ParameterManager."""
    from metapulsar.mockpulsar import MockParameter
    from metapulsar.parameter_manager import ParameterManager
    from nltiming.backends.base import LinearModel
    from nltiming.backends.engines import Tempo2DeltaEngine
    from nltiming.backends.tempo2 import LibstempoEngine

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
    model = LinearModel.from_host(
        fitpars=("A1DOT", "F0"),
        design=np.array([[0.5, 0.0], [0.5, 0.1], [0.5, 0.2], [0.5, 0.3]], dtype=float),
        theta_exact={
            "A1DOT": "8.1279761448223669144e-15",
            "F0": "316.12397933185408713",
        },
    )
    backend = LibstempoEngine.from_session(
        psr, linear_model=model, param_mapping=param_mapping
    )
    np.testing.assert_allclose(
        backend.residual_delta(np.zeros(2, dtype=float)), np.zeros(4)
    )

    engine = Tempo2DeltaEngine(psr)
    with pytest.raises(KeyError, match="A1DOT"):
        engine.delta_residuals({"A1DOT": 1e-18})
