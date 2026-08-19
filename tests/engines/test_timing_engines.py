"""Slice-3 tests for per-PTA timing engines and validators."""

import numpy as np
import pytest

from nltiming.engine_support import (
    LinearModel,
    validate_engine_shapes,
    validate_engine_zero_delta,
)
from metapulsar.engines.jug import (
    JugEngine,
    LinearizedJugEngine,
)
from metapulsar.engines.pint import (
    LinearizedPintEngine,
    PintEngine,
)
from metapulsar.engines.delta import Tempo2DeltaEngine
from metapulsar.engines.tempo2 import (
    LibstempoEngine,
    LinearizedLibstempoEngine,
)


class _FakeDeltaEngine:
    def delta_residuals(self, delta_params):
        delta = np.array([delta_params["F0"], delta_params["F1"]], dtype=float)
        # Fitter sign: Δr ≈ -M δ
        return -(_linear_model().design @ delta)


class _StrictTempo2Engine:
    def __init__(self):
        self._reference_values = {"PB": 1.0}
        self.calls: list[dict[str, float]] = []

    def delta_residuals(self, delta_params):
        unknown = set(delta_params) - set(self._reference_values)
        if unknown:
            raise KeyError(f"unexpected native params: {unknown}")
        self.calls.append(dict(delta_params))
        # Fitter sign: Δr ≈ -M δ for the native PB column.
        return -np.array([2.0, 3.0, 5.0], dtype=float) * delta_params.get("PB", 0.0)


class _FakeLTPulsarParam:
    def __init__(self, val: float):
        self.val = val


class _FakeLTPulsarWithJump:
    def __init__(self):
        self._params = {"PB": _FakeLTPulsarParam(1.0)}

    def pars(self, which=None):
        if which == "set":
            return ["PB", "JUMP"]
        return ["PB", "JUMP"]

    def __getitem__(self, name):
        if name not in self._params:
            raise KeyError(name)
        return self._params[name]

    def residuals(self):
        return np.zeros(3, dtype=float)

    def designmatrix(self):
        return np.array(
            [
                [1.0, 10.0],
                [1.0, 11.0],
                [1.0, 13.0],
            ],
            dtype=float,
        )

    def formbats(self):
        return None


class _SpyLTPulsarParam:
    def __init__(self, val: float):
        self.val = val


class _SpyLTPulsar:
    """libstempo stand-in that records barycenter vs residual calls."""

    def __init__(self):
        self._params = {
            "PB": _SpyLTPulsarParam(1.0),
            "F0": _SpyLTPulsarParam(100.0),
        }
        self.formbats_calls = 0
        self.residuals_calls: list[dict] = []
        self._residuals = np.array([0.1, 0.2, 0.3], dtype=float)

    def pars(self, which=None):
        return ["PB", "F0"]

    def __getitem__(self, name):
        return self._params[name]

    def residuals(self, **kwargs):
        self.residuals_calls.append(dict(kwargs))
        return self._residuals.copy()

    def designmatrix(self):
        return np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 4.0, 5.0],
                [1.0, 6.0, 7.0],
            ],
            dtype=float,
        )

    def formbats(self):
        self.formbats_calls += 1


class _FakeJaxState:
    def residual_delta_np(self, delta):
        return -(_linear_model().design @ np.asarray(delta, dtype=float))

    def residual_delta_jax(self, delta):
        import jax.numpy as jnp

        return -(jnp.asarray(_linear_model().design) @ jnp.asarray(delta))

    def residual_jacobian_native(self):
        return -np.asarray(_linear_model().design, dtype=float)


def _linear_model():
    fitpars = ("F0", "F1")
    design = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )
    theta_exact = {"F0": "1234.567890123456789", "F1": "-1.0e-15"}
    return LinearModel.from_design(
        fitpars=fitpars, design=design, theta_exact=theta_exact
    )


def _assert_linear_tangent(engine):
    delta = np.array([0.2, -0.5], dtype=float)
    # Phase-gauge contract: residual_delta(δ) ≈ -M δ = residual_jacobian @ δ
    np.testing.assert_allclose(
        engine.residual_delta(delta),
        -(engine.design_matrix() @ delta),
        atol=1e-12,
    )
    if hasattr(engine, "residual_jacobian"):
        np.testing.assert_allclose(
            engine.residual_jacobian(),
            -engine.design_matrix(),
            atol=1e-12,
        )
    validate_engine_zero_delta(engine)
    validate_engine_shapes(engine)


def test_pint_engine_linear_contract():
    engine = LinearizedPintEngine.from_linear_model(_linear_model())
    _assert_linear_tangent(engine)
    assert engine.fitpars == ("F0", "F1")
    assert set(engine.reference_theta_exact()) == {"F0", "F1"}


def test_tempo2_engine_linear_contract():
    engine = LinearizedLibstempoEngine.from_linear_model(_linear_model())
    _assert_linear_tangent(engine)
    assert engine.fitpars == ("F0", "F1")


def test_jug_engine_jax_surface_and_precision_metadata():
    engine = LinearizedJugEngine.from_linear_model(
        _linear_model(),
        compatibility="tempo2",
        precision_critical=frozenset({"F0"}),
    )
    _assert_linear_tangent(engine)
    assert engine.compatibility == "tempo2"
    assert engine.precision_critical_fitpars() == frozenset({"F0"})

    jnp = __import__("jax.numpy", fromlist=["*"])
    delta = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(delta)),
        -(engine.design_matrix() @ np.asarray(delta)),
        atol=1e-12,
    )


def test_native_engines_wrap_engines_with_pulsar_metadata():
    model = _linear_model()
    engines = [
        PintEngine(engine=_FakeDeltaEngine(), linear_model=model),
        LibstempoEngine(engine=_FakeDeltaEngine(), linear_model=model),
        JugEngine(state=_FakeJaxState(), linear_model=model),
    ]
    for engine in engines:
        _assert_linear_tangent(engine)
        assert engine.reference_theta_exact()["F0"] == "1234.567890123456789"


def test_jug_engine_adds_exact_linear_to_numpy_and_jax_paths():
    model = LinearModel.from_design(
        fitpars=("PB", "Offset"),
        design=np.array(
            [
                [2.0, 1.0],
                [3.0, 1.0],
                [5.0, 1.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "Offset": "0.0"},
    )
    # JUG native Jacobian is J = -M for the evaluable column.
    native = -model.design[:, :1].copy()
    state = _FakeJaxState()
    state.design_matrix = native
    state.residual_delta_np = lambda delta: native @ np.asarray(delta, dtype=float)
    state.residual_jacobian_native = lambda: np.asarray(native, dtype=float)

    def residual_delta_jax(delta):
        import jax.numpy as jnp

        return jnp.asarray(native) @ jnp.asarray(delta)

    state.residual_delta_jax = residual_delta_jax
    engine = JugEngine(state=state, linear_model=model)
    engine._jug_indices = (0,)
    engine._jug_fitpars = ("PB",)
    engine._exact_linear_indices = (1,)
    engine._exact_linear_fitpars = frozenset({"Offset"})

    delta = np.array([0.5, -0.25], dtype=float)
    expected = -(model.design @ delta)
    np.testing.assert_allclose(engine.residual_delta(delta), expected)

    jnp = __import__("jax.numpy", fromlist=["*"])
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(jnp.asarray(delta))),
        expected,
    )
    np.testing.assert_allclose(engine.residual_jacobian(), -model.design)


def test_jug_engine_converts_astrometry_fit_units_to_native():
    """RAJ/DECJ deltas are scaled from pulsar fit units to JUG native radians.

    ``MetaPulsar.Mmat`` carries RAJ in hourangle and DECJ in degrees, while the
    frozen ``JaxTimingState`` is native (radians). Without the conversion the
    residual response is over-scaled by ``12/pi`` (RAJ) / ``180/pi`` (DECJ);
    with it, ``residual_delta == -design_matrix @ delta`` holds for every axis.
    """
    pytest.importorskip("jax")
    pytest.importorskip("jug.utils.units")
    import jax.numpy as jnp
    from jug.utils.units import native_to_fit_value

    fitpars = ("RAJ", "DECJ", "F0")
    pulsar_design = np.array(
        [
            [2.0, 1.0, 0.5],
            [3.0, -1.0, 1.0],
            [5.0, 0.5, -0.5],
            [1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    model = LinearModel.from_design(
        fitpars=fitpars,
        design=pulsar_design,
        theta_exact={"RAJ": "0.0", "DECJ": "0.0", "F0": "100.0"},
    )
    # Native J = -M_native; M_native columns are host columns * fit_per_native.
    scale = np.array([native_to_fit_value(name, 1.0) for name in fitpars])
    native_design = pulsar_design * scale
    native_J = -native_design

    class _NativeState:
        design_matrix = native_design
        fit_params = fitpars
        param_mapping = ()

        def residual_delta_np(self, delta):
            return native_J @ np.asarray(delta, dtype=float)

        def residual_delta_jax(self, delta):
            return jnp.asarray(native_J) @ jnp.asarray(delta)

        def residual_jacobian_native(self):
            return np.asarray(native_J, dtype=float)

    engine = JugEngine(state=_NativeState(), linear_model=model)
    fit_delta = np.array([7.0e-8, 5.0e-7, 1.0e-9], dtype=float)
    expected = -(pulsar_design @ fit_delta)

    np.testing.assert_allclose(engine.residual_delta(fit_delta), expected, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(jnp.asarray(fit_delta))),
        expected,
        rtol=1e-12,
    )
    # residual_jacobian is served in pulsar fit units (J = -M).
    np.testing.assert_allclose(engine.residual_jacobian(), -pulsar_design, rtol=1e-12)


def test_libstempo_engine_routes_jump_through_exact_linear_design_column():
    model = LinearModel.from_design(
        fitpars=("PB", "JUMP"),
        design=np.array(
            [
                [2.0, 10.0],
                [3.0, 11.0],
                [5.0, 13.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "JUMP": "0.0"},
    )
    strict = _StrictTempo2Engine()
    engine = LibstempoEngine(
        engine=strict,
        linear_model=model,
        native_fitpars=("PB",),
        exact_linear_fitpars=frozenset({"JUMP"}),
    )

    delta = np.array([0.25, -0.5], dtype=float)
    np.testing.assert_allclose(engine.residual_delta(delta), -(model.design @ delta))
    assert strict.calls == [{"PB": 0.25}]
    assert engine.exact_linear_fitpars() == frozenset({"JUMP"})


def test_libstempo_from_contribution_marks_unsettable_jump_exact_linear():
    model = LinearModel.from_design(
        fitpars=("PB", "JUMP"),
        design=np.array(
            [
                [2.0, 10.0],
                [3.0, 11.0],
                [5.0, 13.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "JUMP": "0.0"},
    )
    engine = LibstempoEngine.from_contribution(
        _FakeLTPulsarWithJump(), linear_model=model
    )

    assert engine.exact_linear_fitpars() == frozenset({"JUMP"})
    np.testing.assert_allclose(
        engine.residual_delta(np.array([0.0, 0.5], dtype=float)),
        -(model.design[:, 1] * 0.5),
    )


def test_infer_jug_param_mapping_fdjump_spellings():
    from metapulsar.engines.delta import infer_jug_param_mapping

    mapping = infer_jug_param_mapping(
        ["F0", "FD1JUMP1", "FDJUMPDM1"],
        {"F0", "FDJUMP1_1", "FDJUMPDM_1"},
    )
    assert mapping["FD1JUMP1"] == "FDJUMP1_1"
    assert mapping["FDJUMPDM1"] == "FDJUMPDM_1"
    assert "F0" not in mapping

    # A bare spelling is mask 1 and must never capture a later mask.
    assert infer_jug_param_mapping(["FD1JUMP2"], {"FDJUMP1"}) == {}


def test_tempo2_delta_engine_does_not_call_formbats():
    """MCMC jumps use residuals() (updateBatsAll), never formBatsAll."""
    psr = _SpyLTPulsar()
    engine = Tempo2DeltaEngine(psr)
    formbats_after_init = psr.formbats_calls
    residuals_after_init = len(psr.residuals_calls)

    delta = engine.delta_residuals({"PB": 0.01})

    assert formbats_after_init == 0
    assert psr.formbats_calls == 0
    assert len(psr.residuals_calls) == residuals_after_init + 1
    assert delta.shape == (3,)
    assert psr["PB"].val == 1.0
    assert psr["F0"].val == 100.0


def test_tempo2_delta_engine_restores_params_without_formbats_on_error():
    psr = _SpyLTPulsar()

    def _boom(**kwargs):
        psr.residuals_calls.append(dict(kwargs))
        raise RuntimeError("residual failure")

    engine = Tempo2DeltaEngine(psr)
    psr.residuals = _boom

    with pytest.raises(RuntimeError, match="residual failure"):
        engine.delta_residuals({"F0": 1e-9})

    assert psr.formbats_calls == 0
    assert psr["PB"].val == 1.0
    assert psr["F0"].val == 100.0
