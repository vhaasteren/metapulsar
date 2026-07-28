"""Tests for native nonlinear timing engine wiring."""

from __future__ import annotations

import numpy as np
import pytest

from metapulsar.metapulsar import MetaPulsar, PtaFiles
from metapulsar.metapulsar_factory import MetaPulsarFactory
from metapulsar.engines import JugEngine, LibstempoEngine, PintEngine
from nltiming.engine_support import LinearModel
from nltiming.protocols import JaxTimingEngine


class _FakeJaxState:
    def __init__(self, design):
        self.design = np.asarray(design, dtype=float)

    def residual_delta_np(self, delta):
        return self.design @ np.asarray(delta, dtype=float)

    def residual_delta_jax(self, delta):
        import jax.numpy as jnp

        return jnp.asarray(self.design) @ jnp.asarray(delta)


class _FakeDeltaEngine:
    def __init__(self, design):
        self.design = np.asarray(design, dtype=float)
        self.calls = []

    def delta_residuals(self, delta_params):
        self.calls.append(dict(delta_params))
        delta = np.asarray([delta_params["F0"], delta_params["F1"]], dtype=float)
        return self.design @ delta


def _linear_model():
    design = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, -0.5]], dtype=float)
    return LinearModel.from_design(
        fitpars=("F0", "F1"),
        design=design,
        theta_exact={"F0": "10.0", "F1": "-1e-15"},
    )


def test_jug_engine_is_jax_traceable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    model = _linear_model()
    engine = JugEngine(
        state=_FakeJaxState(model.design),
        linear_model=model,
        precision_critical={"F0"},
    )

    assert isinstance(engine, JaxTimingEngine)
    assert engine.precision_critical_fitpars() == {"F0"}
    np.testing.assert_allclose(engine.design_matrix(), model.design)
    np.testing.assert_allclose(
        engine.residual_delta(np.array([0.2, -0.4])), [0.2, 0.0, 0.4]
    )

    def scalar(delta):
        return jnp.sum(engine.residual_delta_jax(delta) ** 2)

    value = jax.jit(scalar)(jnp.array([0.2, -0.4]))
    grad = jax.grad(scalar)(jnp.array([0.2, -0.4]))
    assert np.isfinite(np.asarray(value))
    assert grad.shape == (2,)


def test_native_pint_and_tempo2_wrappers_use_linear_model_metadata():
    model = _linear_model()
    pint = PintEngine(engine=_FakeDeltaEngine(model.design), linear_model=model)
    tempo2 = LibstempoEngine(engine=_FakeDeltaEngine(model.design), linear_model=model)

    for engine in (pint, tempo2):
        np.testing.assert_allclose(engine.design_matrix(), model.design)
        assert engine.reference_theta_exact()["F0"] == "10.0"
        np.testing.assert_allclose(
            engine.residual_delta(np.array([0.2, -0.4])),
            np.array([0.2, 0.0, 0.4]),
        )


def test_factory_retains_exact_session_file_bytes(tmp_path):
    par = tmp_path / "loaded.par"
    tim = tmp_path / "loaded.tim"
    par.write_text("F0 123\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")

    retained = MetaPulsarFactory()._retain_pta_files(
        pta_name="ng 5",
        timing_package="pint",
        par_path=par,
        tim_path=tim,
        pta_file_dir=tmp_path / "retained",
    )

    assert retained["par_path"].read_text(encoding="utf-8") == "F0 123\n"
    assert retained["tim_path"].read_text(encoding="utf-8") == "FORMAT 1\n"
    assert retained["timing_package"] == "pint"


def test_factory_retains_included_tim_files(tmp_path):
    chunk = tmp_path / "tims" / "chunk.tim"
    chunk.parent.mkdir(parents=True)
    chunk.write_text("FORMAT 1\n obs 1400.0 55000.0 1.0 site\n", encoding="utf-8")
    main = tmp_path / "main.tim"
    main.write_text("FORMAT 1\nINCLUDE tims/chunk.tim\n", encoding="utf-8")
    par = tmp_path / "main.par"
    par.write_text("F0 1\n", encoding="utf-8")

    retained = MetaPulsarFactory()._retain_pta_files(
        pta_name="epta",
        timing_package="tempo2",
        par_path=par,
        tim_path=main,
        pta_file_dir=tmp_path / "retained",
    )

    included = tmp_path / "retained" / "tims" / "chunk.tim"
    assert included.is_file()
    assert included.read_text(encoding="utf-8") == chunk.read_text(encoding="utf-8")
    assert "INCLUDE tims/chunk.tim" in retained["tim_path"].read_text(encoding="utf-8")


def test_jug_capability_requires_readable_pta_files(monkeypatch, tmp_path):
    pulsar = MetaPulsar.__new__(MetaPulsar)

    class _Record:
        timing_package = "pint"

    pulsar._pta_data = {"pta": _Record()}
    pulsar._clock_dir = None
    monkeypatch.setattr(MetaPulsar, "_can_import_jug", staticmethod(lambda: True))

    pulsar._pta_files = {}
    assert not pulsar.can_use_engines("jug")

    par = tmp_path / "session.par"
    tim = tmp_path / "session.tim"
    par.write_text("F0 1\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")
    pulsar._pta_files = {
        "pta": PtaFiles(par_path=par, tim_path=tim, timing_package="pint")
    }
    assert pulsar.can_use_engines("jug")
